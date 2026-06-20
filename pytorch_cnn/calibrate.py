import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

# Set matplotlib backend to non-GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CatsAndDogsDataset
from model import CustomCNN

class TemperatureScaler(nn.Module):
    """
    Optimizes a single temperature parameter T > 0 to calibrate binary classifier logits.
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature


def compute_binary_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for binary classification.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # Include boundaries appropriately
        if i == 0:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            
        count = in_bin.sum().item()
        bin_counts.append(count)
        
        if count > 0:
            # For cats vs dogs (binary), confidence of a prediction is max(prob, 1 - prob)
            # but standard calibration plots look at the probability of predicting Dog (class 1).
            # To measure actual probability calibration, we map directly: prob of predicting Dog vs actual proportion of Dogs.
            accuracy_in_bin = labels[in_bin].float().mean().item()
            avg_confidence_in_bin = probs[in_bin].mean().item()
            
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            
            prop_in_bin = count / len(probs)
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
        else:
            bin_accs.append(np.nan)
            bin_confs.append((bin_lower + bin_upper).item() / 2.0)

    return ece, bin_confs, bin_accs, bin_counts


def plot_reliability_diagram(confs_before, accs_before, ece_before, confs_after, accs_after, ece_after, save_path):
    """
    Generates a comparative reliability diagram before and after temperature scaling calibration.
    """
    plt.figure(figsize=(12, 5.5))
    
    # Perfect calibration diagonal
    x = np.linspace(0, 1, 100)
    
    # 1. Before Calibration
    plt.subplot(1, 2, 1)
    plt.plot(x, x, linestyle="--", color="gray", label="Perfect Calibration")
    # Clean up NaNs for plotting
    valid_idx = ~np.isnan(accs_before)
    plt.bar(np.array(confs_before)[valid_idx], np.array(accs_before)[valid_idx], width=0.08, 
            alpha=0.65, color="#e74c3c", edgecolor="black", label="Model Predict Probs")
    plt.scatter(np.array(confs_before)[valid_idx], np.array(accs_before)[valid_idx], color="#c0392b", zorder=3)
    plt.xlabel("Average Predicted Probability of Class 'Dog'")
    plt.ylabel("Actual Fraction of 'Dog' Class")
    plt.title(f"Before Calibration (ECE: {ece_before * 100.0:.2f}%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    
    # 2. After Calibration
    plt.subplot(1, 2, 2)
    plt.plot(x, x, linestyle="--", color="gray", label="Perfect Calibration")
    valid_idx_after = ~np.isnan(accs_after)
    plt.bar(np.array(confs_after)[valid_idx_after], np.array(accs_after)[valid_idx_after], width=0.08, 
            alpha=0.65, color="#2ecc71", edgecolor="black", label="Calibrated Predict Probs")
    plt.scatter(np.array(confs_after)[valid_idx_after], np.array(accs_after)[valid_idx_after], color="#27ae60", zorder=3)
    plt.xlabel("Average Predicted Probability of Class 'Dog'")
    plt.ylabel("Actual Fraction of 'Dog' Class")
    plt.title(f"After Calibration (ECE: {ece_after * 100.0:.2f}%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📊 Calibration reliability plot saved to: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Temperature Scaling Calibration for CBAM-EfficientNet")
    parser.add_argument("--weights-path", type=str, default="pytorch_cnn/model.pth", help="Path to trained weights")
    parser.add_argument("--data-dir", type=str, default="data/PetImages", help="Path to Cats & Dogs folder")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution")
    parser.add_argument("--batch-size", type=int, default=64, help="Loader batch size")
    parser.add_argument("--attention-type", type=str, default="se", choices=["se", "cbam"], help="Attention type in blocks")
    args = parser.parse_args()
    
    # 1. Device Selection
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"⚙️ Running calibration optimization on device: {device.type.upper()}")
    
    if not os.path.exists(args.weights_path) and os.path.exists("pytorch_cnn/model_swa.pth"):
        args.weights_path = "pytorch_cnn/model_swa.pth"
        print(f"🔄 Selected SWA weights for calibration: {args.weights_path}")
    elif not os.path.exists(args.weights_path) and os.path.exists("model.pth"):
        args.weights_path = "model.pth"
        
    if not os.path.exists(args.weights_path):
        print(f"❌ Error: Weights path {args.weights_path} not found.")
        sys.exit(1)
        
    # 2. Setup Validation / Test Data
    _, test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CatsAndDogsDataset(
        root_dir=args.data_dir,
        split="test",
        image_size=args.image_size,
        transform=test_transform,
        limit_train_per_class=4000,
        limit_test_per_class=1000
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Load Model
    model = CustomCNN(attention_type=args.attention_type).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()
    
    print("🔬 Extracting logits from test/validation set...")
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            
    all_logits = torch.cat(all_logits, dim=0).squeeze() # shape: [N]
    all_labels = torch.cat(all_labels, dim=0) # shape: [N]
    
    # 4. Compute ECE and stats before calibration
    probs_before = torch.sigmoid(all_logits)
    ece_before, confs_before, accs_before, _ = compute_binary_ece(probs_before, all_labels)
    print(f"📊 Initial Expected Calibration Error (ECE): {ece_before * 100.0:.4f}%")
    
    # 5. Optimize Temperature
    scaler = TemperatureScaler()
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=100)
    criterion = nn.BCEWithLogitsLoss()
    
    # We want to optimize T on the logits
    def eval_closure():
        optimizer.zero_grad()
        scaled_logits = scaler(all_logits)
        loss = criterion(scaled_logits, all_labels.float())
        loss.backward()
        return loss
        
    print("⚙️ Optimizing scaling temperature T via L-BFGS...")
    optimizer.step(eval_closure)
    
    T_optimal = scaler.temperature.item()
    print(f"🌟 Optimal Temperature parameter T: {T_optimal:.6f}")
    
    # 6. Compute ECE after calibration
    with torch.no_grad():
        calibrated_logits = scaler(all_logits)
        probs_after = torch.sigmoid(calibrated_logits)
        
    ece_after, confs_after, accs_after, _ = compute_binary_ece(probs_after, all_labels)
    print(f"🌟 Calibrated Expected Calibration Error (ECE): {ece_after * 100.0:.4f}%")
    
    # 7. Plot and Save Reliability Curves
    save_dir = os.path.dirname(args.weights_path)
    plot_path = os.path.join(save_dir, "calibration_reliability.png")
    plot_reliability_diagram(confs_before, accs_before, ece_before, 
                             confs_after, accs_after, ece_after, plot_path)
    
    # 8. Save Temperature scaling info
    calibration_info_path = os.path.join(save_dir, "temperature.txt")
    with open(calibration_info_path, "w") as f:
        f.write(f"{T_optimal:.6f}\n")
    print(f"✓ Saved optimal temperature value to: {calibration_info_path}")


if __name__ == "__main__":
    main()
