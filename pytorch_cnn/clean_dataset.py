import os
import sys
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader

# Ensure imports work from project root or pytorch_cnn subdirectory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset import CatsAndDogsDataset
from model import CustomCNN

@torch.no_grad()
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🚀 Running data cleaner on: {device.type.upper()}", flush=True)
    
    data_dir = "data/PetImages"
    if not os.path.exists(data_dir) and os.path.exists("../data/PetImages"):
        data_dir = "../data/PetImages"
    print(f"Using dataset directory: {data_dir}", flush=True)
    
    weights_path = "pytorch_cnn/model_swa.pth"
    if not os.path.exists(weights_path) and os.path.exists("model_swa.pth"):
        weights_path = "model_swa.pth"
    print(f"Loading weights from: {weights_path}", flush=True)
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: SWA checkpoint not found at {weights_path}.", flush=True)
        print("Please train a model first, or verify your weights path.", flush=True)
        sys.exit(1)
        
    # 1. Initialize dataset with transform=None (applies standard Resize, ToTensor, and Normalize)
    train_dataset = CatsAndDogsDataset(
        root_dir=data_dir,
        split="train",
        image_size=224,
        transform=None,
        limit_train_per_class=4000
    )
    
    # 2. Instantiate and load model
    model = CustomCNN(attention_type="se") # SWA weights were trained with se attention
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    flagged_corrupt = []
    flagged_mislabeled = []
    threshold = 0.85
    
    print("🔍 Scanning training set for corrupt files and mislabels...", flush=True)
    sample_idx = 0
    
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        logits = model(inputs)
        probs = torch.sigmoid(logits)
        
        # Calculate absolute error
        errors = torch.abs(labels - probs).cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        for i in range(inputs.size(0)):
            path = train_dataset.paths[sample_idx]
            label = train_dataset.labels[sample_idx]
            prob = probs_np[i][0]
            error = errors[i][0]
            
            # 1. Double check PIL full decode to detect hidden corruption or truncation
            try:
                with Image.open(path) as img:
                    img.verify() # Verify file structure integrity
            except Exception as e:
                print(f"  • Flagged corrupted image: {os.path.basename(path)} - Error: {e}", flush=True)
                flagged_corrupt.append(path)
                sample_idx += 1
                continue
                
            # 2. If error is higher than threshold, the model is highly confident the label is wrong
            if error > threshold:
                pred_label = "Dog" if prob >= 0.5 else "Cat"
                orig_label = "Dog" if label == 1.0 else "Cat"
                print(f"  • Flagged mislabel: {os.path.basename(path)} | Label: {orig_label} | Pred Prob: {prob:.4f} | Error: {error:.4f}", flush=True)
                flagged_mislabeled.append({
                    "path": path,
                    "original_label": orig_label,
                    "predicted_label": pred_label,
                    "predicted_prob_dog": float(prob),
                    "error_score": float(error)
                })
            sample_idx += 1
            
    print(f"\n✨ Scan Complete!", flush=True)
    print(f"  • Flagged Corrupt/Broken: {len(flagged_corrupt)} files", flush=True)
    print(f"  • Flagged Mislabeled Outliers (Error > {threshold}): {len(flagged_mislabeled)} files", flush=True)
    
    # Extract set of flagged paths for subtraction
    flagged_paths = set(flagged_corrupt + [m["path"] for m in flagged_mislabeled])
    clean_paths = [p for p in train_dataset.paths if p not in flagged_paths]
    
    output_metadata = {
        "corrupt_files": flagged_corrupt,
        "mislabeled_files": flagged_mislabeled,
        "clean_train_paths": clean_paths
    }
    
    # Save programmatically to a JSON file
    metadata_out_path = "pytorch_cnn/cleaned_dataset_metadata.json"
    with open(metadata_out_path, "w") as f:
        json.dump(output_metadata, f, indent=4)
        
    print(f"✓ Cleaned dataset metadata successfully exported to: '{metadata_out_path}'", flush=True)
    print(f"✓ Training set prunes: {len(train_dataset.paths)} -> {len(clean_paths)} clean images ({len(flagged_paths)} removed)", flush=True)

if __name__ == "__main__":
    main()
