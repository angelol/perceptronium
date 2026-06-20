import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Set matplotlib backend to non-GUI to avoid errors on headless runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CatsAndDogsDataset
from model import CustomCNN

def set_seed(seed):
    """Enforces determinism across python, numpy, and PyTorch CPU/GPU backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(image_size):
    """Defines high-generalization training (highly augmented) and test image transformations."""
    train_transform = transforms.Compose([
        # Scale-invariant crop to prevent position-memorization overfitting
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss incorporating Target Label Smoothing.
    Pushes target boundaries from [0, 1] towards [0.025, 0.975] to prevent logit saturation.
    """
    def __init__(self, smoothing=0.05):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, smoothed_targets)


def mixup_data(x, y, alpha=0.2, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda for Mixup."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda for CutMix."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    y_a, y_b = y, y[index]
    
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    return mixed_x, y_a, y_b, lam


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, mixup_prob=0.3, cutmix_prob=0.3):
    """Executes a single epoch of training with optional Mixup and CutMix batch-mixing."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1) # Reshape to (batch_size, 1)
        
        # Decide probabilistically which mix augmentation to apply
        r = random.random()
        if r < mixup_prob:
            mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2, device=device)
            mixed = True
        elif r < (mixup_prob + cutmix_prob):
            mixed_inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=1.0, device=device)
            mixed = True
        else:
            mixed = False
            
        optimizer.zero_grad()
        
        if mixed:
            logits = model(mixed_inputs)
            loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
        else:
            logits = model(inputs)
            loss = criterion(logits, labels)
            
        loss.backward()
        optimizer.step()
        
        # Step OneCycleLR per batch
        if scheduler is not None:
            scheduler.step()
            
        # Accumulate metrics
        running_loss += loss.item() * inputs.size(0)
        
        # Approximate training accuracy (keeps execution extremely fast with no redundant forward pass)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            if mixed:
                dominant_labels = labels_a if lam >= 0.5 else labels_b
                correct += (preds == dominant_labels).sum().item()
            else:
                correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_tta=True):
    """Evaluates the model on validation/testing set with optional Horizontal-Flip TTA."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        # Standard forward pass
        logits = model(inputs)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * inputs.size(0)
        
        if use_tta:
            # Horizontally flipped images: flip width dimension 3 of [B, C, H, W]
            inputs_flipped = torch.flip(inputs, dims=[3])
            logits_flipped = model(inputs_flipped)
            probs = 0.5 * (torch.sigmoid(logits) + torch.sigmoid(logits_flipped))
        else:
            probs = torch.sigmoid(logits)
            
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="PyTorch CBAM-EfficientNet - Cats vs Dogs (Option C)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (20 recommended for super-convergence)")
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Max learning rate for OneCycleLR")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="L2 regularization / weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution (224x224 standard)")
    parser.add_argument("--data-dir", type=str, default="data/PetImages", help="Path to Cats & Dogs folder")
    parser.add_argument("--weights-path", type=str, default="pytorch_cnn/model.pth", help="Path to save weights")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 1. Device Selection (MPS preferred on Apple Silicon Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using hardware-accelerated device: METAL GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using hardware-accelerated device: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU device (slow)")
        
    if not os.path.exists(args.data_dir) and os.path.exists("../data/PetImages"):
        args.data_dir = "../data/PetImages"
        
    print(f"Dataset target folder: {args.data_dir}")
    
    # 2. Setup Data Loading & Splits
    train_transform, test_transform = get_transforms(args.image_size)
    
    train_dataset = CatsAndDogsDataset(
        root_dir=args.data_dir,
        split="train",
        image_size=args.image_size,
        transform=train_transform,
        limit_train_per_class=4000,
        limit_test_per_class=1000
    )
    
    test_dataset = CatsAndDogsDataset(
        root_dir=args.data_dir,
        split="test",
        image_size=args.image_size,
        transform=test_transform,
        limit_train_per_class=4000,
        limit_test_per_class=1000
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Model, Loss, Optimizer, and LR Scheduler
    model = CustomCNN().to(device)
    
    # Using Label Smoothing BCE loss to improve final boundary calibration
    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # OneCycleLR scheduler for Super-Convergence (stepped per-batch)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
        anneal_strategy='cos'
    )
    
    print("\n---------------------- MODEL HYPERPARAMETERS ----------------------")
    print(f"  • Device:          {device.type.upper()}")
    print(f"  • Resolution:      {args.image_size} x {args.image_size} (RGB)")
    print(f"  • Model Params:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • Batch Size:      {args.batch_size}")
    print(f"  • Max Learning LR: {args.lr}")
    print(f"  • Weight Decay:    {args.weight_decay}")
    print(f"  • Seed:            {args.seed}")
    print(f"  • Epochs:          {args.epochs}")
    print("-------------------------------------------------------------------\n")
    
    # Metric tracking
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }
    best_test_acc = 0.0
    
    # Table header
    print("+" + "-"*96 + "+")
    print("| Epoch    | Train Loss | Train Acc | Test Loss  | Test Acc   | Learning Rate | Epoch Time (sec) |")
    print("+" + "-"*96 + "+")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Get learning rate at start of epoch
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            scheduler=scheduler, mixup_prob=0.3, cutmix_prob=0.3
        )
        
        # Evaluate with Test-Time Augmentation (TTA)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_tta=True)
        
        elapsed = time.time() - start_time
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        print(f"| Epoch {epoch:02d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_loss:10.4f} | {test_acc:9.2f}% | {current_lr:13.6f} | {elapsed:15.1f}s |")
        
        # Save weights of best epoch
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), args.weights_path)
            
    print("+" + "-"*96 + "+")
    print(f"✓ Training complete! Best Validation Accuracy achieved (with TTA): {best_test_acc:.2f}%")
    print(f"✓ Model weights saved to: {args.weights_path}")
    
    # 4. Save Learning Curve Plots
    epochs_range = range(1, args.epochs + 1)
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss (Mixed)", color="#ff7f0e", lw=2)
    plt.plot(epochs_range, history["test_loss"], label="Test Loss (Clean)", color="#1f77b4", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Testing Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Accuracy (Approx)", color="#2ca02c", lw=2)
    plt.plot(epochs_range, history["test_acc"], label="Test Accuracy (with TTA)", color="#d62728", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Testing Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plot_path = "pytorch_cnn/learning_curves.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✓ Learning curve plots saved to: {plot_path}")

if __name__ == "__main__":
    main()
