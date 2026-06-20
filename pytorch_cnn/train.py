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


class Lookahead:
    """
    Lookahead Optimizer Wrapper (Zhang et al., 2019).
    Wraps an existing PyTorch optimizer to maintain 'slow weights' for improved generalization.
    This wrapper inherits from object and delegates attribute/method requests to the underlying
    optimizer via __getattr__, bypassing internal PyTorch 2.x hook requirements during .step().
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = self.optimizer.defaults
        self.lk_counter = 0
        
        # Save slow weights in an isolated dictionary to avoid polluting self.optimizer.state[p],
        # which would cause AdamW to skip initializing its momentum buffers ('exp_avg', 'exp_avg_sq').
        self.slow_weights = {}
        for group in self.param_groups:
            for p in group["params"]:
                self.slow_weights[p] = p.data.clone()

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.lk_counter += 1
        
        if self.lk_counter % self.k == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    slow_p = self.slow_weights[p]
                    # Interpolation: slow_p = slow_p + alpha * (fast_p - slow_p)
                    slow_p.add_(self.alpha * (p.data - slow_p))
                    # Sync fast weights to slow weights
                    p.data.copy_(slow_p)
                    
        return loss

    def __getattr__(self, item):
        return getattr(self.optimizer, item)


def get_transforms(image_size):
    """Defines high-generalization training (highly augmented) and test image transformations."""
    train_transform = transforms.Compose([
        # Allow crops down to 20% of the image area to force local feature extraction
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), ratio=(3./4., 4./3.)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Apply Auto-tuned Wide Augmentations (Rotation, Shear, Solarize, Posterize, etc.)
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        # Randomly erase small rectangular regions (acting as single-image CutMix)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random'),
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


def mixup_data(x, y, alpha=0.4, device='cpu'):
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
    """Executes a single epoch of training with optional Mixup and CutMix batch-mixing and decoupled label smoothing."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1) # Reshape to (batch_size, 1)
        
        # Decide probabilistically which mix augmentation to apply
        mixed = False
        if mixup_prob > 0.0 or cutmix_prob > 0.0:
            r = random.random()
            if r < mixup_prob:
                mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4, device=device)
                mixed = True
            elif r < (mixup_prob + cutmix_prob):
                mixed_inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=1.0, device=device)
                mixed = True
            
        optimizer.zero_grad()
        
        if mixed:
            logits = model(mixed_inputs)
            # Decoupled Label Smoothing: use raw un-smoothed BCE loss for mixed batches
            raw_criterion = criterion.bce if hasattr(criterion, 'bce') else nn.BCEWithLogitsLoss()
            loss = lam * raw_criterion(logits, labels_a) + (1.0 - lam) * raw_criterion(logits, labels_b)
        else:
            logits = model(inputs)
            loss = criterion(logits, labels)
            
        loss.backward()
        optimizer.step()
        
        # Step OneCycleLR per batch
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
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
    import torch.optim.swa_utils as swa_utils
    
    parser = argparse.ArgumentParser(description="PyTorch CBAM-EfficientNet - Cats vs Dogs (Option C)")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.0008, help="Peak or max learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="L2 regularization / weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution (224x224 standard)")
    parser.add_argument("--data-dir", type=str, default="data/PetImages", help="Path to Cats & Dogs folder")
    parser.add_argument("--weights-path", type=str, default="pytorch_cnn/model.pth", help="Path to save weights")
    parser.add_argument("--optimizer", type=str, default="lookahead", choices=["adamw", "lookahead"], help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="cosine-restarts", choices=["onecycle", "cosine-restarts"], help="Learning rate scheduler")
    parser.add_argument("--attention-type", type=str, default="se", choices=["se", "cbam"], help="Attention type in blocks")
    parser.add_argument("--swa", action="store_true", help="Enable Stochastic Weight Averaging (SWA)")
    parser.add_argument("--swa-start", type=int, default=35, help="Epoch to start SWA")
    parser.add_argument("--save-snapshots", action="store_true", help="Save snapshots of final epochs for ensembling")
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
    model = CustomCNN(attention_type=args.attention_type).to(device)
    
    # Using Label Smoothing BCE loss to improve final boundary calibration
    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=0.05)
    
    # Define Optimizer
    base_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "lookahead":
        print("⚙️ Wrapping AdamW with Lookahead Optimizer (k=5, alpha=0.5)...")
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    else:
        optimizer = base_optimizer
        
    # Define Scheduler
    if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            base_optimizer if args.optimizer == "lookahead" else optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0,
            anneal_strategy='cos'
        )
    elif args.scheduler == "cosine-restarts":
        # Cycle 1 = 15 epochs, Cycle 2 = 30 epochs (doubling cycle length). Total = 45 epochs.
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_optimizer if args.optimizer == "lookahead" else optimizer,
            T_0=15,
            T_mult=2,
            eta_min=1e-6
        )
    else:
        scheduler = None
    
    if args.swa:
        print("⚙️ Initializing Stochastic Weight Averaging (SWA) wrapper...")
        swa_model = swa_utils.AveragedModel(model)
    
    print("\n---------------------- MODEL HYPERPARAMETERS ----------------------")
    print(f"  • Device:          {device.type.upper()}")
    print(f"  • Resolution:      {args.image_size} x {args.image_size} (RGB)")
    print(f"  • Model Params:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • Attention:       {args.attention_type.upper()}")
    print(f"  • Optimizer:       {args.optimizer.upper()}")
    print(f"  • Scheduler:       {args.scheduler.upper()}")
    print(f"  • Batch Size:      {args.batch_size}")
    print(f"  • Peak Learning:   {args.lr}")
    print(f"  • Weight Decay:    {args.weight_decay}")
    print(f"  • Seed:            {args.seed}")
    print(f"  • Epochs:          {args.epochs}")
    if args.swa:
        print(f"  • SWA Enabled:     True (Starts at epoch {args.swa_start})")
    if args.save_snapshots:
        print(f"  • Snapshot Ensem.: True (Saves snapshots of final epochs)")
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
        current_lr = base_optimizer.param_groups[0]['lr'] if args.optimizer == "lookahead" else optimizer.param_groups[0]['lr']
        
        # Cooldown schedule: Disable Mixup/CutMix in the first 2 epochs and last 5 epochs
        if epoch <= 2 or epoch > (args.epochs - 5):
            mix_prob = 0.0
        else:
            mix_prob = 0.3
            
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            scheduler=scheduler, mixup_prob=mix_prob, cutmix_prob=mix_prob
        )
        
        # Evaluate with Test-Time Augmentation (TTA)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_tta=True)
        
        # Step epoch-level schedulers (CosineAnnealingWarmRestarts is stepped at epoch boundary)
        if scheduler is not None and args.scheduler == "cosine-restarts":
            scheduler.step()
            
        elapsed = time.time() - start_time
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        print(f"| Epoch {epoch:02d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_loss:10.4f} | {test_acc:9.2f}% | {current_lr:13.6f} | {elapsed:15.1f}s |", flush=True)
        
        # Save weights of best epoch
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), args.weights_path)
            
        # Update SWA parameters if SWA is enabled and we are in SWA phase
        if args.swa and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            print(f"⚙️ SWA: Averaged model weights at epoch {epoch}", flush=True)
            
        # Save snapshots of final epochs
        if args.save_snapshots and epoch > (args.epochs - 3):
            snapshot_path = args.weights_path.replace(".pth", f"_epoch{epoch}.pth")
            torch.save(model.state_dict(), snapshot_path)
            print(f"📸 Saved snapshot checkpoint: {snapshot_path}", flush=True)
            
    print("+" + "-"*96 + "+")
    print(f"✓ Training complete! Best Validation Accuracy achieved (with TTA): {best_test_acc:.2f}%")
    print(f"✓ Model weights saved to: {args.weights_path}")
    
    # SWA finalization
    if args.swa:
        print("\n⚙️ Updating Batch Normalization statistics for SWA model...", flush=True)
        swa_model = swa_model.to(device)
        swa_utils.update_bn(train_loader, swa_model, device)
        
        # Evaluate SWA model
        swa_test_loss, swa_test_acc = evaluate(swa_model, test_loader, criterion, device, use_tta=True)
        print("+" + "-"*96 + "+")
        print(f"🌟 SWA Model Test Loss: {swa_test_loss:.4f} | Test Acc (with TTA): {swa_test_acc:.2f}%")
        print("+" + "-"*96 + "+")
        
        # Save SWA weights
        swa_weights_path = args.weights_path.replace(".pth", "_swa.pth")
        torch.save(swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict(), swa_weights_path)
        print(f"✓ SWA model weights saved to: {swa_weights_path}")
        
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
