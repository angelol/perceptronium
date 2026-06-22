import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_trans
from tqdm import tqdm

# Set matplotlib backend to non-GUI to avoid errors on headless runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CatsAndDogsDataset
from model import CustomCNN


class CustomRandAugment:
    """
    Customized RandAugment that excludes extreme color inversions (Solarize, Invert)
    which destroy biological fur color representation of dogs and cats.
    """
    def __init__(self, num_ops=2, magnitude=9):
        self.num_ops = num_ops
        self.magnitude = magnitude  # out of 10
        self.op_list = [
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,
            self.rotate,
            self.brightness,
            self.color,
            self.contrast,
            self.sharpness,
            self.posterize,
            self.equalize,
            self.autocontrast
        ]

    def shear_x(self, img, val):
        deg = val * 30.0
        return F_trans.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[deg, 0.0])

    def shear_y(self, img, val):
        deg = val * 30.0
        return F_trans.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0.0, deg])

    def translate_x(self, img, val):
        w, _ = img.size
        shift = int(val * w * 0.3)
        return F_trans.affine(img, angle=0, translate=[shift, 0], scale=1.0, shear=[0.0, 0.0])

    def translate_y(self, img, val):
        _, h = img.size
        shift = int(val * h * 0.3)
        return F_trans.affine(img, angle=0, translate=[0, shift], scale=1.0, shear=[0.0, 0.0])

    def rotate(self, img, val):
        deg = val * 30.0
        return F_trans.rotate(img, deg)

    def brightness(self, img, val):
        factor = 1.0 + (val * 0.9) if random.random() < 0.5 else 1.0 - (val * 0.9)
        return F_trans.adjust_brightness(img, max(0.1, factor))

    def color(self, img, val):
        factor = 1.0 + (val * 0.9) if random.random() < 0.5 else 1.0 - (val * 0.9)
        return F_trans.adjust_saturation(img, max(0.1, factor))

    def contrast(self, img, val):
        factor = 1.0 + (val * 0.9) if random.random() < 0.5 else 1.0 - (val * 0.9)
        return F_trans.adjust_contrast(img, max(0.1, factor))

    def sharpness(self, img, val):
        factor = 1.0 + (val * 0.9) if random.random() < 0.5 else 1.0 - (val * 0.9)
        return F_trans.adjust_sharpness(img, max(0.1, factor))

    def posterize(self, img, val):
        bits = int(8 - (val * 4))
        return F_trans.posterize(img, max(1, bits))

    def equalize(self, img, val):
        return F_trans.equalize(img)

    def autocontrast(self, img, val):
        return F_trans.autocontrast(img)

    def __call__(self, img):
        ops = random.sample(self.op_list, self.num_ops)
        for op in ops:
            val = (self.magnitude / 10.0) * (1.0 if random.random() < 0.5 else -1.0)
            if op.__name__ in ["posterize", "equalize", "autocontrast"]:
                val = abs(val)
            img = op(img, val)
        return img


class ExponentialMovingAverage:
    """
    Maintains Exponential Moving Average (EMA) shadow weights for deep learning models.
    Smooths optimization trajectories and settles in flat generalization basins.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register shadow weights
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self):
        # Run inside torch.no_grad()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])
        self.backup.clear()
        
    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow
        }
        
    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        for name, val in state_dict["shadow"].items():
            if name in self.shadow:
                self.shadow[name].copy_(val.to(self.shadow[name].device))

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
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = self.optimizer.defaults
        self.lk_counter = 0
        
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
                    slow_p.add_(self.alpha * (p.data - slow_p))
                    p.data.copy_(slow_p)
                    
        return loss

    def __getattr__(self, item):
        return getattr(self.optimizer, item)


class SharpnessAwareMinimization:
    """
    Sharpness-Aware Minimization (SAM) and Scale-Invariant Adaptive SAM (ASAM).
    An optimizer wrapper that executes double-forward/backward passes natively on MPS/CUDA GPU.
    """
    def __init__(self, optimizer, rho=0.5, eta=0.01, adaptive=True):
        self.optimizer = optimizer
        self.rho = rho
        self.eta = eta
        self.adaptive = adaptive
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        
        # Save old weights in an isolated dictionary to avoid polluting self.optimizer.state[p],
        # which would cause AdamW/Lion to skip initializing their momentum buffers.
        self.old_weights = {}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        if grad_norm == 0:
            return
            
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Save the original weights in our isolated dictionary
                self.old_weights[p] = p.data.clone()
                
                grad = p.grad.data
                if self.adaptive:
                    tw = torch.abs(p.data) + self.eta
                    eps = grad * tw * tw
                else:
                    eps = grad.clone()
                    
                eps.mul_(self.rho / grad_norm)
                p.data.add_(eps)
                
        if zero_grad:
            self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p in self.old_weights:
                    p.data.copy_(self.old_weights[p])
                    
        self.optimizer.step()
        
        # Clean up old_weights to avoid holding references to gradients or weights indefinitely
        self.old_weights.clear()
        
        if zero_grad:
            self.optimizer.zero_grad()

    def _grad_norm(self):
        norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if self.adaptive:
                    tw = torch.abs(p.data) + self.eta
                    norm += torch.sum((grad * tw) ** 2)
                else:
                    norm += torch.sum(grad ** 2)
                    
        norm = torch.sqrt(norm)
        return norm.item()

    def step(self, closure=None):
        raise NotImplementedError("SAM/ASAM requires a dual-step training loop. Use first_step and second_step.")

    def __getattr__(self, item):
        return getattr(self.optimizer, item)


class Lion(optim.Optimizer):
    """
    Lion Optimizer (Evo-discovered Sign Momentum Optimizer, Chen et al., 2023).
    Saves memory and regularizes updates along consensus gradient dimensions.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                update = exp_avg * beta1 + grad * (1.0 - beta1)
                p.add_(torch.sign(update), alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


def get_transforms(image_size):
    """Defines high-generalization training (highly augmented) and test image transformations."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), ratio=(3./4., 4./3.)),
        transforms.RandomHorizontalFlip(p=0.5),
        CustomRandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None, 
                    mixup_prob=0.3, cutmix_prob=0.3, progressive=False, use_sam=False, use_asam=False,
                    scaler=None, amp_dtype=torch.float16, ema=None, total_epochs=200):
    """Executes a single epoch of training with optional Mixup and CutMix batch-mixing, progressive resizing, and SAM/ASAM."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progressive resizing target dimension matching 200-epoch curriculum
    if progressive:
        if total_epochs == 200:
            if epoch <= 60:
                current_size = 160
            elif epoch <= 130:
                current_size = 224
            else:
                current_size = 288
        else:
            # Proportional scaling for non-standard runs (e.g. smoke tests)
            stage_1_end = int(0.3 * total_epochs)
            stage_2_end = int(0.65 * total_epochs)
            if epoch <= stage_1_end:
                current_size = 160
            elif epoch <= stage_2_end:
                current_size = 224
            else:
                current_size = 288
    else:
        current_size = 224

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch:02d}", leave=True)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1) # Reshape to (batch_size, 1)
        
        # GPU progressive resizing
        if progressive and inputs.shape[-1] != current_size:
            inputs = F.interpolate(inputs, size=(current_size, current_size), mode="bilinear", align_corners=False)

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
            
        raw_criterion = criterion.bce if hasattr(criterion, 'bce') else nn.BCEWithLogitsLoss()
        
        # Determine autocast activation based on scaler presence or BF16 usage
        amp_enabled = (scaler is not None) or (amp_dtype == torch.bfloat16 and device.type == "cuda")
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        
        if use_sam or use_asam:
            # First forward-backward pass
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
                if mixed:
                    logits = model(mixed_inputs)
                    loss = lam * raw_criterion(logits, labels_a) + (1.0 - lam) * raw_criterion(logits, labels_b)
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.first_step(zero_grad=True)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass (on perturbed parameters)
            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
                if mixed:
                    logits2 = model(mixed_inputs)
                    loss2 = lam * raw_criterion(logits2, labels_a) + (1.0 - lam) * raw_criterion(logits2, labels_b)
                else:
                    logits2 = model(inputs)
                    loss2 = criterion(logits2, labels)
            if scaler is not None:
                scaler.scale(loss2).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.second_step(zero_grad=True)
            else:
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.second_step(zero_grad=True)
                
            if ema is not None:
                ema.update()
        else:
            # Standard single forward-backward pass
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
                if mixed:
                    logits = model(mixed_inputs)
                    loss = lam * raw_criterion(logits, labels_a) + (1.0 - lam) * raw_criterion(logits, labels_b)
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            if ema is not None:
                ema.update()
            
        # Step OneCycleLR per batch
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
            
        # Accumulate metrics
        running_loss += loss.item() * inputs.size(0)
        
        # Approximate training accuracy (no redundant forward pass)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            if mixed:
                dominant_labels = labels_a if lam >= 0.5 else labels_b
                correct += (preds == dominant_labels).sum().item()
            else:
                correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        # Real-time metrics in tqdm progress bar
        progress_bar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{(correct / total) * 100.0:.2f}%"
        })
            
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
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        logits = model(inputs)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * inputs.size(0)
        
        if use_tta:
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


@torch.no_grad()
def evaluate_advanced_tta(model, dataloader, criterion, device):
    """
    Evaluates the model on validation/testing set with 12-View Center-Crop & Flip TTA.
    Uses 6 scales: [160, 192, 224, 256, 288, 320], averaging original and horizontal flip outputs.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Validating (12-View TTA)", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        # Calculate standard loss (with base resolution)
        logits = model(inputs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * inputs.size(0)
        
        # 12-View Multi-Scale TTA
        scales = [160, 192, 224, 256, 288, 320]
        all_probs = []
        
        for scale in scales:
            if inputs.shape[-1] != scale:
                inputs_scaled = F.interpolate(inputs, size=(scale, scale), mode="bilinear", align_corners=False)
            else:
                inputs_scaled = inputs
                
            logits_orig = model(inputs_scaled)
            probs_orig = torch.sigmoid(logits_orig)
            all_probs.append(probs_orig)
            
            inputs_flipped = torch.flip(inputs_scaled, dims=[3])
            logits_flipped = model(inputs_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)
            all_probs.append(probs_flipped)
            
        avg_probs = torch.stack(all_probs).mean(dim=0)
        
        preds = (avg_probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def main():
    import torch.optim.swa_utils as swa_utils
    from torch.optim.swa_utils import SWALR
    
    parser = argparse.ArgumentParser(description="PyTorch Hybrid CNN-Transformer v3 - Cats vs Dogs")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.0008, help="Peak or max learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="L2 regularization / weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution")
    parser.add_argument("--data-dir", type=str, default="data/PetImages", help="Path to Cats & Dogs folder")
    parser.add_argument("--extra-dir", type=str, default="/Users/al/Projects/angelo/cats_dogs_dataset", help="Path to additional high-quality Cats & Dogs folder")
    parser.add_argument("--weights-path", type=str, default="pytorch_cnn/model.pth", help="Path to save weights")
    parser.add_argument("--optimizer", type=str, default="asam", choices=["adamw", "lookahead", "sam", "asam", "lion"], help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="cosine-restarts", choices=["onecycle", "cosine-restarts", "sequential", "none"], help="Learning rate scheduler")
    parser.add_argument("--attention-type", type=str, default="se", choices=["se", "cbam"], help="Attention type in blocks")
    parser.add_argument("--swa", action="store_true", help="Enable Stochastic Weight Averaging (SWA)")
    parser.add_argument("--swa-start", type=int, default=35, help="Epoch to start SWA")
    parser.add_argument("--save-snapshots", action="store_true", help="Save snapshots of final epochs for ensembling")
    parser.add_argument("--progressive", action="store_true", help="Enable progressive resizing curriculum")
    parser.add_argument("--limit-train", type=int, default=12150, help="Max training images per class")
    parser.add_argument("--limit-test", type=int, default=1349, help="Max testing/validation images per class")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--bf16", action="store_true", help="Enable Native Bfloat16 Mixed Precision (recommended for L4)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile model compilation")
    parser.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
    parser.add_argument("--resume", action="store_true", help="Enable auto-resume if checkpoint exists")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using hardware-accelerated device: METAL GPU (MPS)", flush=True)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using hardware-accelerated device: NVIDIA GPU (CUDA)", flush=True)
        # Enable TF32 globally for matrix multiplications and convolutions on L4/Ampere
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("⚙️ TensorFloat-32 (TF32) Enabled for high-performance matmul/conv", flush=True)
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU device (slow)", flush=True)
        
    if not os.path.exists(args.data_dir) and os.path.exists("../data/PetImages"):
        args.data_dir = "../data/PetImages"
        
    print(f"Dataset target folder: {args.data_dir}", flush=True)
    print(f"Extra dataset folder: {args.extra_dir}", flush=True)
    
    # 2. Setup Data Loading & Splits
    train_transform, test_transform = get_transforms(args.image_size)
    
    train_dataset = CatsAndDogsDataset(
        root_dir=args.data_dir,
        split="train",
        image_size=args.image_size,
        transform=train_transform,
        limit_train_per_class=args.limit_train,
        limit_test_per_class=args.limit_test,
        extra_dir=args.extra_dir
    )
    
    test_dataset = CatsAndDogsDataset(
        root_dir=args.data_dir,
        split="test",
        image_size=args.image_size,
        transform=test_transform,
        limit_train_per_class=args.limit_train,
        limit_test_per_class=args.limit_test,
        extra_dir=args.extra_dir
    )
    
    # Optimized DataLoader configurations (persistent_workers + drop_last for static compile traces)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # 3. Model, Loss, Optimizer, and LR Scheduler
    model = CustomCNN(attention_type=args.attention_type).to(device)
    
    # Model Compilation (Optional, high-performance on G2 GPU)
    if args.compile:
        print(f"⚙️ Compiling model with torch.compile (mode={args.compile_mode})...", flush=True)
        model = torch.compile(model, mode=args.compile_mode)
        
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = None
    if args.amp and device.type == "cuda":
        if args.bf16:
            print("⚙️ Bfloat16 (BF16) Mixed Precision Enabled (No GradScaler needed)", flush=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            print("⚙️ FP16 Automatic Mixed Precision (AMP) Enabled with GradScaler", flush=True)
        
    # Using Label Smoothing BCE loss to improve final boundary calibration
    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=0.05)
    
    # Define Base Optimizer
    if args.optimizer == "lion":
        print("⚙️ Initializing Lion Optimizer...", flush=True)
        base_optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        base_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    # Wrap Optimizer if needed
    use_sam = args.optimizer == "sam"
    use_asam = args.optimizer == "asam"
    
    if args.optimizer == "lookahead":
        print("⚙️ Wrapping AdamW with Lookahead Optimizer (k=5, alpha=0.5)...", flush=True)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    elif use_sam:
        print("⚙️ Wrapping AdamW with SAM Optimizer (rho=0.05)...", flush=True)
        optimizer = SharpnessAwareMinimization(base_optimizer, rho=0.05, adaptive=False)
    elif use_asam:
        print("⚙️ Wrapping AdamW with ASAM Optimizer (rho=0.5, scale-invariant)...", flush=True)
        optimizer = SharpnessAwareMinimization(base_optimizer, rho=0.5, adaptive=True)
    else:
        optimizer = base_optimizer
        
    # Define Scheduler
    if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            base_optimizer if args.optimizer in ["lookahead", "sam", "asam"] else optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0,
            anneal_strategy='cos'
        )
    elif args.scheduler == "cosine-restarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_optimizer if args.optimizer in ["lookahead", "sam", "asam"] else optimizer,
            T_0=15,
            T_mult=2,
            eta_min=1e-6
        )
    elif args.scheduler == "sequential":
        print("⚙️ Initializing SequentialLR (10-epoch Linear Warmup + Cosine Annealing)...", flush=True)
        if args.epochs == 200:
            warmup_epochs = 10
            cosine_epochs = 190
        else:
            warmup_epochs = max(1, int(0.05 * args.epochs))
            cosine_epochs = args.epochs - warmup_epochs
            
        opt = base_optimizer if args.optimizer in ["lookahead", "sam", "asam"] else optimizer
        
        scheduler1 = optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=cosine_epochs,
            eta_min=1e-6
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[scheduler1, scheduler2],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = None
    
    # Decoupled SWA Scheduler
    swa_scheduler = None
    if args.swa:
        print("⚙️ Initializing Stochastic Weight Averaging (SWA) wrapper...", flush=True)
        swa_model = swa_utils.AveragedModel(model)
        swa_scheduler = SWALR(
            base_optimizer if args.optimizer in ["lookahead", "sam", "asam"] else optimizer,
            swa_lr=1.5e-4,
            anneal_epochs=3,
            anneal_strategy="cos"
        )
    
    print("\n---------------------- MODEL HYPERPARAMETERS ----------------------")
    print(f"  • Device:          {device.type.upper()}", flush=True)
    print(f"  • Resolution:      {args.image_size} x {args.image_size} (RGB)", flush=True)
    print(f"  • Model Params:    {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"  • Attention:       {args.attention_type.upper()}", flush=True)
    print(f"  • Optimizer:       {args.optimizer.upper()}", flush=True)
    print(f"  • Scheduler:       {args.scheduler.upper()}", flush=True)
    print(f"  • Batch Size:      {args.batch_size}", flush=True)
    print(f"  • Peak Learning:   {args.lr}", flush=True)
    print(f"  • Weight Decay:    {args.weight_decay}", flush=True)
    print(f"  • Seed:            {args.seed}", flush=True)
    print(f"  • Epochs:          {args.epochs}", flush=True)
    print(f"  • Progressive:     {args.progressive}", flush=True)
    if args.swa:
        print(f"  • SWA Enabled:     True (Starts at epoch {args.swa_start})", flush=True)
    if args.save_snapshots:
        print(f"  • Snapshot Ensem.: True (Saves snapshots of final epochs)", flush=True)
    print("-------------------------------------------------------------------\n")
    
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }
    best_test_acc = 0.0
    
    # Initialize EMA shadow weights
    ema = ExponentialMovingAverage(model, decay=0.999)
    
    # Resume training state if requested and checkpoint exists
    start_epoch = 1
    checkpoint_path = args.weights_path.replace(".pth", "_checkpoint_latest.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"🔄 Found latest checkpoint: {checkpoint_path}. Resuming training...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        if args.swa and "swa_model_state_dict" in checkpoint and checkpoint["swa_model_state_dict"] is not None:
            swa_model.load_state_dict(checkpoint["swa_model_state_dict"])
        best_test_acc = checkpoint.get("best_test_acc", 0.0)
        history = checkpoint.get("history", history)
        start_epoch = checkpoint["epoch"] + 1
        print(f"✓ Successfully resumed from Epoch {checkpoint['epoch']} with Best Test Acc: {best_test_acc:.2f}%", flush=True)
        
    print("+" + "-"*96 + "+")
    print("| Epoch    | Train Loss | Train Acc | Test Loss  | Test Acc   | Learning Rate | Epoch Time (sec) |")
    print("+" + "-"*96 + "+")
    
    current_size = args.image_size
    current_bs = args.batch_size
    
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        
        # Reconstruct dataloaders for Progressive Resizing
        if args.progressive:
            rebuild_loader = False
            if epoch == 1:
                current_size, current_bs = 160, 128
                rebuild_loader = True
            elif epoch == 61:
                current_size, current_bs = 224, 128
                rebuild_loader = True
            elif epoch == 131:
                current_size, current_bs = 288, 64
                rebuild_loader = True
                
            if rebuild_loader:
                current_bs = min(current_bs, args.batch_size)
                print(f"\n🔄 [Progressive Resizing] Rebuilding dataloaders for spatial size {current_size}x{current_size} with Batch Size {current_bs}", flush=True)
                train_transform, test_transform = get_transforms(current_size)
                
                train_dataset = CatsAndDogsDataset(
                    root_dir=args.data_dir,
                    split="train",
                    image_size=current_size,
                    transform=train_transform,
                    limit_train_per_class=args.limit_train,
                    limit_test_per_class=args.limit_test,
                    extra_dir=args.extra_dir
                )
                test_dataset = CatsAndDogsDataset(
                    root_dir=args.data_dir,
                    split="test",
                    image_size=current_size,
                    transform=test_transform,
                    limit_train_per_class=args.limit_train,
                    limit_test_per_class=args.limit_test,
                    extra_dir=args.extra_dir
                )
                
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=current_bs, 
                    shuffle=True, 
                    num_workers=4, 
                    pin_memory=True,
                    persistent_workers=True,
                    drop_last=True
                )
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=current_bs, 
                    shuffle=False, 
                    num_workers=4, 
                    pin_memory=True,
                    persistent_workers=True
                )
        
        current_lr = base_optimizer.param_groups[0]['lr'] if args.optimizer in ["lookahead", "sam", "asam"] else optimizer.param_groups[0]['lr']
        
        # Scheduled Mixup/CutMix Regularization:
        if args.epochs == 200:
            if epoch <= 10 or epoch > 180:
                mix_prob = 0.0
            else:
                mix_prob = 0.3
        else:
            warmup_end = max(1, int(0.05 * args.epochs))
            cooldown_start = int(0.9 * args.epochs)
            if epoch <= warmup_end or epoch > cooldown_start:
                mix_prob = 0.0
            else:
                mix_prob = 0.3
            
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scheduler=scheduler, mixup_prob=mix_prob, cutmix_prob=mix_prob,
            progressive=args.progressive, use_sam=use_sam, use_asam=use_asam,
            scaler=scaler, amp_dtype=amp_dtype, ema=ema, total_epochs=args.epochs
        )
        
        # Apply EMA shadow weights for validation/testing evaluation
        ema.apply_shadow()
        test_loss, test_acc = evaluate_advanced_tta(model, test_loader, criterion, device)
        # Restore original parameter weights for training
        ema.restore()
        
        # Step epoch-level schedulers (except during SWA constant phase)
        if scheduler is not None and args.scheduler in ["cosine-restarts", "sequential"]:
            if not args.swa or epoch < args.swa_start:
                scheduler.step()
            
        elapsed = time.time() - start_time
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        print(f"| Epoch {epoch:02d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_loss:10.4f} | {test_acc:9.2f}% | {current_lr:13.6f} | {elapsed:15.1f}s |", flush=True)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save the best EMA shadow weights to args.weights_path
            ema.apply_shadow()
            torch.save(model.state_dict(), args.weights_path)
            ema.restore()
            print(f"🌟 Saved new best EMA weights with accuracy: {best_test_acc:.2f}%", flush=True)
            
        # Update SWA parameters if SWA is enabled and we are in SWA phase
        if args.swa and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            print(f"⚙️ SWA: Averaged model weights at epoch {epoch}", flush=True)
            if swa_scheduler is not None:
                swa_scheduler.step()
            
        # Save snapshots of specific epochs for Snapshot Ensembling
        if args.save_snapshots:
            should_save = False
            if args.epochs == 200:
                if epoch in [15, 45, 105, 150]:
                    should_save = True
            else:
                milestones = [int(p * args.epochs) for p in [0.075, 0.225, 0.525, 0.75]]
                milestones = sorted(list(set(m for m in milestones if 1 <= m <= args.epochs)))
                if epoch in milestones:
                    should_save = True
                    
            if should_save:
                snapshot_path = args.weights_path.replace(".pth", f"_epoch{epoch}.pth")
                # Save EMA shadow weights as the snapshot
                ema.apply_shadow()
                torch.save(model.state_dict(), snapshot_path)
                ema.restore()
                print(f"📸 Saved snapshot checkpoint (EMA weights): {snapshot_path}", flush=True)
                
        # Save continuous checkpoint for resiliency/auto-resume
        checkpoint_path = args.weights_path.replace(".pth", "_checkpoint_latest.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_test_acc": best_test_acc,
            "history": history,
            "ema_state_dict": ema.state_dict(),
        }
        if args.swa:
            checkpoint["swa_model_state_dict"] = swa_model.state_dict()
        torch.save(checkpoint, checkpoint_path)
        print(f"📸 Saved latest training state checkpoint to: {checkpoint_path}", flush=True)
            
    print("+" + "-"*96 + "+")
    print(f"✓ Training complete! Best Validation Accuracy achieved (with 6-View TTA): {best_test_acc:.2f}%")
    print(f"✓ Model weights saved to: {args.weights_path}")
    
    # SWA finalization
    if args.swa:
        print("\n⚙️ Updating Batch Normalization statistics for SWA model...", flush=True)
        swa_model = swa_model.to(device)
        swa_utils.update_bn(train_loader, swa_model, device)
        
        swa_test_loss, swa_test_acc = evaluate_advanced_tta(swa_model, test_loader, criterion, device)
        print("+" + "-"*96 + "+")
        print(f"🌟 SWA Model Test Loss: {swa_test_loss:.4f} | Test Acc (with 6-View TTA): {swa_test_acc:.2f}%")
        print("+" + "-"*96 + "+")
        
        swa_weights_path = args.weights_path.replace(".pth", "_swa.pth")
        torch.save(swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict(), swa_weights_path)
        print(f"✓ SWA model weights saved to: {swa_weights_path}")
        
    # 4. Save Learning Curve Plots
    epochs_range = range(1, args.epochs + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss (Mixed)", color="#ff7f0e", lw=2)
    plt.plot(epochs_range, history["test_loss"], label="Test Loss (Clean)", color="#1f77b4", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Testing Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Accuracy (Approx)", color="#2ca02c", lw=2)
    plt.plot(epochs_range, history["test_acc"], label="Test Accuracy (with 6-View TTA)", color="#d62728", lw=2)
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
