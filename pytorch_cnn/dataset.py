import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CatsAndDogsDataset(Dataset):
    """
    Custom PyTorch Dataset for Microsoft Cats & Dogs, guaranteeing perfect
    deterministic splits and alphabetical path sorting to match Rust splits.
    Uses class-level path caching and bounded scanning to optimize load times.
    """
    # Shared cache between train/test dataset instances to prevent redundant Disk IO
    _cached_paths = {}

    def __init__(self, root_dir, split="train", image_size=224, transform=None,
                 limit_train_per_class=10000, limit_test_per_class=2501):
        self.root_dir = root_dir
        self.split = split.lower()
        self.image_size = image_size
        self.transform = transform
        
        cats_dir = os.path.join(root_dir, "Cat")
        dogs_dir = os.path.join(root_dir, "Dog")
        
        # We only ever need limit_train + limit_test images from each category
        max_needed = limit_train_per_class + limit_test_per_class
        
        print(f"Loading paths for split: {self.split.upper()} (checking up to {max_needed} valid files per class)...")
        cat_paths = self._get_valid_sorted_paths(cats_dir, max_needed)
        dog_paths = self._get_valid_sorted_paths(dogs_dir, max_needed)
        
        # Partition paths deterministically matching the Rust offsets
        if self.split == "train":
            # First limit_train_per_class valid images per category
            self.paths = (
                cat_paths[:limit_train_per_class] + 
                dog_paths[:limit_train_per_class]
            )
            # Label 0 for Cat, 1 for Dog
            self.labels = (
                [0.0] * len(cat_paths[:limit_train_per_class]) + 
                [1.0] * len(dog_paths[:limit_train_per_class])
            )
        elif self.split in ["test", "val", "validation"]:
            # Next limit_test_per_class valid images (skip first limit_train_per_class)
            cat_test = cat_paths[limit_train_per_class : limit_train_per_class + limit_test_per_class]
            dog_test = dog_paths[limit_train_per_class : limit_train_per_class + limit_test_per_class]
            
            self.paths = cat_test + dog_test
            self.labels = [0.0] * len(cat_test) + [1.0] * len(dog_test)
        else:
            raise ValueError(f"Unknown split: {self.split}")
            
        print(f"✓ Loaded {len(self.paths)} images for {self.split} split ({self.labels.count(0.0)} Cats, {self.labels.count(1.0)} Dogs)")

    def _get_valid_sorted_paths(self, dir_path, max_needed):
        # Return cached paths if available
        if dir_path in CatsAndDogsDataset._cached_paths:
            return CatsAndDogsDataset._cached_paths[dir_path]
            
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
            
        files = os.listdir(dir_path)
        files.sort()  # Alphabetical sort first to preserve exact order
        
        valid_paths = []
        for f in files:
            # Bounded scanning: Stop once we have gathered all the files we will ever need
            if len(valid_paths) >= max_needed:
                break
                
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(dir_path, f)
                if os.path.isfile(full_path):
                    # Fast validation: check if PIL can open it
                    try:
                        with Image.open(full_path) as img:
                            img.draft(img.mode, (32, 32))  # Load minimal pixel draft
                        valid_paths.append(full_path)
                    except Exception:
                        print(f"  [Warning] Skipping corrupted image during scanning: {f}")
                        
        # Save in class-level cache
        CatsAndDogsDataset._cached_paths[dir_path] = valid_paths
        return valid_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        
        try:
            # Load as RGB
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback to an empty/dummy image if reading fails at runtime
            print(f"  [Warning] Failed to load image {path} during training, returning dummy image. Error: {e}")
            img = Image.new("RGB", (self.image_size, self.image_size), (127, 127, 127))
            
        if self.transform:
            img = self.transform(img)
        else:
            # Default transformation if none specified
            default_trans = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            img = default_trans(img)
            
        # Target label is float for BCEWithLogitsLoss
        return img, torch.tensor(label, dtype=torch.float32)
