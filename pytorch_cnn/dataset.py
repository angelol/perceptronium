import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LCG:
    """
    Deterministic Pseudo-Random Number Generator based on POSIX LCG constants.
    Guarantees 100% cross-language alignment with the Rust Lcg implementation.
    """
    def __init__(self, seed):
        self.state = seed

    def next_u32(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7fffffff
        return self.state

    def next_f64(self):
        return self.next_u32() / 2147483648.0

def deterministic_shuffle(paths, seed):
    """
    In-place deterministic Fisher-Yates shuffle matching the Rust implementation exactly.
    """
    lcg = LCG(seed)
    n = len(paths)
    for i in range(n - 1, 0, -1):
        j = int(lcg.next_f64() * (i + 1))
        paths[i], paths[j] = paths[j], paths[i]

class CatsAndDogsDataset(Dataset):
    """
    Custom PyTorch Dataset for Microsoft Cats & Dogs, guaranteeing perfect
    deterministic splits and alphabetical path sorting to match Rust splits.
    Uses class-level path caching and bounded scanning to optimize load times.
    """
    # Shared cache between train/test dataset instances to prevent redundant Disk IO
    _cached_paths = {}

    def __init__(self, root_dir, split="train", image_size=224, transform=None,
                 limit_train_per_class=10000, limit_test_per_class=2501, seed=42,
                 extra_dir=None):
        self.root_dir = root_dir
        self.split = split.lower()
        self.image_size = image_size
        self.transform = transform
        
        cats_dir = os.path.join(root_dir, "Cat")
        dogs_dir = os.path.join(root_dir, "Dog")
        
        print(f"Loading paths for split: {self.split.upper()}...")
        # Create fresh lists from the sorted directory caches to prevent side effects
        cat_paths = list(self._get_valid_sorted_paths(cats_dir))
        dog_paths = list(self._get_valid_sorted_paths(dogs_dir))
        
        if extra_dir and os.path.exists(extra_dir):
            print(f"Loading extra high-quality paths from {extra_dir}...")
            extra_cats_dir = os.path.join(extra_dir, "cat")
            extra_dogs_dir = os.path.join(extra_dir, "dog")
            if os.path.exists(extra_cats_dir):
                cat_paths.extend(self._get_valid_sorted_paths(extra_cats_dir))
            if os.path.exists(extra_dogs_dir):
                dog_paths.extend(self._get_valid_sorted_paths(extra_dogs_dir))
        
        print(f"Applying aligned deterministic shuffle (seed={seed})...")
        deterministic_shuffle(cat_paths, seed)
        deterministic_shuffle(dog_paths, seed)
        
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

    def _get_valid_sorted_paths(self, dir_path):
        # Return cached paths if available
        if dir_path in CatsAndDogsDataset._cached_paths:
            return CatsAndDogsDataset._cached_paths[dir_path]
            
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
            
        files = os.listdir(dir_path)
        files.sort()  # Alphabetical sort first to preserve exact order
        
        valid_paths = []
        for f in files:
            # Skip hidden files and Apple resource forks (e.g. ._filename.jpg)
            if f.startswith("."):
                continue
                
            # Skip the exactly two corrupted files in the raw Microsoft dataset
            if f in ["666.jpg", "11702.jpg"]:
                print(f"  [Info] Skipping known corrupted image by name: {f}")
                continue
                
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(dir_path, f)
                if os.path.isfile(full_path):
                    valid_paths.append(full_path)
                        
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
