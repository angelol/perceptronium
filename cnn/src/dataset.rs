use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use crate::math::Lcg;

/// Ensures that the cats and dogs dataset is downloaded and extracted.
/// Checks locally first, and if not present but `../data` exists, it uses `../data`
/// to avoid redundant downloads.
pub fn ensure_dataset_downloaded() -> Result<PathBuf, String> {
    let mut data_dir = PathBuf::from("data");
    
    // Check if running from inside the sub-crate and ../data exists
    if !data_dir.exists() && Path::new("../data").exists() {
        data_dir = PathBuf::from("../data");
    }

    let dataset_dir = data_dir.join("PetImages");

    // Check if dataset already exists
    if dataset_dir.exists() {
        println!("✓ Dataset found at: {}", dataset_dir.display());
        return Ok(dataset_dir);
    }

    println!("Dataset not found. Creating 'data' folder and downloading full Microsoft Kaggle Cats & Dogs dataset (787MB)...");
    fs::create_dir_all(&data_dir).map_err(|e| format!("Failed to create data dir: {}", e))?;

    let zip_path = data_dir.join("kagglecatsanddogs_5340.zip");

    // Download using curl
    println!("Downloading dataset via curl...");
    let status = Command::new("curl")
        .args(&[
            "-L", // follow redirects
            "-o",
            zip_path.to_str().unwrap(),
            "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip",
        ])
        .status()
        .map_err(|e| format!("Failed to run curl: {}", e))?;

    if !status.success() {
        return Err("curl download failed".to_string());
    }

    // Extract using unzip
    println!("Extracting dataset via unzip...");
    let unzip_status = Command::new("unzip")
        .args(&[
            "-q", // quiet mode
            zip_path.to_str().unwrap(),
            "-d",
            data_dir.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("Failed to run unzip: {}. Please ensure 'unzip' is installed.", e))?;

    if !unzip_status.success() {
        return Err("unzip extraction failed".to_string());
    }

    // Clean up zip file
    let _ = fs::remove_file(zip_path);

    println!("✓ Dataset successfully downloaded and extracted!");
    Ok(dataset_dir)
}

/// Loads a single image, resizes it, converts to grayscale, and normalizes it.
pub fn load_image(path: &Path, size: u32) -> Result<Vec<f64>, String> {
    // Open image using the 'image' crate
    let img = image::open(path)
        .map_err(|e| format!("Failed to open image {}: {}", path.display(), e))?;

    // Downscale using thumbnail_exact (much faster, ensures dimensions are exactly size x size)
    let resized = img.thumbnail_exact(size, size);

    // Convert to 8-bit grayscale (Luma)
    let grayscale = resized.into_luma8();

    // Verify dimensions match expected size
    if grayscale.width() != size || grayscale.height() != size {
        return Err(format!(
            "Resized image dimensions ({:?}) do not match target size ({}x{})",
            grayscale.dimensions(),
            size,
            size
        ));
    }

    // Convert pixels to f64 normalized values in [0.0, 1.0]
    let flat_pixels: Vec<f64> = grayscale
        .pixels()
        .map(|pixel| pixel.0[0] as f64 / 255.0)
        .collect();

    Ok(flat_pixels)
}

/// Renders a grayscale image in the terminal as detailed ASCII art.
pub fn print_ascii_preview(pixels: &[f64], size: usize) {
    // ASCII density character set (from dark to bright)
    const ASCII_CHARS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '%', '#', '@'];
    
    // Standard terminal character cell aspect ratio is roughly 2:1.
    // For 128x128 we might step by 3 or 4 rows to keep the terminal output readable in height,
    // let's step by 4 rows for 128x128 and 2 cols for better display.
    let step_row = if size >= 128 { 4 } else { 2 };
    let step_col = if size >= 128 { 2 } else { 1 };

    for r in (0..size).step_by(step_row) {
        for c in (0..size).step_by(step_col) {
            let pixel = pixels[r * size + c];
            let idx = (pixel * (ASCII_CHARS.len() - 1) as f64).round() as usize;
            print!("{}", ASCII_CHARS[idx]);
        }
        println!();
    }
}

fn scan_sorted_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_file() {
            if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                if file_name == "666.jpg" || file_name == "11702.jpg" {
                    println!("  [Info] Skipping known corrupted image by name: {}", file_name);
                    continue;
                }
            }
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let ext = ext.to_lowercase();
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    paths.push(path);
                }
            }
        }
    }
    paths.sort();
    Ok(paths)
}

/// Helper to load a deterministically sorted and sliced list of image vectors from a given category directory.
fn load_from_dir_deterministic(
    primary_dir: &Path,
    extra_dir: Option<&Path>,
    skip: usize,
    limit: usize,
    label: f64,
    image_size: u32,
    seed: u32,
    dest: &mut Vec<(Vec<f64>, f64)>,
) -> Result<(), String> {
    println!("Scanning primary directory {}...", primary_dir.display());
    let mut paths = scan_sorted_dir(primary_dir)?;

    if let Some(ed) = extra_dir {
        if ed.exists() {
            println!("Scanning extra directory {}...", ed.display());
            let extra_paths = scan_sorted_dir(ed)?;
            paths.extend(extra_paths);
        }
    }

    // Apply deterministic random Fisher-Yates shuffle matching the Python PyTorch dataset
    let mut shuffle_prng = Lcg::new(seed);
    let num_paths = paths.len();
    for i in (1..num_paths).rev() {
        let j = (shuffle_prng.next_f64() * (i + 1) as f64).floor() as usize;
        paths.swap(i, j);
    }

    println!("Found {} total combined images. Loading up to {} images (skipping first {})...", paths.len(), limit, skip);

    let mut count = 0;
    let mut skipped = 0;

    for path in paths {
        if count >= limit {
            break;
        }

        if skipped < skip {
            skipped += 1;
            continue;
        }

        match load_image(&path, image_size) {
            Ok(pixels) => {
                dest.push((pixels, label));
                count += 1;
            }
            Err(e) => {
                // Gracefully log and skip corrupted files (common in Kaggle Cats & Dogs)
                println!("  [Warning] Skipping corrupted image {}: {}", path.display(), e);
            }
        }
    }

    println!("Successfully loaded {} images.", count);
    Ok(())
}

/// Loads and prepares the training and testing datasets.
pub fn load_split(
    dataset_dir: &Path,
    limit_train_per_class: usize,
    limit_test_per_class: usize,
    image_size: u32,
    seed: u32,
) -> Result<
    (
        Vec<Vec<f64>>, // Train inputs
        Vec<f64>,      // Train labels (0.0=Cat, 1.0=Dog)
        Vec<Vec<f64>>, // Test inputs
        Vec<f64>,      // Test labels
    ),
    String,
> {
    let mut train_data = Vec::new();
    let mut test_data = Vec::new();

    let cats_dir = dataset_dir.join("Cat");
    let dogs_dir = dataset_dir.join("Dog");

    let extra_dir_base = Path::new("/Users/al/Projects/angelo/cats_dogs_dataset");
    let (extra_cats, extra_dogs) = if extra_dir_base.exists() {
        (Some(extra_dir_base.join("cat")), Some(extra_dir_base.join("dog")))
    } else {
        (None, None)
    };

    // 1. Load training data: first limit_train_per_class elements (skip = 0)
    load_from_dir_deterministic(&cats_dir, extra_cats.as_deref(), 0, limit_train_per_class, 0.0, image_size, seed, &mut train_data)?;
    load_from_dir_deterministic(&dogs_dir, extra_dogs.as_deref(), 0, limit_train_per_class, 1.0, image_size, seed, &mut train_data)?;

    // 2. Load testing data: next limit_test_per_class elements (skip = limit_train_per_class)
    load_from_dir_deterministic(&cats_dir, extra_cats.as_deref(), limit_train_per_class, limit_test_per_class, 0.0, image_size, seed, &mut test_data)?;
    load_from_dir_deterministic(&dogs_dir, extra_dogs.as_deref(), limit_train_per_class, limit_test_per_class, 1.0, image_size, seed, &mut test_data)?;

    if train_data.is_empty() || test_data.is_empty() {
        return Err("Loaded datasets are empty. Check dataset paths and contents.".to_string());
    }

    // 3. Shuffle datasets (using deterministic seeded PRNG)
    let mut prng = Lcg::new(seed);
    
    let shuffle = |data: &mut Vec<(Vec<f64>, f64)>, prng: &mut Lcg| {
        let n = data.len();
        for i in (1..n).rev() {
            let j = (prng.next_f64() * (i + 1) as f64).floor() as usize;
            data.swap(i, j);
        }
    };

    shuffle(&mut train_data, &mut prng);
    shuffle(&mut test_data, &mut prng);

    // 4. Split inputs and labels
    let (train_inputs, train_labels): (Vec<Vec<f64>>, Vec<f64>) = train_data.into_iter().unzip();
    let (test_inputs, test_labels): (Vec<Vec<f64>>, Vec<f64>) = test_data.into_iter().unzip();

    Ok((train_inputs, train_labels, test_inputs, test_labels))
}

/// Loads and prepares ONLY the validation/testing dataset.
pub fn load_test_split(
    dataset_dir: &Path,
    limit_train_per_class: usize,
    limit_test_per_class: usize,
    image_size: u32,
    seed: u32,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let mut test_data = Vec::new();

    let cats_dir = dataset_dir.join("Cat");
    let dogs_dir = dataset_dir.join("Dog");

    let extra_dir_base = Path::new("/Users/al/Projects/angelo/cats_dogs_dataset");
    let (extra_cats, extra_dogs) = if extra_dir_base.exists() {
        (Some(extra_dir_base.join("cat")), Some(extra_dir_base.join("dog")))
    } else {
        (None, None)
    };

    // Load testing data: skip the training data entirely, grab the test subset
    load_from_dir_deterministic(&cats_dir, extra_cats.as_deref(), limit_train_per_class, limit_test_per_class, 0.0, image_size, seed, &mut test_data)?;
    load_from_dir_deterministic(&dogs_dir, extra_dogs.as_deref(), limit_train_per_class, limit_test_per_class, 1.0, image_size, seed, &mut test_data)?;

    if test_data.is_empty() {
        return Err("Loaded validation dataset is empty. Check dataset path.".to_string());
    }

    // Shuffle validation set deterministically
    let mut prng = Lcg::new(seed);
    let n = test_data.len();
    for i in (1..n).rev() {
        let j = (prng.next_f64() * (i + 1) as f64).floor() as usize;
        test_data.swap(i, j);
    }

    let (test_inputs, test_labels): (Vec<Vec<f64>>, Vec<f64>) = test_data.into_iter().unzip();
    Ok((test_inputs, test_labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_language_split_alignment() {
        let mut data_dir = std::path::PathBuf::from("data");
        if !data_dir.exists() && std::path::Path::new("../data").exists() {
            data_dir = std::path::PathBuf::from("../data");
        }
        let cats_dir = data_dir.join("PetImages").join("Cat");
        if !cats_dir.exists() {
            println!("Skipping split alignment test: dataset not found.");
            return;
        }

        // Collect and sort paths just like in the real implementation
        let mut paths = scan_sorted_dir(&cats_dir).unwrap();

        let extra_dir_base = std::path::Path::new("/Users/al/Projects/angelo/cats_dogs_dataset");
        if extra_dir_base.exists() {
            let extra_cats = extra_dir_base.join("cat");
            let extra_paths = scan_sorted_dir(&extra_cats).unwrap();
            paths.extend(extra_paths);
        }

        // Apply deterministic random Fisher-Yates shuffle matching the Python PyTorch dataset
        let mut shuffle_prng = Lcg::new(42);
        let num_paths = paths.len();
        for i in (1..num_paths).rev() {
            let j = (shuffle_prng.next_f64() * (i + 1) as f64).floor() as usize;
            paths.swap(i, j);
        }

        let first_3: Vec<&str> = paths.iter()
            .take(3)
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();

        println!("First 3 shuffled Cats in Rust: {:?}", first_3);
        if extra_dir_base.exists() {
            assert_eq!(first_3, vec!["10928.jpg", "10819.jpg", "5645.jpg"]);
        } else {
            assert_eq!(first_3, vec!["6202.jpg", "9357.jpg", "3342.jpg"]);
        }
    }
}

