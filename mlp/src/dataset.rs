use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use crate::math::Lcg;

/// Ensures that the cats and dogs dataset is downloaded and extracted.
/// If it doesn't exist, it downloads Google's filtered subset (~68MB) using `curl`
/// and extracts it using `unzip` (or `tar` if unzip isn't available).
pub fn ensure_dataset_downloaded() -> Result<PathBuf, String> {
    let mut data_dir = PathBuf::from("data");

    // If running from inside a sub-crate folder (e.g. mlp/), look one level up
    if !data_dir.exists() && Path::new("../data").exists() {
        data_dir = PathBuf::from("../data");
    }

    let dataset_dir = data_dir.join("cats_and_dogs_filtered");

    // Check if dataset already exists
    if dataset_dir.exists() {
        println!("✓ Dataset found at: {}", dataset_dir.display());
        return Ok(dataset_dir);
    }

    println!("Dataset not found. Creating 'data' folder and downloading standard Cats & Dogs subset (68MB)...");
    fs::create_dir_all(&data_dir).map_err(|e| format!("Failed to create data dir: {}", e))?;

    let zip_path = data_dir.join("cats_and_dogs_filtered.zip");

    // Download using curl (native on macOS/Linux)
    println!("Downloading dataset via curl...");
    let status = Command::new("curl")
        .args(&[
            "-L",
            "https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip",
            "-o",
            zip_path.to_str().unwrap(),
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

/// Renders a 64x64 grayscale image in the terminal as detailed ASCII art.
pub fn print_ascii_preview(pixels: &[f64], size: usize) {
    // ASCII density character set (from dark to bright)
    const ASCII_CHARS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '%', '#', '@'];
    
    // Standard terminal character cell aspect ratio is roughly 2:1 (vertical to horizontal).
    // To prevent the image from looking stretched vertically, we step by 2 rows.
    for r in (0..size).step_by(2) {
        for c in 0..size {
            let pixel = pixels[r * size + c];
            // Map [0.0, 1.0] float to [0, ASCII_CHARS.len() - 1] integer index
            let idx = (pixel * (ASCII_CHARS.len() - 1) as f64).round() as usize;
            print!("{}", ASCII_CHARS[idx]);
        }
        println!();
    }
}

/// Loads and prepares the training and testing datasets.
/// Combines cats and dogs, shuffles them deterministically using our LCG PRNG,
/// and returns splits for inputs and labels.
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

    // Paths inside Google's standard filtered dataset
    let train_cats_dir = dataset_dir.join("train").join("cats");
    let train_dogs_dir = dataset_dir.join("train").join("dogs");
    let test_cats_dir = dataset_dir.join("validation").join("cats");
    let test_dogs_dir = dataset_dir.join("validation").join("dogs");

    // Helper to read a directory up to a limit and assign a label
    let load_from_dir = |dir: &Path, limit: usize, label: f64, dest: &mut Vec<(Vec<f64>, f64)>| {
        println!("Loading up to {} images from {}...", limit, dir.display());
        let entries = fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

        let mut count = 0;
        for entry in entries {
            if count >= limit {
                break;
            }
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_file() {
                // Check if file is an image by extension
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    let ext = ext.to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" {
                        match load_image(&path, image_size) {
                            Ok(pixels) => {
                                dest.push((pixels, label));
                                count += 1;
                            }
                            Err(e) => {
                                println!("  [Warning] Skipping image {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }
        }
        println!("Loaded {} images successfully.", count);
        Ok::<(), String>(())
    };

    // Load Training sets (Label 0.0 for Cat, 1.0 for Dog)
    load_from_dir(&train_cats_dir, limit_train_per_class, 0.0, &mut train_data)?;
    load_from_dir(&train_dogs_dir, limit_train_per_class, 1.0, &mut train_data)?;

    // Load Testing sets
    load_from_dir(&test_cats_dir, limit_test_per_class, 0.0, &mut test_data)?;
    load_from_dir(&test_dogs_dir, limit_test_per_class, 1.0, &mut test_data)?;

    if train_data.is_empty() || test_data.is_empty() {
        return Err("Loaded datasets are empty. Check dataset paths and file permissions.".to_string());
    }

    // Shuffle the loaded datasets using our custom deterministic PRNG.
    // Shuffling is crucial to make sure our training batches contain a healthy,
    // alternating mix of cats and dogs, preventing the model from developing ordering bias.
    let mut prng = Lcg::new(seed);
    
    // Knuth/Fisher-Yates Shuffle
    let shuffle = |data: &mut Vec<(Vec<f64>, f64)>, prng: &mut Lcg| {
        let n = data.len();
        for i in (1..n).rev() {
            let j = (prng.next_f64() * (i + 1) as f64).floor() as usize;
            data.swap(i, j);
        }
    };

    shuffle(&mut train_data, &mut prng);
    shuffle(&mut test_data, &mut prng);

    // Split inputs and labels
    let (train_inputs, train_labels): (Vec<Vec<f64>>, Vec<f64>) = train_data.into_iter().unzip();
    let (test_inputs, test_labels): (Vec<Vec<f64>>, Vec<f64>) = test_data.into_iter().unzip();

    Ok((train_inputs, train_labels, test_inputs, test_labels))
}

/// Loads and prepares ONLY the validation/testing dataset.
/// Combines cats and dogs, shuffles them deterministically using our LCG PRNG,
/// and returns splits for inputs and labels.
pub fn load_test_split(
    dataset_dir: &Path,
    limit_test_per_class: usize,
    image_size: u32,
    seed: u32,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let mut test_data = Vec::new();

    let test_cats_dir = dataset_dir.join("validation").join("cats");
    let test_dogs_dir = dataset_dir.join("validation").join("dogs");

    let load_from_dir = |dir: &Path, limit: usize, label: f64, dest: &mut Vec<(Vec<f64>, f64)>| {
        println!("Loading up to {} images from {}...", limit, dir.display());
        let entries = fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

        let mut count = 0;
        for entry in entries {
            if count >= limit {
                break;
            }
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    let ext = ext.to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" {
                        match load_image(&path, image_size) {
                            Ok(pixels) => {
                                dest.push((pixels, label));
                                count += 1;
                            }
                            Err(e) => {
                                println!("  [Warning] Skipping image {}: {}", path.display(), e);
                            }
                        }
                    }
                }
            }
        }
        println!("Loaded {} images successfully.", count);
        Ok::<(), String>(())
    };

    load_from_dir(&test_cats_dir, limit_test_per_class, 0.0, &mut test_data)?;
    load_from_dir(&test_dogs_dir, limit_test_per_class, 1.0, &mut test_data)?;

    if test_data.is_empty() {
        return Err("Loaded validation dataset is empty. Check dataset path.".to_string());
    }

    let mut prng = Lcg::new(seed);
    let n = test_data.len();
    for i in (1..n).rev() {
        let j = (prng.next_f64() * (i + 1) as f64).floor() as usize;
        test_data.swap(i, j);
    }

    let (test_inputs, test_labels): (Vec<Vec<f64>>, Vec<f64>) = test_data.into_iter().unzip();
    Ok((test_inputs, test_labels))
}

