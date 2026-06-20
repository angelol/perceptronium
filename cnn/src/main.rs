use std::io::{self, Write};
use std::path::Path;

mod math;
mod dataset;
mod nn;

const WEIGHTS_FILE: &str = "weights.txt";
const IMAGE_SIZE: u32 = 128;

fn main() -> Result<(), String> {
    // -------------------------------------------------------------------------
    // 1. Welcome & ASCII Art Banner
    // -------------------------------------------------------------------------
    println!(r#"
==========================================================================
   ____   _____  ____    ____  _____ ____  _____  ____   _  _   _  _  _  _ 
  |  _ \ | ____||  _ \  / ___|| ____|  _ \|_   _||  _ \ | || | | || || || |
  | |_) ||  _|  | |_) || |    |  _|  | |_) | | |  | |_) || || |_| || || || |
  |  __/ | |___ |  _ < | |___ | |___ |  __/  | |  |  _ < |__   _||_||_||_||_|
  |_|    |_____||_| \_\ \____||_____||_|     |_|  |_| \_\   |_|  (_)(_)(_)(_)
                                                                          
       PERCEPTRONIUM-CNN: Zero-Library Cat vs. Dog Convolutional Network
                       Written from scratch in Rust
==========================================================================
"#);

    println!("Welcome! This program is an educational demonstration of an image-based");
    println!("Convolutional Neural Network (CNN) implemented completely from scratch");
    println!("without any machine learning libraries (no PyTorch, Candle, or ndarray).");
    println!("We only use the standard 'image' crate to load and downscale pictures.");
    println!("--------------------------------------------------------------------------\n");

    let weights_path_buf = get_weights_path();
    let weights_path = weights_path_buf.as_path();

    // Parse command line arguments using pure standard library
    let args: Vec<String> = std::env::args().collect();
    let is_train = args.contains(&"--train".to_string()) || args.contains(&"-t".to_string());
    let is_play = args.contains(&"--play".to_string()) || args.contains(&"-p".to_string());
    let is_eval = args.contains(&"--eval".to_string()) || args.contains(&"-e".to_string());

    if is_train {
        run_training(weights_path)
    } else if is_play {
        if !weights_path.exists() {
            return Err(format!(
                "❌ Weights file '{}' not found. Please train the network first using:\n\n   cargo run -p perceptronium-cnn --release -- --train\n",
                weights_path.display()
            ));
        }
        run_playground(weights_path)
    } else if is_eval {
        run_evaluation(weights_path)
    } else {
        // No explicit command provided: run a smart default behavior
        if weights_path.exists() {
            println!("💡 Saved model weights found at '{}'. Automatically launching into interactive playground...", weights_path.display());
            println!("(To retrain the network from scratch, run: cargo run -p perceptronium-cnn --release -- --train)\n");
            run_playground(weights_path)
        } else {
            println!("💡 No saved model weights found. Automatically defaulting to model training mode...");
            println!("(Next time, once trained, you can start the playground instantly using --play!)\n");
            run_training(weights_path)
        }
    }
}

/// Dynamically resolves the weights file path, supporting workspace-level and subproject run environments.
fn get_weights_path() -> std::path::PathBuf {
    let default_path = std::path::PathBuf::from(WEIGHTS_FILE);
    // If weights.txt exists locally, use it
    if default_path.exists() {
        return default_path;
    }
    // If run from workspace root, prefer cnn/weights.txt if it exists
    let cnn_path = std::path::Path::new("cnn").join(WEIGHTS_FILE);
    if cnn_path.exists() {
        return cnn_path;
    }
    // Fallback: if we are at workspace root but cnn/weights.txt doesn't exist yet (e.g. before training),
    // save weights inside cnn/ to keep workspace root clean.
    if std::path::Path::new("cnn").is_dir() {
        return cnn_path;
    }
    default_path
}

/// Runs the full dataset preloading, model weight initialization, training, testing evaluation, and saving weights.
fn run_training(weights_path: &Path) -> Result<(), String> {
    let dataset_dir = dataset::ensure_dataset_downloaded()?;
    println!();

    // -------------------------------------------------------------------------
    // Hyperparameters
    // -------------------------------------------------------------------------
    let learning_rate: f64 = 0.03;
    let epochs: usize = 15;
    let seed: u32 = 42;

    // Use the entire dataset for a comprehensive training run
    let limit_train_per_class = 4000; // 8000 training images total
    let limit_test_per_class = 1000;  // 2000 test images total

    println!("------------------- HYPERPARAMETERS -------------------");
    println!("  • Resolution:      {} x {} pixels ({} float inputs)", IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE * IMAGE_SIZE);
    println!("  • Conv Layer 1:    {} filters of size 3x3 with ReLU", nn::FILTERS1);
    println!("  • Max Pooling 1:   2x2 with stride 2");
    println!("  • Conv Layer 2:    {} filters of size 3x3x{} with ReLU", nn::FILTERS2, nn::FILTERS1);
    println!("  • Max Pooling 2:   2x2 with stride 2 (cropped)");
    println!("  • Dense Layer:     {} neurons with ReLU ({} inputs)", nn::DENSE_NEURONS, nn::FLATTENED_SIZE);
    println!("  • Output Layer:    1 neuron with Sigmoid");
    println!("  • Regularization:  L2 weight decay with lambda = {}", nn::L2_REG);
    println!("  • Learning Rate:   {}", learning_rate);
    println!("  • Epochs:          {}", epochs);
    println!("  • Random Seed:     {} (determines weight initialization)", seed);
    println!("  • Training Set:    {} cats + {} dogs = {} images (with random horizontal flip augmentation)", limit_train_per_class, limit_train_per_class, limit_train_per_class * 2);
    println!("  • Testing Set:     {} cats + {} dogs = {} images", limit_test_per_class, limit_test_per_class, limit_test_per_class * 2);
    println!("-------------------------------------------------------");
    println!("Loading images and converting to grayscale vectors...");

    // Load & Preprocess Dataset
    let (train_inputs, train_labels, test_inputs, test_labels) = dataset::load_split(
        &dataset_dir,
        limit_train_per_class,
        limit_test_per_class,
        IMAGE_SIZE,
        seed,
    )?;

    println!("\n✓ Datasets loaded successfully.");

    // Display a random image from our shuffled training set as ASCII art.
    let preview_idx = 0;
    let preview_label = if train_labels[preview_idx] == 1.0 { "DOG" } else { "CAT" };
    println!("\nASCII rendering of training sample #{} (Labeled as: {}):", preview_idx + 1, preview_label);
    println!("================================================================");
    dataset::print_ascii_preview(&train_inputs[preview_idx], IMAGE_SIZE as usize);
    println!("================================================================");
    println!("(Notice how details like ears and outline are visible even at 128x128!)");

    // Initialize Neural Network
    println!("\nInitializing network weights using Xavier uniform initialization...");
    let mut nn = nn::ConvolutionalNetwork::new(seed);
    println!("✓ Convolutional Network successfully initialized!");

    // Training Loop
    println!("\nStarting training loop (Stochastic Gradient Descent)...");
    println!("-------------------------------------------------------------------------------------------");

    let num_samples = train_inputs.len();
    let mut rng = math::Lcg::new(seed + 100);

    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let mut correct = 0;

        for i in 0..num_samples {
            let original_inputs = &train_inputs[i];
            let target = train_labels[i];

            // Randomly apply horizontal flip data augmentation with a 50% probability
            let use_flip = rng.next_f64() < 0.5;
            let flipped_inputs;
            let inputs = if use_flip {
                flipped_inputs = math::flip_horizontal(original_inputs, IMAGE_SIZE as usize);
                &flipped_inputs
            } else {
                original_inputs
            };

            // Forward prediction check
            let prediction = nn.predict(inputs);
            let predicted_class = if prediction >= 0.5 { 1.0 } else { 0.0 };
            if predicted_class == target {
                correct += 1;
            }

            // Perform Backpropagation and parameter updates
            let loss = nn.backprop(inputs, target, learning_rate);
            epoch_loss += loss;
        }

        let avg_loss = epoch_loss / num_samples as f64;
        let accuracy = (correct as f64 / num_samples as f64) * 100.0;

        // Validation/Test evaluation at the end of each epoch
        let mut test_epoch_correct = 0;
        let mut test_epoch_loss = 0.0;
        let num_test_samples = test_inputs.len();

        for j in 0..num_test_samples {
            let t_inputs = &test_inputs[j];
            let t_target = test_labels[j];

            let prediction = nn.predict(t_inputs);
            let loss = math::binary_cross_entropy(prediction, t_target);
            test_epoch_loss += loss;

            let predicted_class = if prediction >= 0.5 { 1.0 } else { 0.0 };
            if predicted_class == t_target {
                test_epoch_correct += 1;
            }
        }

        let avg_test_loss = test_epoch_loss / num_test_samples as f64;
        let test_accuracy = (test_epoch_correct as f64 / num_test_samples as f64) * 100.0;

        println!(
            "  Epoch {:02}/{} | Train Loss: {:.5} | Train Acc: {:5.2}% | Test Loss: {:.5} | Test Acc: {:5.2}%",
            epoch, epochs, avg_loss, accuracy, avg_test_loss, test_accuracy
        );
    }
    println!("-------------------------------------------------------------------------------------------");
    println!("✓ Training complete!");

    // Evaluation Phase
    println!("\nEvaluating network on unseen test dataset...");
    let mut test_correct = 0;
    let mut test_loss = 0.0;
    let test_samples = test_inputs.len();

    for i in 0..test_samples {
        let inputs = &test_inputs[i];
        let target = test_labels[i];

        let prediction = nn.predict(inputs);
        let loss = math::binary_cross_entropy(prediction, target);
        test_loss += loss;

        let predicted_class = if prediction >= 0.5 { 1.0 } else { 0.0 };
        if predicted_class == target {
            test_correct += 1;
        }
    }

    let avg_test_loss = test_loss / test_samples as f64;
    let test_accuracy = (test_correct as f64 / test_samples as f64) * 100.0;

    println!("--------------------- EVALUATION RESULTS ---------------------");
    println!("  • Unseen Test Samples:   {}", test_samples);
    println!("  • Average BCE Test Loss: {:.5}", avg_test_loss);
    println!("  • Final Test Accuracy:   {:.2}%", test_accuracy);
    println!("--------------------------------------------------------------");

    // Save Weights
    println!("\nSaving trained network weights to '{}'...", weights_path.display());
    nn.save_to_file(weights_path)?;
    println!("✓ Weights successfully preserved locally!");
    println!("You can now start the playground instantly without training using:");
    println!("   cargo run -p perceptronium-cnn --release -- --play\n");

    // Transition to interactive playground
    run_playground_loop(nn)
}

/// Loads the saved model state, prepares validation data, and evaluates model accuracy.
fn run_evaluation(weights_path: &Path) -> Result<(), String> {
    if !weights_path.exists() {
        return Err(format!(
            "❌ Weights file '{}' not found. Please train the network first using:\n\n   cargo run -p perceptronium-cnn --release -- --train\n",
            weights_path.display()
        ));
    }

    println!("Restoring trained model weights from '{}'...", weights_path.display());
    let mut nn = nn::ConvolutionalNetwork::load_from_file(weights_path)?;
    println!("✓ Model successfully loaded!");

    let dataset_dir = dataset::ensure_dataset_downloaded()?;
    println!();

    let limit_train_per_class = 4000; // Match training subset size to skip those images
    let limit_test_per_class = 1000;  // 2000 validation images total
    let seed = 42;

    println!("Loading unseen testing/validation dataset...");
    let (test_inputs, test_labels) = dataset::load_test_split(
        &dataset_dir,
        limit_train_per_class,
        limit_test_per_class,
        IMAGE_SIZE,
        seed,
    )?;

    println!("\nEvaluating network on unseen test dataset...");
    let mut test_correct = 0;
    let mut test_loss = 0.0;
    let test_samples = test_inputs.len();

    for i in 0..test_samples {
        let inputs = &test_inputs[i];
        let target = test_labels[i];

        let prediction = nn.predict(inputs);
        let loss = math::binary_cross_entropy(prediction, target);
        test_loss += loss;

        let predicted_class = if prediction >= 0.5 { 1.0 } else { 0.0 };
        if predicted_class == target {
            test_correct += 1;
        }
    }

    let avg_test_loss = test_loss / test_samples as f64;
    let test_accuracy = (test_correct as f64 / test_samples as f64) * 100.0;

    println!("===================== EVALUATION RESULTS =====================");
    println!("  • Model Location:        {}", weights_path.display());
    println!("  • Unseen Test Samples:   {}", test_samples);
    println!("  • Average BCE Test Loss: {:.5}", avg_test_loss);
    println!("  • Final Test Accuracy:   {:.2}%", test_accuracy);
    println!("==============================================================");

    Ok(())
}

/// Loads the saved model state and triggers the playground loop.
fn run_playground(weights_path: &Path) -> Result<(), String> {
    println!("Restoring trained model weights from '{}'...", weights_path.display());
    let nn = nn::ConvolutionalNetwork::load_from_file(weights_path)?;
    println!("✓ Model successfully loaded!");
    run_playground_loop(nn)
}

/// Core interactive playground CLI loop.
fn run_playground_loop(mut nn: nn::ConvolutionalNetwork) -> Result<(), String> {
    println!("\n================== INTERACTIVE PLAYGROUND ==================");
    println!("Test the network on your own custom image! Put any .jpg/.jpeg/.png image");
    println!("on your computer and provide the full path to it below.");
    println!("============================================================");

    let mut input_buffer = String::new();
    loop {
        input_buffer.clear();
        print!("\nEnter image path (or type 'q' to quit): ");
        io::stdout().flush().unwrap();

        match io::stdin().read_line(&mut input_buffer) {
            Ok(_) => {
                let path_str = input_buffer.trim();
                if path_str.eq_ignore_ascii_case("q") || path_str.eq_ignore_ascii_case("quit") {
                    println!("Exiting playground. Goodbye!");
                    break;
                }

                if path_str.is_empty() {
                    continue;
                }

                let custom_path = Path::new(path_str);
                if !custom_path.exists() {
                    println!("❌ File not found at '{}'. Please double-check the path.", path_str);
                    continue;
                }

                println!("Loading and processing custom image (resizing to 128x128 grayscale)...");
                match dataset::load_image(custom_path, IMAGE_SIZE) {
                    Ok(pixels) => {
                        println!("\nASCII rendering of processed image:");
                        println!("================================================================");
                        dataset::print_ascii_preview(&pixels, IMAGE_SIZE as usize);
                        println!("================================================================");

                        print!("Evaluating network prediction... ");
                        io::stdout().flush().unwrap();

                        let probability = nn.predict(&pixels);
                        let predicted_label = if probability >= 0.5 { "DOG 🐶" } else { "CAT 🐱" };
                        let confidence = if probability >= 0.5 { probability } else { 1.0 - probability } * 100.0;

                        println!("Done!");
                        println!("----------------------- PREDICTION -----------------------");
                        println!("  • Predicted Class:    {}", predicted_label);
                        println!("  • Dog Probability:     {:.4}", probability);
                        println!("  • Model Confidence:   {:.2}%", confidence);
                        println!("----------------------------------------------------------");
                    }
                    Err(e) => {
                        println!("❌ Failed to process image: {}", e);
                    }
                }
            }
            Err(e) => {
                return Err(format!("Failed to read line: {}", e));
            }
        }
    }

    Ok(())
}
