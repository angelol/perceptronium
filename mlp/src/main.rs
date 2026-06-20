use std::io::{self, Write};
use std::path::Path;

mod math;
mod dataset;
mod nn;

const WEIGHTS_FILE: &str = "weights.txt";
const IMAGE_SIZE: u32 = 64;

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
                                                                          
       PERCEPTRONIUM: Zero-Library Cat vs. Dog Neural Network Classifier
                      Written from scratch in Rust
==========================================================================
"#);

    println!("Welcome! This program is an educational demonstration of an image-based");
    println!("neural network (Multi-Layer Perceptron) implemented completely from scratch");
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
                "❌ Weights file '{}' not found. Please train the network first using:\n\n   cargo run --release -- --train\n",
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
            println!("(To retrain the network from scratch, run: cargo run --release -- --train)\n");
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
    // If run from workspace root, prefer mlp/weights.txt if it exists
    let mlp_path = std::path::Path::new("mlp").join(WEIGHTS_FILE);
    if mlp_path.exists() {
        return mlp_path;
    }
    // Fallback: if we are at workspace root but mlp/weights.txt doesn't exist yet (e.g. before training),
    // save weights inside mlp/ to keep workspace root clean.
    if std::path::Path::new("mlp").is_dir() {
        return mlp_path;
    }
    default_path
}


/// Runs the full dataset preloading, model weight initialization, training, testing evaluation, and saving weights.
fn run_training(weights_path: &Path) -> Result<(), String> {
    // -------------------------------------------------------------------------
    // 2. Ensure Dataset is Downloaded
    // -------------------------------------------------------------------------
    let dataset_dir = dataset::ensure_dataset_downloaded()?;
    println!();

    // -------------------------------------------------------------------------
    // 3. Hyperparameters
    // -------------------------------------------------------------------------
    let hidden_size: usize = 32; // 32 neurons in the hidden layer
    let learning_rate: f64 = 0.05;
    let epochs: usize = 150;
    let seed: u32 = 42; // Deterministic seed for reproducible weights and shuffles

    // We'll load the full dataset (1,000 images per class for training, 250 for testing)
    let limit_train_per_class = 1000; // 2000 training images total
    let limit_test_per_class = 250;   // 500 test images total

    println!("------------------- HYPERPARAMETERS -------------------");
    println!("  • Resolution:      {} x {} pixels ({} float inputs)", IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE * IMAGE_SIZE);
    println!("  • Hidden Layer:    {} neurons ({} weights, {} biases)", hidden_size, (IMAGE_SIZE * IMAGE_SIZE) as usize * hidden_size, hidden_size);
    println!("  • Output Layer:    1 neuron ({} weights, 1 bias)", hidden_size);
    println!("  • Learning Rate:   {}", learning_rate);
    println!("  • Epochs:          {}", epochs);
    println!("  • Random Seed:     {} (determines weight initialization)", seed);
    println!("  • Training Set:    {} cats + {} dogs = {} images", limit_train_per_class, limit_train_per_class, limit_train_per_class * 2);
    println!("  • Testing Set:     {} cats + {} dogs = {} images", limit_test_per_class, limit_test_per_class, limit_test_per_class * 2);
    println!("-------------------------------------------------------");
    println!("Loading images and converting to grayscale vectors...");

    // -------------------------------------------------------------------------
    // 4. Load & Preprocess Dataset
    // -------------------------------------------------------------------------
    let (train_inputs, train_labels, test_inputs, test_labels) = dataset::load_split(
        &dataset_dir,
        limit_train_per_class,
        limit_test_per_class,
        IMAGE_SIZE,
        seed,
    )?;

    println!("\n✓ Datasets loaded successfully.");

    // Display a random image from our shuffled training set as ASCII art.
    // This lets the user see exactly what the downscaled image looks like to the network!
    let preview_idx = 0;
    let preview_label = if train_labels[preview_idx] == 1.0 { "DOG" } else { "CAT" };
    println!("\nASCII rendering of training sample #{} (Labeled as: {}):", preview_idx + 1, preview_label);
    println!("================================================================");
    dataset::print_ascii_preview(&train_inputs[preview_idx], IMAGE_SIZE as usize);
    println!("================================================================");
    println!("(Notice how details like ears and outline are visible even at 64x64!)");

    // -------------------------------------------------------------------------
    // 5. Initialize Neural Network
    // -------------------------------------------------------------------------
    println!("\nInitializing network weights using Xavier uniform initialization...");
    let mut nn = nn::NeuralNetwork::new(
        (IMAGE_SIZE * IMAGE_SIZE) as usize,
        hidden_size,
        seed,
    );
    println!("✓ Neural network successfully initialized!");

    // -------------------------------------------------------------------------
    // 6. Training Loop
    // -------------------------------------------------------------------------
    println!("\nStarting training loop (Stochastic Gradient Descent)...");
    println!("-------------------------------------------------------------");

    let num_samples = train_inputs.len();
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let mut correct = 0;

        for i in 0..num_samples {
            let inputs = &train_inputs[i];
            let target = train_labels[i];

            // Evaluate current prediction (before weight update)
            let prediction = nn.predict(inputs);
            let predicted_class = if prediction >= 0.5 { 1.0 } else { 0.0 };
            if predicted_class == target {
                correct += 1;
            }

            // Perform Stochastic Gradient Descent (SGD) update for this sample
            let loss = nn.train_sample(inputs, target, learning_rate);
            epoch_loss += loss;
        }

        let avg_loss = epoch_loss / num_samples as f64;
        let accuracy = (correct as f64 / num_samples as f64) * 100.0;

        // Print training progress every epoch (or skip to make console neat, but every epoch is very rewarding!)
        if epoch == 1 || epoch % 5 == 0 || epoch == epochs {
            println!(
                "  Epoch {:03}/{} | Avg BCE Loss: {:.5} | Training Accuracy: {:5.2}%",
                epoch, epochs, avg_loss, accuracy
            );
        }
    }
    println!("-------------------------------------------------------------");
    println!("✓ Training complete!");

    // -------------------------------------------------------------------------
    // 7. Evaluation Phase
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // 8. Save Weights
    // -------------------------------------------------------------------------
    println!("\nSaving trained network weights to '{}'...", weights_path.display());
    nn.save_to_file(weights_path)?;
    println!("✓ Weights successfully preserved locally!");
    println!("You can now start the playground instantly without training using:");
    println!("   cargo run --release -- --play\n");

    // Seamlessly transition directly into the interactive playground with the trained network
    run_playground_loop(nn)
}

/// Loads the saved model state, prepares validation data, and evaluates model accuracy.
fn run_evaluation(weights_path: &Path) -> Result<(), String> {
    if !weights_path.exists() {
        return Err(format!(
            "❌ Weights file '{}' not found. Please train the network first using:\n\n   cargo run --release -- --train\n",
            weights_path.display()
        ));
    }

    println!("Restoring trained model weights from '{}'...", weights_path.display());
    let nn = nn::NeuralNetwork::load_from_file(weights_path)?;
    println!("✓ Model successfully loaded!");

    // Ensure dataset downloaded
    let dataset_dir = dataset::ensure_dataset_downloaded()?;
    println!();

    let limit_test_per_class = 250; // 500 validation images total
    let seed = 42;

    println!("Loading unseen testing/validation dataset...");
    let (test_inputs, test_labels) = dataset::load_test_split(
        &dataset_dir,
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

    println!("\n===================== EVALUATION RESULTS =====================");
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
    let nn = nn::NeuralNetwork::load_from_file(weights_path)?;
    println!("✓ Model successfully loaded!");
    run_playground_loop(nn)
}

/// Core interactive playground CLI loop.
fn run_playground_loop(nn: nn::NeuralNetwork) -> Result<(), String> {
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

                let path = Path::new(path_str);
                if !path.exists() {
                    println!("❌ File not found at path: {}. Please check the path and try again.", path_str);
                    continue;
                }

                println!("Processing image: {}...", path.display());
                match dataset::load_image(path, IMAGE_SIZE) {
                    Ok(pixels) => {
                        println!("\nASCII rendering of your image resized to {}x{}:", IMAGE_SIZE, IMAGE_SIZE);
                        println!("----------------------------------------------------------------");
                        dataset::print_ascii_preview(&pixels, IMAGE_SIZE as usize);
                        println!("----------------------------------------------------------------");

                        // Predict
                        let prediction = nn.predict(&pixels);
                        let cat_prob = (1.0 - prediction) * 100.0;
                        let dog_prob = prediction * 100.0;

                        println!("\nNeural Network Feedforward Pass Results:");
                        println!("  🐱 Cat Confidence: {:.2}%", cat_prob);
                        println!("  🐶 Dog Confidence: {:.2}%", dog_prob);

                        if prediction >= 0.5 {
                            println!("\nVerdict: The network classifies this image as a **DOG** 🐶 (score: {:.4})", prediction);
                        } else {
                            println!("\nVerdict: The network classifies this image as a **CAT** 🐱 (score: {:.4})", prediction);
                        }
                    }
                    Err(e) => {
                        println!("❌ Error loading or preprocessing image: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("❌ Error reading input: {}", e);
            }
        }
    }

    Ok(())
}
