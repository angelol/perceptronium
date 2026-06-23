# Perceptronium: Zero-Library Cat vs. Dog Neural Network Classifier in Rust

**Perceptronium** is a simple, educational multi-layer perceptron (MLP) neural network built **completely from scratch in Rust** with **zero external machine learning libraries** (no PyTorch, Candle, `ndarray`, or `tch-rs`). 

The only dependency used is the standard Rust `image` crate, solely for loading and resizing JPEG/PNG image files to a standardized grayscale representation. All neural network components—matrix math, weight initializations, activation functions, losses, backpropagation, and deterministic pseudo-random shuffling—are implemented purely using standard library vectors (`Vec<f64>`) and basic mathematical primitives.

---

## Features

- **No ML Frameworks:** Pure Rust mathematical calculations for all linear algebra and neural network layers.
- **Grayscale Resizing Pipeline:** Standardizes arbitrary input images into normalized $64 \times 64$ float arrays ($4,096$ flat features in range `[0.0, 1.0]`) using fast `thumbnail_exact` downsampling.
- **Terminal ASCII Art Previews:** Renders downscaled image previews directly inside the console as detailed, recognizable ASCII characters.
- **Deterministic Training Reproducibility:** Uses a custom-written Linear Congruential Generator (LCG) for deterministic weight initializations and training dataset shuffles.
- **Mathematical Transparency:** Clean, educational code documentations with mathematical derivations of backpropagation gradients (Binary Cross-Entropy Loss + Sigmoid pre-activation derivative simplifies analytically to $p - y$).
- **Interactive Playground CLI:** Allows users to feed *any* image from their local computer (such as a photo of their own pet) and watch the network scale it, render it in ASCII, and predict whether it is a cat or a dog.

---

## Math Primitive Specifications

### 1. Sigmoid Activation Function
Activates hidden and output layers, scaling inputs to a probability-like range `(0.0, 1.0)`:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Its derivative, used during backpropagation to distribute errors through the layers:

$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

### 2. Binary Cross-Entropy (BCE) Loss
Used to measure classification performance for binary labels ($0.0 = \text{Cat}$, $1.0 = \text{Dog}$):

$$
\mathcal{L} = - [y \ln(p) + (1 - y) \ln(1 - p)]
$$

### 3. Backpropagation Simplification
When using BCE loss coupled with a Sigmoid output activation, the partial derivative of the loss function with respect to the output pre-activation ($z_o$) simplifies analytically to:

$$
\frac{\partial \mathcal{L}}{\partial z_o} = p - y
$$

*(Where $p$ is the prediction score and $y$ is the target label, ensuring numerical stability and avoiding division-by-zero errors during training).*

---

## Installation & Requirements

To compile and run this project, you need the standard Rust toolchain (Rustup, Rustc, and Cargo) installed on your system.

If you don't have Rust installed, follow the instructions at [rustup.rs](https://rustup.rs/).

---

## How to Use

### 1. Verify and Run Tests
Run the mathematical, structural, and serialization unit tests to verify our custom Sigmoid, BCE loss, PRNG, serialization, and backpropagation mechanics:
```bash
cargo test
```

### 2. Command-Line Options
The program divides training and playground loops into distinct, fast pipelines:

- **Train the Model:** Preprocess the dataset, train from scratch, test on validation data, and persist weights:
  ```bash
  cargo run --release -- --train   # or -t
  ```
- **Instant Playground:** Startup instantly (in ~15ms) by loading saved parameters directly from `weights.txt` without processing the full dataset:
  ```bash
  cargo run --release -- --play    # or -p
  ```
- **Evaluate Model:** Load the saved `weights.txt` file and run an immediate evaluation metrics pass on the unseen testing/validation dataset:
  ```bash
  cargo run --release -- --eval    # or -e
  ```
- **Default (No arguments):** Auto-loads weights from `weights.txt` if they exist; otherwise, boots into training mode automatically.

---

## Weight Persistence (`weights.txt`)

To ensure educational transparency, the neural network's parameters are serialized as structured, comma-separated plain text rather than binary formats. 

Open `weights.txt` in any text editor to inspect the raw floating-point numbers learned by the network:
1. **Line 1:** Input size and Hidden size (`4096,32`)
2. **Line 2:** Output bias
3. **Line 3:** Hidden biases (comma-separated list of size 32)
4. **Line 4:** Output weights (comma-separated list of size 32)
5. **Remaining 32 Lines:** Each line represents the 4,096 weights connecting all inputs to a specific hidden neuron.

---

## Interactive Playground

Once you launch the playground (or complete a training run), you will be greeted by the interactive shell:
```text
================== INTERACTIVE PLAYGROUND ==================
Test the network on your own custom image! Put any .jpg/.jpeg/.png image
on your computer and provide the full path to it below.
============================================================

Enter image path (or type 'q' to quit): 
```

Provide the path to any pet picture on your machine (e.g. `/Users/yourname/Desktop/my_dog.png`), and press **Enter**. The program will:
1. Load, convert, and downscale the image to $64 \times 64$ pixels.
2. Render a stunning ASCII preview of the resized image in the terminal.
3. Compute predictions instantly using the loaded parameters and print confidence scores.

---

## Codebase Organization

The project is structured logically into four core modules:

- **[src/math.rs](src/math.rs)**: Core mathematical primitives—Sigmoid activation, its derivative, Binary Cross-Entropy loss, dot-product calculations, and our custom deterministic Linear Congruential Generator (LCG) PRNG.
- **[src/dataset.rs](src/dataset.rs)**: Dataset manager. Automatically downloads, extracts, and preloads JPEG/PNG images using `image::open` and `thumbnail_exact`, and handles ASCII rendering.
- **[src/nn.rs](src/nn.rs)**: Implements the `NeuralNetwork` struct, forward propagation caching, and gradient calculation backpropagation.
- **[src/main.rs](src/main.rs)**: Entry point of the program. Configures hyperparameters, drives the training loop, prints stats, and manages the interactive playground shell.
