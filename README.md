# Perceptronium 🌌

Perceptronium is an educational sandbox consisting of zero-dependency, from-scratch neural network implementations in pure Rust. The goal of this project is to explore how machine learning models function under the hood by implementing all linear algebra, forward/backward passes, and training loops completely from first principles.

No external libraries (like `ndarray`, `serde`, `clap`, or deep learning frameworks) are used for model architecture or training, keeping the codebase completely clean, readable, and highly educational. We only use the standard `image` crate solely for loading and aspect-ratio scaling of JPEG/PNG files.

---

## 📂 Project Structure

This project is organized as a Cargo workspace:

```text
perceptronium/
├── data/                             # Centralized dataset folder (shared by all approaches)
│   └── cats_and_dogs_filtered/       # ~68MB filtered dataset of dogs and cats images
├── mlp/                              # Approach 1: Multi-Layer Perceptron (Fully Connected Dense)
│   ├── src/                          # MLP training and inference logic
│   ├── weights.txt                   # Saved weights for the trained MLP model
│   └── README.md                     # MLP-specific manual and explanations
├── cnn/                              # Approach 2: Convolutional Neural Network (CNN)
│   ├── src/                          # CNN layers, backpropagation, and CLI logic
│   ├── weights.txt                   # Saved weights for the trained CNN model
│   └── README.md                     # CNN-specific manual and explanations
└── README.md                         # This file (Workspace-level overview)
```

---

## 🧠 Implemented Approaches

### 1. Multi-Layer Perceptron (MLP)
* **Location:** [`mlp/`](file:///Users/al/Projects/angelo/perceptronium/mlp)
* **Description:** A feedforward neural network utilizing standard dense layers, Sigmoid activation functions, and gradient descent via backpropagation.
* **Input Resolution:** Resizes images to $64 \times 64$ pixels in grayscale ($4,096$ input nodes).
* **Execution Commands:**
  * **Train the model:**
    ```bash
    cargo run -p perceptronium-mlp --release -- --train
    ```
  * **Evaluate the model on validation set:**
    ```bash
    cargo run -p perceptronium-mlp --release -- --eval
    ```

### 2. Convolutional Neural Network (CNN)
* **Location:** [`cnn/`](file:///Users/al/Projects/angelo/perceptronium/cnn)
* **Description:** A custom-built convolutional network featuring $3 \times 3$ kernel convolutions, ReLU activation maps, $2 \times 2$ Max Pooling, flattening, and a fully connected Sigmoid dense backpropagation classifier.
* **Input Resolution:** Resizes images to $128 \times 128$ pixels in grayscale ($16,384$ input nodes).
* **Execution Commands:**
  * **Train the model:**
    ```bash
    cargo run -p perceptronium-cnn --release -- --train
    ```
  * **Evaluate the model on validation set:**
    ```bash
    cargo run -p perceptronium-cnn --release -- --eval
    ```
  * **Interactive Playground:**
    ```bash
    cargo run -p perceptronium-cnn --release -- --play
    ```

---

## 🛠️ Workspace Commands

You can run workspace-level commands from the root directory:

### Run Unit Tests
To run all tests in the workspace (all approaches):
```bash
cargo test
```

### Run a Specific Subproject
To run a specific approach, use the `-p` (package) flag followed by the package name:
```bash
cargo run -p perceptronium-mlp --release -- [ARGS]
```
or
```bash
cargo run -p perceptronium-cnn --release -- [ARGS]
```
