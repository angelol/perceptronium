# Perceptronium 🌌

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust CI](https://github.com/angelol/perceptronium/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/angelol/perceptronium/actions)
[![Python CI](https://github.com/angelol/perceptronium/actions/workflows/python-ci.yml/badge.svg)](https://github.com/angelol/perceptronium/actions)

**Perceptronium** is an educational deep learning sandbox tracing the evolution of image classifiers from first principles. It guides you from building zero-dependency CPU-bound neural networks in pure Rust to training state-of-the-art hybrid Attention-Transformer models on GPUs, culminating in an exquisite local Web Explorer dashboard.

---

## 🧭 The Evolution of Perceptronium

```text
┌─────────────────────────────────┐      ┌──────────────────────────────────┐      ┌─────────────────────────────────┐
│     Phase 1: Pure Rust CPU      │ ───> │   Phase 2: PyTorch GPU (MPS)     │ ───> │     Phase 3: Web Dashboard      │
│  Zero-dependency feedforward    │      │  CBAM-EfficientNet Attention     │      │   Glassmorphic real-time local  │
│  MLP and CNN from scratch       │      │  regularized SWA model (93.3%)   │      │  inference web server explorer  │
└─────────────────────────────────┘      └──────────────────────────────────┘      └─────────────────────────────────┘
```

---

## 🚀 Key Features

* **🦀 Phase 1: Pure Rust CPU Sandboxes (`mlp/` & `cnn/`)**
  * Complete zero-dependency implementations of fully connected Layer networks (MLP) and Convolutional Neural Networks (CNN) with backpropagation.
  * Implemented linear algebra, convolutional sliding kernels, max-pooling, L2 regularization, and image transformations in raw Rust with no third-party linear algebra libraries.
* **🔥 Phase 2: PyTorch Hardware-Accelerated Pipelines (`pytorch_cnn/`)**
  * Deployed an advanced, 24.31M parameter **CBAM-EfficientNet v4** block architecture featuring Depthwise Inverted Bottlenecks, Convolutional Block Attention Modules (CBAM), and Stochastic Depth.
  * Accelerated via macOS **Metal Performance Shaders (MPS)** and cloud-based NVIDIA **CUDA** backends.
  * Uses advanced regularizations: Stochastic Weight Averaging (SWA), Post-hoc Probability Temperature Calibration, Test-Time Augmentation (TTA), and CutMix/Mixup data-blending.
* **🌐 Phase 3: Exquisite Cybernetic Local Web UI (`pytorch_cnn/web_server.py`)**
  * A self-contained, zero-dependency local server written utilizing standard Python libraries.
  * Hosts a premium, glassmorphic dark-themed user interface to interact with the trained neural model in real-time.
  * Features high-performance binary-byte streaming uploads, predicted class glowing badges, and an interactive continuous dog-vs-cat probability slider scale.

---

## 📂 Repository Structure

```text
perceptronium/
├── mlp/                  # Rust CPU Multi-Layer Perceptron (Fully Connected)
│   ├── src/              # Raw Rust matrix algebra & backpropagation
│   └── README.md         # Mathematical overview and CLI guide
├── cnn/                  # Rust CPU Convolutional Neural Network
│   ├── src/              # Raw sliding kernel convs, max pooling, & backpropagation
│   └── README.md         # CNN layers & CLI guide
├── pytorch_cnn/          # PyTorch Accelerated GPU Pipeline & Web UI
│   ├── model.py          # CBAM-EfficientNet attention-transformer architecture
│   ├── train.py          # PyTorch training pipeline with CutMix, Mixup, and OneCycleLR
│   ├── predict.py        # ASCII terminal art visualizer & CLI predictor
│   ├── predict_ensemble.py # SWA snapshots & Calibrated ensembling engine
│   └── web_server.py     # Zero-dependency HTTP server & Glassmorphic UI dashboard
└── README.md             # This file (Project landing page)
```

---

## 🛠️ Quick Start

### 1. Clone & Set Up
```bash
git clone https://github.com/angelol/perceptronium.git
cd perceptronium
```

### 2. Run Pure Rust Sandboxes
```bash
# Run the zero-dependency Rust CNN validation playground
cargo run -p perceptronium-cnn --release -- --play
```

### 3. Start PyTorch Web UI Explorer
To experience the GPU-accelerated local model explorer (make sure to download weights as instructed):
```bash
# Start the web server
python3 pytorch_cnn/web_server.py
```
Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your browser!

---

## 🔬 Performance Comparison

| Model | Hardware | Input Size | Params | Val Acc | Test Acc | Regularizations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rust MLP Baseline** | CPU (Single-thread) | 64x64 Gray | ~131K | 50.00% | 50.00% | None |
| **Rust CNN Baseline** | CPU (Single-thread) | 128x128 Gray | ~924K | 68.35% | 66.90% | L2 Weight Decay |
| **Custom Residual** | Metal GPU (MPS) | 224x224 RGB | ~4.91M | 87.60% | 87.60% | Dropout, Weight Decay |
| **CBAM-EfficientNet v2**| Metal GPU (MPS) | 224x224 RGB | ~8.69M | 93.30% | 92.35% (SWA) | SWA, Mixup, CutMix, Calibration |
| **CBAM-EfficientNet v4**| GCP L4 GPU (CUDA) | 288x288 RGB | ~24.31M| **98.96%** | **98.63%** (SWA) | Transformer Stage, Progressive Resizing, TTA |

---

## 📜 License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
