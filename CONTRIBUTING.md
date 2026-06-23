# Contributing to Perceptronium 🌌

First of all, thank you for taking the time to contribute to Perceptronium! We are super excited to see your ideas and improvements.

---

## 🛠️ Onboarding Setup

### 1. Rust Environment (CPU Sandboxes)
Ensure you have the standard stable Rust toolchain installed. If not, install via [rustup](https://rustup.rs/):
```bash
rustup update
```

### 2. Python Environment (PyTorch Pipelines)
Ensure you have Python 3.10+ installed. Then:
1. Create and activate a python virtual environment:
   ```bash
   python3 -m venv pytorch_cnn/.venv
   source pytorch_cnn/.venv/bin/activate
   ```
2. Install the required deep learning dependencies:
   ```bash
   pip install -r pytorch_cnn/requirements.txt
   ```

---

## 🔬 Running Local Verification

Before submitting a Pull Request, please verify your changes locally to ensure everything works beautifully:

### 1. Test the Rust Workspace
Run all unit tests and format checks to guarantee Rust matrix algebra and CNN/MLP blocks are structurally sound:
```bash
# Run unit tests
cargo test --workspace

# Check formatting compliance
cargo fmt --all -- --check
```

### 2. Verify PyTorch Pipelines
Verify your Python edits don't break neural layers:
```bash
# Run a quick forward pass verification on the dummy CPU/GPU
python3 -u pytorch_cnn/predict.py --help
```

---

## 📬 Submitting a Pull Request

1. **Fork the Repo:** Create a fork of `perceptronium` on your GitHub account.
2. **Branch Out:** Use standard branch namings (e.g. `feature/channel-attention-tweak` or `bugfix/rust-gradient-fix`).
3. **Commit with Hygiene:** Keep commits logical and descriptive.
4. **Create a Pull Request:** Submit your PR back to our `master` branch. Ensure you outline what changes were made, any performance gains, and validation metrics!
