# Perceptronium Workspace Guidelines

These guidelines are automatically loaded and followed by Antigravity (and all spawned subagents) when working in the Perceptronium workspace.

---

## 🚀 Execution & Logging

### 1. Unbuffered Realtime Logging (CRITICAL)
* Whenever launching a Python script as a background process or redirecting standard output to log files, **always** run python with the unbuffered flag (`-u`):
  ```bash
  python -u path/to/script.py
  ```
* Ensure all critical print statements in custom Python scripts include explicit flushing (`print(..., flush=True)`) to guarantee real-time log updates and eliminate OS block-buffering delays.

---

## 🔬 Model & Training Principles

### 1. Pure Scratch Mandate
* Always train models completely from scratch (`weights=None` or no pre-trained backbones). Transfer learning and loaded ImageNet weights are strictly disallowed unless explicitly requested by the user.

### 2. Strict Deterministic Splits
* Maintain perfect alignment between training (8,000 images) and validation (2,000 images) splits. Always sort file paths alphabetically prior to split slicing to ensure zero data leakage across separate training runs or languages (Rust vs. PyTorch).

---

## 🧹 Workspace & Git Hygiene

### 1. Keep Workspace Clean (No Garbage)
* Never leave garbage, temporary test scripts, unused checkpoints, or raw debug logs in the workspace. Always clean up temporary files immediately before ending a turn. If temporary files or scratch scripts are absolutely necessary during a turn, ensure they are added to `.gitignore` and deleted immediately as soon as they are no longer needed.

### 2. Clean Git Commits
* Always commit completed code changes to Git at the end of each turn. Ensure commit messages are clean, professional, and descriptive. Never commit broken code, scratch files, or binary weight checkpoint backups (like `.pth` files) unless explicitly asked.

### 3. Maintain Accuracy Log (CRITICAL)
* Always update [accuracy_log.md](file:///Users/al/Projects/angelo/perceptronium/accuracy_log.md) at the end of any training or model iteration run. Record the date, architectural adjustments, exact hyperparameter configurations, and key metric benchmarks to preserve absolute comparative context over time.

