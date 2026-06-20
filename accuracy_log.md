# 📈 Perceptronium Accuracy Log

This log tracks our model architectures, training runs, hyperparameter choices, and accuracy progress over time as we refine our Cat vs. Dog classifiers.

---

## 🏆 Current Accuracy Leaderboard

| Rank | Model Identifier | Architecture Style | Total Params | Best Test Acc | Final Test Acc | Overfitting Status | Key Innovations |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **1** | **Option A+ (Residual CNN)** | Residual feedforward CNN | ~4.91M | **87.60%** | **87.60%** | Minimal | Skip connections, RandomResizedCrop, Cosine Annealing |
| **2** | **Option C (CBAM-EfficientNet)** | Depthwise Attention Bottleneck | ~1.28M | **84.90%** | **84.90%** | Negative (None) | CBAM Attention, MBConv stages, OneCycleLR, Mixup/CutMix |
| **3** | **Option A (Custom CNN)** | Standard feedforward CNN | ~441K | **76.15%** | **74.50%** | Minimal | AdaptiveAvgPool, BatchNorm, Dropout |
| **4** | **Rust CNN Baseline** | From-scratch CPU CNN | ~924K | **68.35%** | **66.90%** | Moderate | 2 Conv, 2 Pool, 1 Dense, horizontal flips in Rust |
| **5** | **Rust MLP Baseline** | From-scratch CPU MLP | ~131K | **~50.00%** | **~50.00%** | High Bias | Fully connected nodes, no spatial representation |

---

## 📝 Detailed Training Run Records

### Run 5: CBAM-EfficientNet (Option C)
* **Date:** June 20, 2026
* **Hardware Device:** Metal GPU (MPS)
* **Input Resolution:** $224 \times 224$ RGB (3 channels)
* **Architecture Highlights:**
  * Stem convolution followed by 4 `MBConvCBAM` inverted bottleneck stages with depthwise separable convolutions (factorizing spatial/channel calculations).
  * Dual-attention mechanisms: **Channel Attention** (global average and max spatial pooling through shared MLPs) and **Spatial Attention** ($7 \times 7$ filters over spatial pooling dimensions).
  * Feature compression using Global `AdaptiveAvgPool2d((1, 1))` followed by double Dropout layers (`0.4`/`0.2`).
* **Hyperparameters & Regularization:**
  * **Optimizer:** `AdamW` (learning rate peaked at `1e-3` via `OneCycleLR` scheduling, weight decay `0.01`).
  * **Loss Function:** `LabelSmoothingBCEWithLogitsLoss` ($\epsilon = 0.05$).
  * **Data-level Regularization:** Probabilistic batch-level **Mixup** (30% prob, $\alpha = 0.2$) and **CutMix** (30% prob, $\alpha = 1.0$).
  * **Evaluation:** Horizontally-flipped **Test-Time Augmentation (TTA)**.
* **Epoch-by-Epoch Convergence:**
  ```
  | Epoch 01 | Train Loss: 0.6735 | Test Loss: 0.6446 | Test Acc: 61.10% |
  | Epoch 05 | Train Loss: 0.5891 | Test Loss: 0.5686 | Test Acc: 72.65% |
  | Epoch 10 | Train Loss: 0.5249 | Test Loss: 0.4876 | Test Acc: 78.75% |
  | Epoch 15 | Train Loss: 0.4735 | Test Loss: 0.4387 | Test Acc: 82.25% |
  | Epoch 20 | Train Loss: 0.4470 | Test Loss: 0.4016 | Test Acc: 84.90% |
  ```
* **Best Test Accuracy:** **`84.90%`** (Epoch 20)
* **Overfitting / Generalization Analysis:**
  * **Result:** No overfitting (Approximate Train Acc `83.20%` on mixed images vs Clean Test Acc `84.90%`). 
  * **Insights:** Mixup and CutMix make training targets highly noisy, acting as a massive regularizer. While they lower training accuracy, they smooth out the decision boundaries. This model generalized exceptionally well with 4x fewer parameters than Option A+, though it would likely scale higher with more epochs (as mixed-image training standardly converges over 50-100 epochs).

---

### Run 4: Custom Residual CNN (Option A+)
* **Date:** June 20, 2026
* **Hardware Device:** Metal GPU (MPS)
* **Input Resolution:** $224 \times 224$ RGB (3 channels)
* **Architecture Highlights:**
  * 19-layer network stacking 4 stages of dual-conv `ResidualBlock` structures. 
  * Skip-connections feeding identity projections directly to outputs to resolve gradient vanishing.
* **Hyperparameters & Regularization:**
  * **Optimizer:** `AdamW` (initial LR `1e-3` decaying smoothly to `1e-5` via `CosineAnnealingLR`, weight decay `0.01`).
  * **Data-level Regularization:** `RandomResizedCrop` and rich color jitter.
* **Best Test Accuracy:** **`87.60%`** (Epoch 20)
* **Best Validation Loss:** **`0.2907`** (BCE loss)
* **Overfitting / Generalization Analysis:**
  * **Result:** Near-zero overfitting (Train Acc `89.56%` vs Test Acc `87.60%`).
  * **Insights:** The introduction of residual skip connections and scale-invariant augmentations allowed the model to represent fine animal anatomy (e.g., ear forms, muzzles) far better than shallow networks without over-indexing on absolute position.

---

### Run 3: PyTorch Custom CNN (Option A)
* **Date:** June 20, 2026
* **Hardware Device:** Metal GPU (MPS)
* **Input Resolution:** $224 \times 224$ RGB (3 channels)
* **Architecture Highlights:**
  * Feedforward 11-layer architecture stacking 4 Conv blocks with interleaved BatchNorm, ReLU, and Max Pooling.
  * Spatial dimension collapsed from $256 \times 14 \times 14$ to $256 \times 1 \times 1$ via `AdaptiveAvgPool2d` to prevent classifier parameter explosion.
* **Hyperparameters & Regularization:**
  * **Optimizer:** `AdamW` (learning rate `1e-3`, weight decay `0.01`).
  * **Regularization:** Double Dropout (`0.4` and `0.2`) on the final fully connected head.
* **Best Test Accuracy:** **`76.15%`** (Epoch 13), final `74.50%`
* **Best Validation Loss:** **`0.5034`** (BCE loss)
* **Overfitting / Generalization Analysis:**
  * **Result:** Extremely stable convergence with minimal gap between train and test performance.
  * **Insights:** Showed that reducing spatial dimensions via adaptive average pooling rather than flattening giant maps prevents dense-layer weight memorization.

---

### Run 2: Rust CPU Convolutional Neural Network
* **Date:** June 20, 2026
* **Hardware Device:** CPU (Single-threaded)
* **Input Resolution:** $128 \times 128$ Grayscale
* **Architecture Highlights:**
  * Fully from-scratch neural engine in pure Rust with zero external libraries.
  * 2 Conv layers ($3 \times 3$ kernels), 2 Max Pooling layers ($2 \times 2$), flatten, ReLU hidden dense layer ($64$ nodes), and Sigmoid binary classifier.
* **Hyperparameters & Regularization:**
  * **Optimizer:** SGD with classic momentum (learning rate `0.05`).
  * **Regularization:** L2 weight decay ($\lambda = 0.001$).
* **Best Test Accuracy:** **`68.35%`** (Epoch 11)
* **Best Validation Loss:** **`0.6756`**
* **Overfitting / Generalization Analysis:**
  * **Result:** Moderate overfitting. 
  * **Insights:** Reached the performance limit of CPU-bound, zero-dependency code. Lack of RGB color data and small filter/receptive field bounds restricted high-fidelity feature extraction.

---

### Run 1: Rust CPU Multi-Layer Perceptron (MLP)
* **Date:** June 19, 2026
* **Hardware Device:** CPU (Single-threaded)
* **Input Resolution:** $64 \times 64$ Grayscale
* **Architecture Highlights:**
  * Fully from-scratch fully connected feedforward network.
  * Sigmoid activations and basic SGD backpropagation.
* **Best Test Accuracy:** **`~50.00%`** (essentially random guessing)
* **Overfitting / Generalization Analysis:**
  * **Result:** Extremely high bias (cannot learn).
  * **Insights:** Flattening raw pixels directly into a fully connected layer completely discards spatial invariance, preventing the model from resolving complex anatomical patterns.
