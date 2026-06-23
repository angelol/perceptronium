# 📈 Perceptronium Accuracy Log

This log tracks our model architectures, training runs, hyperparameter choices, and accuracy progress over time as we refine our Cat vs. Dog classifiers.

---

## 🏆 Current Accuracy Leaderboard

| Rank | Model Identifier | Architecture Style | Total Params | Best Test Acc | Final Test Acc | Overfitting Status | Key Innovations |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **1** | **CBAM-EfficientNet v4 (GCP Run 11)** | CBAM-EfficientNet v4 (Ultra Hybrid) | ~24.31M | **98.96%** | **98.63%** | Minimal | 200 epochs, Lookahead-AdamW, 2D Transformer stage, Progressive resizing, TTA, Post-hoc calibration |
| **2** | **CBAM-EfficientNet v2 (SWA)** | Depthwise Attention Bottleneck v2 | ~8.69M | **93.30%** | **92.35%** | Minimal | Multi-block MBConv stage, Lookahead optimizer, Cosine Warm Restarts, TrivialAugment, SWA & Temperature Scaling |
| **3** | **Option A+ (Residual CNN)** | Residual feedforward CNN | ~4.91M | **87.60%** | **87.60%** | Minimal | Skip connections, RandomResizedCrop, Cosine Annealing |
| **4** | **CBAM-EfficientNet v3 (SWA)** | CBAM-EfficientNet v3 (CNN-Transformer Hybrid) | ~13.88M | **87.30%** | **86.40%** | Negative (Underfitting) | Lookahead wrapper, 2D MHSA, Fused-MBConv, SWA, Progressive resizing |
| **5** | **Option C (CBAM-EfficientNet)** | Depthwise Attention Bottleneck | ~1.28M | **84.90%** | **84.90%** | Negative (None) | CBAM Attention, MBConv stages, OneCycleLR, Mixup/CutMix |
| **6** | **CBAM-EfficientNet v3 (GCP Run 10)** | CBAM-EfficientNet v3 (CNN-Transformer Hybrid) | ~13.88M | **76.22%** | **74.04%** | Moderate | GCP CUDA migration, AMP, compile, 60 epochs of Lookahead-AdamW, SWA |
| **7** | **Option A (Custom CNN)** | Standard feedforward CNN | ~441K | **76.15%** | **74.50%** | Minimal | AdaptiveAvgPool, BatchNorm, Dropout |
| **8** | **Rust CNN Baseline** | From-scratch CPU CNN | ~924K | **68.35%** | **66.90%** | Moderate | 2 Conv, 2 Pool, 1 Dense, horizontal flips in Rust |
| **9** | **Rust MLP Baseline** | From-scratch CPU MLP | ~131K | **~50.00%** | **~50.00%** | High Bias | Fully connected nodes, no spatial representation |
| **10** | **CBAM-EfficientNet v3 (Local Run 9)** | CBAM-EfficientNet v3 (CNN-Transformer Hybrid) | ~13.88M | **55.27%** | - | Failed/Degraded | Stuck at ~55% on local MPS due to silent gradient underflow/clipping |
| **11** | **CBAM-EfficientNet v3 (ASAM)** | CBAM-EfficientNet v3 (CNN-Transformer Hybrid) | ~13.88M | **50.00%** | **50.00%** | Failed Run | ASAM (rho=0.5) destroyed weight initialization |

---

## 📝 Detailed Training Run Records

### Run 11: CBAM-EfficientNet v4 (Ultra Hybrid - GCP Run 11)
* **Date:** June 23, 2026
* **Hardware Device:** GCP Spot VM (1x NVIDIA L4 GPU with CUDA)
* **Input Resolution:** Progressive Resizing (Epochs 1-60: 160x160 RGB, Epochs 61-130: 224x224 RGB, Epochs 131-200: 288x288 RGB)
* **Architecture Highlights:**
  - **CBAM-EfficientNet v4 (Ultra Hybrid)** architecture (~24.31M parameters) trained completely from scratch.
  - Multi-block stage repetitions with SE attention.
  - Features a **2D Transformer Block** with Pre-LayerNorm, 8-head Self-Attention, and Feed-Forward Networks (FFN) at the deepest ($7 \times 7$) resolution before pooling.
  - Pre-classifier Head Projection: Projects 480 to 1536 channels (+20% feature space capacity) before Global Average Pooling.
  - Single-linear classifier with Dropout of `0.3`.
* **Hyperparameters & Regularization:**
  - **Optimizer:** Lookahead Optimizer Wrapper over `AdamW` ($k=5, \alpha=0.5$), peak learning rate `8e-4`, weight decay `0.05`.
  - **Scheduler:** `SequentialLR` featuring a 10-epoch Linear Warmup followed by Cosine Annealing over 200 epochs.
  - **Augmentation Curriculum:** `TrivialAugmentWide` + linearly decaying DropPath (Stochastic Depth).
  - **Mixup & CutMix Cooldown:** Enabled Mixup/CutMix with active phase during training, then completely disabled Mixup/CutMix in the final 20 epochs (Cooldown Phase: Epochs 181-200) to let the model sharpen its decision boundaries.
  - **Post-Hoc Calibration:** Temperature scaling using L-BFGS optimization on the validation split, finding optimal $T = 1.409243$.
* **Epoch-by-Epoch Convergence Highlights:**
  ```
  | Epoch 10  | Train Loss: 0.5220 | Test Loss: 0.6927 | Test Acc: 50.00% | (Warmup End, Mixup On, weight-mutation evaluation bug active)
  | Epoch 60  | Train Loss: 0.3462 | Test Loss: 0.2194 | Test Acc: 97.66% | (Resize to 224x224, compilation conflict resolved)
  | Epoch 121 | Train Loss: 0.2884 | Test Loss: 0.1510 | Test Acc: 98.96% | (Peak Validation Acc #1)
  | Epoch 130 | Train Loss: 0.2890 | Test Loss: 0.1505 | Test Acc: 98.81% | (Resize to 288x288)
  | Epoch 134 | Train Loss: 0.2812 | Test Loss: 0.1637 | Test Acc: 98.96% | (Peak Validation Acc #2)
  | Epoch 180 | Train Loss: 0.2772 | Test Loss: 0.1585 | Test Acc: 98.55% | (Cooldown End, Mixup Off)
  | Epoch 188 | Train Loss: 0.1445 | Test Loss: 0.1343 | Test Acc: 98.78% | (Min Validation Loss)
  | Epoch 200 | Train Loss: 0.1458 | Test Loss: 0.1350 | Test Acc: 98.63% | (Final Validation Acc)
  ```
* **Best Test Accuracy:** **`98.96%`** (Epochs 121 and 134, with 12-View TTA)
* **Final Test Accuracy:** **`98.63%`** (Epoch 200, with 12-View TTA, Test Loss: `0.1350`)
* **Overfitting / Generalization Analysis:**
  - **Result:** Minimal/Zero Overfitting.
  - **Insights:**
    - **CUDA backend success:** Migrating the training pipeline to an NVIDIA L4 GPU completely resolved Apple Silicon/MPS graph compilation gradient conflicts, reducing per-epoch time dramatically.
    - **Weight-Mutation Evaluation Anomaly:** In early epochs (before Epoch 59), a graph compilation conflict caused validation accuracy to be reported as exactly `50.00%`. This was solved by extracting the uncompiled original model from the compiled EMA wrapper during validation, showing immediate jumps to the actual high accuracy.
    - **Curriculum and Cooldown benefits:** Training at $160 \times 160$ px initially enabled massive computational speedups. Resizing progressively up to $288 \times 288$ px captured high-frequency fur and whisker textures. Disabling Mixup/CutMix in the final 20 epochs allowed the model to rapidly sharpen decision boundaries, jumping from ~93% clean validation accuracy to over **98.7%** stable accuracy, and validation loss bottomed out at **0.1343** (Epoch 188).

### Run 6: CBAM-EfficientNet v2 (SWA)
* **Date:** June 20, 2026
* **Hardware Device:** Metal GPU (MPS)
* **Input Resolution:** $224 \times 224$ RGB (3 channels)
* **Architecture Highlights:**
  - True Multi-Block Stage Repetitions (Stage 2: 2 reps, Stage 3: 3 reps, Stage 4: 3 reps, Stage 5: 2 reps) giving **7 active identity skip connections** flowing gradients cleanly.
  - Large receptive fields ($5 \times 5$ depthwise separable convolutions) in deeper blocks (Stages 4 and 5) to capture complex local geometry.
  - Pre-classifier $1 \times 1$ projection to 1280 channels before Global Avg Pooling, reducing the classification head to a single linear layer with Dropout of `0.3`.
  - Stochastically regularized depth via linearly decaying DropPath (scaling $0.0 \to 0.2$).
  - Toggled dual attention: Squeeze-and-Excitation (SE) block with reduction ratio of 4.
* **Hyperparameters & Regularization:**
  - **Optimizer:** Lookahead Optimizer Wrapper over `AdamW` (learning rate `8e-4`, weight decay `0.05`).
  - **Scheduler:** `CosineAnnealingWarmRestarts` ($T_0=15$, $T_{mult}=2$), running restarts at Epoch 15 to escape suboptimal basins.
  - **Augmentation curriculum:** `TrivialAugmentWide` + cutout-like `RandomErasing(p=0.2)`. Progressive resizing context / Wider crop scale `(0.2, 1.0)`.
  - **Mixup & CutMix Cooldown:** Enabled Mixup/CutMix with warmup (Epochs 1-2 disabled) and active phase (Epochs 3-40 enabled). Cooldown phase (Epochs 41-45 completely disabled Mixup/CutMix) to let the model sharpen its decision boundaries.
  - **Decoupled Label Smoothing:** $\epsilon = 0.05$ only on clean batches; mixed batches used raw mixed-target ratio.
  - **Calibration Scaling:** Post-hoc calibration using L-BFGS temperature scaling on the validation split, finding optimal *T* = 0.8088 and lowering ECE from 4.38% to 2.88%.
* **Best Test Accuracy:** **`93.30%`** (Epoch 38, Peak with TTA)
* **SWA Test Accuracy:** **`92.35%`** (with TTA, after SWA BN stabilization, SWA Test Loss: `0.2599`)
* **Overfitting / Generalization Analysis:**
  - **Result:** Minimal/Zero Overfitting.
  - **Insights:** The multi-block skip connections completely resolved the training bottleneck of the prior attention bottleneck model. Lookahead and warm restarts combined with SWA allowed the weights to settle in flat minima, which generalized much better to the validation set.

### Run 5: CBAM-EfficientNet (Option C)
* **Date:** June 20, 2026
* **Hardware Device:** Metal GPU (MPS)
* **Input Resolution:** 224x224 RGB (3 channels)
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
* **Input Resolution:** 224x224 RGB (3 channels)
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
* **Input Resolution:** 224x224 RGB (3 channels)
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
* **Input Resolution:** 128x128 Grayscale
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
* **Input Resolution:** 64x64 Grayscale
* **Architecture Highlights:**
  * Fully from-scratch fully connected feedforward network.
  * Sigmoid activations and basic SGD backpropagation.
* **Best Test Accuracy:** **`~50.00%`** (essentially random guessing)
* **Overfitting / Generalization Analysis:**
  * **Result:** Extremely high bias (cannot learn).
  * **Insights:** Flattening raw pixels directly into a fully connected layer completely discards spatial invariance, preventing the model from resolving complex anatomical patterns.

---

### Run 7: CBAM-EfficientNet v3 (ASAM - Failed Run)
* **Date:** June 21, 2026
* **Hardware Device:** Metal GPU (MPS)
* **Input Resolution:** 224x224 RGB
* **Architecture Highlights:**
  * Custom CBAM-EfficientNet v3 (~13.88M parameters) combining fused early stages and late standard stage.
  * Incorporates 2D Multi-Head Self-Attention (MHSA) at the deep feature extraction layer.
* **Hyperparameters & Regularization:**
  * **Optimizer:** ASAM ($\rho = 0.5$, adaptive) over base AdamW.
  * **Scheduler:** Cosine Warm Restarts.
  * **SWA & Snapshots:** SWA enabled (start epoch 35), snapshot checkpoints of final epochs.
* **Best Test Accuracy:** **`50.00%`** (Stuck at coin-toss)
* **Overfitting / Generalization Analysis:**
  * **Result:** Complete failure to learn.
  * **Insights:** With weights trained from scratch, the scale-invariant perturbation computed by ASAM under a high radius of $\rho = 0.5$ causes massive weight changes (up to 50-100% of their actual magnitude) in early steps. This completely scrambled the model's initialization, ruined gradient backpropagation, and locked the model in a random-guess state.

---

### Run 8: CBAM-EfficientNet v3 (Lookahead-AdamW)
* **Date:** June 21, 2026
* **Hardware Device:** Metal GPU (MPS)
* **Input Resolution:** 224x224 RGB (Progressive Resizing curriculum: 15 epochs at 128x128, 15 epochs at 192x192, and 15 epochs at 224x224)
* **Architecture Highlights:**
  * Custom CBAM-EfficientNet v3 (~13.88M parameters) combining fused early stages, late standard stage, and a 2D Multi-Head Self-Attention layer.
* **Hyperparameters & Regularization:**
  * **Optimizer:** Lookahead wrapper over AdamW ($k=5, \alpha=0.5$), peak learning rate `5e-4`, weight decay `0.05`.
  * **Scheduler:** Cosine Warm Restarts.
  * **Data Regularization:** Mixup, Cutmix, TrivialAugmentWide, RandomErasing, linearly-decaying DropPath.
  * **SWA & Snapshots:** SWA enabled (start epoch 35), snapshot checkpoints of final epochs.
* **Best Test Accuracy:** **`87.30%`** (Epoch 42, 6-View TTA)
* **SWA Test Accuracy:** **`86.40%`** (with 6-View TTA, Test Loss: `0.3662`)
* **Overfitting / Generalization Analysis:**
  - **Result:** Severe underfitting (approximate Train Acc `84.28%` vs Test Acc `85.40%` at Epoch 45).
  - **Insights:** Lookahead resolved the learning failure entirely, leading to highly stable and smooth loss curves. However, 45 epochs was not nearly enough for a 13.8M parameters model trained entirely from scratch under such heavy regularization and progressive resizing (which restricted full-resolution training to only the final 15 epochs). The model did not overfit at all and has substantial untapped capacity. For future runs, we should extend epochs to 100+, increase the peak learning rate slightly, or train at full resolution to extract this capacity.

---

### Run 9: CBAM-EfficientNet v3 (Local MPS Run)
* **Date:** June 21, 2026
* **Hardware Device:** Metal GPU (MPS)
* **Input Resolution:** 224x224 RGB
* **Architecture Highlights:**
  - Upgraded custom CBAM-EfficientNet v3 (~13.88M parameters) combining fused early stages, late standard stage, and a 2D Multi-Head Self-Attention layer.
* **Hyperparameters & Regularization:**
  - **Optimizer:** Lookahead over AdamW, peak learning rate `8e-4`, weight decay `0.05`.
  - **Scheduler:** Cosine Warm Restarts.
  - **Regularization:** SWA enabled (start epoch 35), Mixup, Cutmix, TrivialAugmentWide.
* **Best Test Accuracy:** **`55.27%`** (Epoch 36, slow/degraded convergence)
* **Overfitting / Generalization Analysis:**
  - **Result:** Complete learning degradation (stuck near coin-toss).
  - **Insights:** The training run encountered silent learning degradation on local Apple Silicon GPUs (MPS backend). Lookahead combined with AMP and gradient clipping caused silent gradient underflow/clipping errors in the MPS graph compiler, preventing proper weights update and stalling learning around ~53-55% accuracy. This prompted a migration to cloud-based CUDA hardware.

---

### Run 10: CBAM-EfficientNet v3 (GCP Spot VM Run)
* **Date:** June 21, 2026
* **Hardware Device:** GCP Spot VM (1x NVIDIA L4 GPU with CUDA)
* **Input Resolution:** 224x224 RGB (Full resolution training)
* **Architecture Highlights:**
  - Upgraded custom CBAM-EfficientNet v3 (~13.88M parameters) combining fused early stages, late standard stage, and a 2D Multi-Head Self-Attention layer.
* **Hyperparameters & Regularization:**
  - **Optimizer:** Lookahead over AdamW, peak learning rate `8e-4`, weight decay `0.05`.
  - **Scheduler:** Cosine Warm Restarts.
  - **Regularization:** SWA enabled (start epoch 35), Mixup, Cutmix, TrivialAugmentWide.
  - **Acceleration:** PyTorch Automatic Mixed Precision (AMP) and Model Compilation (`torch.compile`).
* **Best Test Accuracy:** **`76.22%`** (Epoch 59, with 6-View Multi-Scale TTA)
* **SWA Test Accuracy:** **`74.04%`** (with 6-View Multi-Scale TTA, SWA Test Loss: `0.5565`)
* **Overfitting / Generalization Analysis:**
  - **Result:** Successful convergence with moderate underfitting / slower convergence.
  - **Insights:** Migrating the training to GCP NVIDIA L4 GPU under the CUDA backend completely resolved the backend compilation gradient bug. The model trained successfully and hit **76.22%** validation accuracy, taking exactly **110 seconds** per epoch (compared to ~12 minutes on Mac MPS). However, training a 13.88M parameters model from scratch is a highly challenging optimization problem, and the model was still actively converging at 60 epochs. SWA stabilized the weights at 74.04% accuracy. To push this architecture to its full potential, a longer training budget (150+ epochs) or higher initial learning rate is recommended.
