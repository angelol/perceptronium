#!/usr/bin/env bash
set -euxo pipefail

# Perceptronium GCP GPU Startup Training Script
# Runs inside the Deep Learning VM instance on GCE.

# 1. Initialize Paths
# Using high-performance PD-SSD boot disk (/opt/perceptronium) with 100GB space to prevent tmpfs space limits.
WORKDIR="/opt/perceptronium"
mkdir -p "$WORKDIR"
mkdir -p "$WORKDIR/data"

# Redirect stdout and stderr to a local log file for live tail and debugging
exec > >(tee -i /var/log/perceptronium_startup.log) 2>&1

echo "=== PERCEPTRONIUM GCP STARTUP RUN STARTING ==="
date

# GCS Bucket Config
BUCKET_NAME="ai-studio-bucket-734017220015-us-west1"
GCS_PREFIX="perceptronium"

# 2. Check if a local training checkpoint exists to prevent overwriting progress
if [ -d "$WORKDIR/pytorch_cnn" ] && [ -f "$WORKDIR/pytorch_cnn/model_checkpoint_latest.pth" ]; then
    echo "🔄 Detected existing training workspace and checkpoint from interrupted run!"
    echo "⚠️ Bypassing download & extraction of code to protect local training progress."
else
    echo "📥 Downloading archives from GCS bucket: gs://$BUCKET_NAME..."
    gcloud storage cp "gs://$BUCKET_NAME/$GCS_PREFIX/data/PetImages.tar.gz" "$WORKDIR/data/PetImages.tar.gz"
    gcloud storage cp "gs://$BUCKET_NAME/$GCS_PREFIX/data/cats_dogs_dataset.tar.gz" "$WORKDIR/data/cats_dogs_dataset.tar.gz"
    gcloud storage cp "gs://$BUCKET_NAME/$GCS_PREFIX/code/pytorch_cnn.tar.gz" "$WORKDIR/pytorch_cnn.tar.gz"

    echo "📦 Extracting archives..."
    tar -xzf "$WORKDIR/data/PetImages.tar.gz" -C "$WORKDIR/data"
    tar -xzf "$WORKDIR/data/cats_dogs_dataset.tar.gz" -C "$WORKDIR"
    tar -xzf "$WORKDIR/pytorch_cnn.tar.gz" -C "$WORKDIR"
fi

# 3. Enter the PyTorch training workspace
cd "$WORKDIR/pytorch_cnn"

# 4. Detect and configure the Deep Learning VM Python environment
PYTHON_BIN="python3"
PIP_BIN="pip3"

# Candidate paths for pre-installed Python environments (with GPU support)
CANDIDATES=(
    "/opt/conda/envs/pytorch/bin/python"
    "/opt/conda/envs/pytorch/bin/python3"
    "/opt/conda/envs/c2d-dl-platform-py310/bin/python"
    "/opt/conda/envs/c2d-dl-platform-py310/bin/python3"
    "/opt/conda/bin/python"
    "/opt/conda/bin/python3"
)

for CANDIDATE in "${CANDIDATES[@]}"; do
    if [ -x "$CANDIDATE" ]; then
        echo "🔍 Testing python candidate: $CANDIDATE"
        # Test if PyTorch works in this candidate
        if "$CANDIDATE" -c "import torch" &>/dev/null; then
            PYTHON_BIN="$CANDIDATE"
            # Corresponding pip should be in the same bin directory
            CANDIDATE_PIP="${CANDIDATE%/*}/pip"
            if [ -x "$CANDIDATE_PIP" ]; then
                PIP_BIN="$CANDIDATE_PIP"
            else
                PIP_BIN="$PYTHON_BIN -m pip"
            fi
            echo "✓ Successfully selected high-performance PyTorch Python environment: $PYTHON_BIN"
            break
        fi
    fi
done

if [ "$PYTHON_BIN" = "python3" ]; then
    echo "⚠️ Warning: No pre-configured conda PyTorch environment found. Using system Python..."
    # On Ubuntu 24.04, pip3 needs --break-system-packages (can be enabled via env var to prevent syntax errors with older pips)
    export PIP_BREAK_SYSTEM_PACKAGES=1
    PIP_BIN="pip3"
fi

echo "🐍 Using Python binary: $PYTHON_BIN"
$PYTHON_BIN --version

# 5. Install optional package dependencies if needed
if [ -f "requirements.txt" ]; then
    echo "pip: Installing requirements from requirements.txt..."
    $PIP_BIN install -r requirements.txt || echo "⚠️ Warning: Some pip installations returned errors, proceeding anyway..."
fi


# 6. Execute Training Run 11 with hardware-optimized throughput parameters
# Running with Lookahead AdamW, progressive resizing, sequential lr, native BF16, global TF32, 128 batch size, 
# and CUDA Graphs model compilation (reduce-overhead mode)
echo "🏋️ Launching hardware-optimized training Run 11 on NVIDIA L4 GPU..."
TRAINING_FAILED=0
$PYTHON_BIN -u train.py \
    --epochs 200 \
    --batch-size 128 \
    --optimizer lookahead \
    --scheduler sequential \
    --progressive \
    --save-snapshots \
    --amp \
    --bf16 \
    --compile \
    --compile-mode reduce-overhead \
    --data-dir "$WORKDIR/data/PetImages" \
    --extra-dir "$WORKDIR/cats_dogs_dataset" \
    --weights-path "$WORKDIR/pytorch_cnn/model.pth" \
    --resume \
    > "$WORKDIR/pytorch_cnn/training.log" 2>&1 || TRAINING_FAILED=1

if [ "$TRAINING_FAILED" -eq 1 ]; then
    echo "❌ Error: Training run failed! Copying partial logs to GCS..."
else
    echo "✅ Training completed successfully!"
    
    # Run post-training calibration using our newly integrated 12-View TTA-averaged temperature scaler
    echo "🔬 Running post-training calibration (Expected Calibration Error optimization)..."
    $PYTHON_BIN -u calibrate.py \
        --weights-path "$WORKDIR/pytorch_cnn/model.pth" \
        --data-dir "$WORKDIR/data/PetImages" \
        --image-size 224 \
        --batch-size 128 \
        --attention-type se \
        > "$WORKDIR/pytorch_cnn/calibration.log" 2>&1 || echo "⚠️ Warning: Calibration optimization returned an error."
fi

# 7. Upload checkpoints, snapshots, calibration results, and logs to GCS
echo "📤 Uploading weights, checkpoints, and logs to GCS..."
if [ -f "$WORKDIR/pytorch_cnn/model.pth" ]; then
    gcloud storage cp "$WORKDIR/pytorch_cnn/model.pth" "gs://$BUCKET_NAME/$GCS_PREFIX/results/model_run11.pth" || true
fi
if [ -f "$WORKDIR/pytorch_cnn/learning_curves.png" ]; then
    gcloud storage cp "$WORKDIR/pytorch_cnn/learning_curves.png" "gs://$BUCKET_NAME/$GCS_PREFIX/results/learning_curves_run11.png" || true
fi
if [ -f "$WORKDIR/pytorch_cnn/calibration_reliability.png" ]; then
    gcloud storage cp "$WORKDIR/pytorch_cnn/calibration_reliability.png" "gs://$BUCKET_NAME/$GCS_PREFIX/results/calibration_reliability_run11.png" || true
fi
if [ -f "$WORKDIR/pytorch_cnn/temperature.txt" ]; then
    gcloud storage cp "$WORKDIR/pytorch_cnn/temperature.txt" "gs://$BUCKET_NAME/$GCS_PREFIX/results/temperature_run11.txt" || true
fi
if [ -f "$WORKDIR/pytorch_cnn/training.log" ]; then
    gcloud storage cp "$WORKDIR/pytorch_cnn/training.log" "gs://$BUCKET_NAME/$GCS_PREFIX/results/training_run11.log" || true
fi
if [ -f "$WORKDIR/pytorch_cnn/calibration.log" ]; then
    gcloud storage cp "$WORKDIR/pytorch_cnn/calibration.log" "gs://$BUCKET_NAME/$GCS_PREFIX/results/calibration_run11.log" || true
fi
if [ -f "/var/log/perceptronium_startup.log" ]; then
    gcloud storage cp "/var/log/perceptronium_startup.log" "gs://$BUCKET_NAME/$GCS_PREFIX/results/startup_run11.log" || true
fi

# Upload all epoch snapshots
echo "📤 Uploading epoch snapshots to GCS..."
gcloud storage cp "$WORKDIR/pytorch_cnn/model_epoch"*.pth "gs://$BUCKET_NAME/$GCS_PREFIX/results/" || true

echo "=== PERCEPTRONIUM CLOUD TRAINING FINISHED ==="
date

# 8. Self-termination: Immediately poweroff the VM to terminate running costs
echo "🛑 Self-terminating instance now to prevent idle billing..."
sudo poweroff
