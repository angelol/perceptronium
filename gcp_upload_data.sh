#!/bin/bash
set -euo pipefail

# GCP Upload Dataset and Code Script
# Coordinates archive and upload of code + datasets to Google Cloud Storage (GCS)

# Locate workspace root dynamically
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load and export environment variables from .env
if [ -f "$WORKSPACE_DIR/.env" ]; then
    echo "🔑 Loading environment variables from .env..."
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        # Strip any leading/trailing whitespace
        line=$(echo "$line" | xargs)
        # Export the key/value pair
        export "$line"
    done < "$WORKSPACE_DIR/.env"
else
    echo "❌ Error: $WORKSPACE_DIR/.env file not found!"
    echo "👉 Please copy .env.example to .env and configure your GCP credentials."
    exit 1
fi

BUCKET_NAME="${GCP_BUCKET_NAME:-ai-studio-bucket-734017220015-us-west1}"
GCS_PREFIX="${GCP_GCS_PREFIX:-perceptronium}"
CATS_DOGS_DATASET_DIR="${LOCAL_CATS_DOGS_DATASET_DIR:-/Users/al/Projects/angelo/cats_dogs_dataset}"

echo "========================================="
echo "🚀 Preparing resources for GCP Training Run"
echo "========================================="

# 1. Setup temporary directory for archives
TEMP_DIR=$(mktemp -d -t perceptronium_gcp_XXXXXX)
trap 'echo "🧹 Cleaning up temporary archives..."; rm -rf "$TEMP_DIR"' EXIT

echo "📂 Temporary directory: $TEMP_DIR"

# 2. Check local paths
if [ ! -d "$WORKSPACE_DIR/data/PetImages" ]; then
    echo "❌ Error: $WORKSPACE_DIR/data/PetImages directory not found!"
    exit 1
fi

if [ ! -d "$CATS_DOGS_DATASET_DIR" ]; then
    echo "❌ Error: $CATS_DOGS_DATASET_DIR directory not found!"
    exit 1
fi

# 3. Archive primary Microsoft PetImages dataset
echo "📦 Archiving primary Microsoft PetImages dataset..."
tar -czf "$TEMP_DIR/PetImages.tar.gz" -C "$WORKSPACE_DIR/data" PetImages
echo "✓ Primary dataset archived. Size: $(du -sh "$TEMP_DIR/PetImages.tar.gz" | cut -f1)"

# 4. Archive extra high-quality cats_dogs_dataset
echo "📦 Archiving extra high-quality cats_dogs_dataset..."
tar -czf "$TEMP_DIR/cats_dogs_dataset.tar.gz" -C "$(dirname "$CATS_DOGS_DATASET_DIR")" "$(basename "$CATS_DOGS_DATASET_DIR")"
echo "✓ Extra dataset archived. Size: $(du -sh "$TEMP_DIR/cats_dogs_dataset.tar.gz" | cut -f1)"

# 5. Archive training code (excluding virtual envs, caches, checkpoints, and curves)
echo "📦 Archiving pytorch_cnn code..."
tar --exclude='.venv' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pth' \
    --exclude='*.png' \
    --exclude='.DS_Store' \
    -czf "$TEMP_DIR/pytorch_cnn.tar.gz" \
    -C "$WORKSPACE_DIR" pytorch_cnn
echo "✓ Code archived. Size: $(du -sh "$TEMP_DIR/pytorch_cnn.tar.gz" | cut -f1)"

# 6. Upload archives to GCS
echo "📤 Uploading archives to GCS bucket: gs://$BUCKET_NAME..."
gcloud storage cp "$TEMP_DIR/PetImages.tar.gz" "gs://$BUCKET_NAME/$GCS_PREFIX/data/PetImages.tar.gz"
gcloud storage cp "$TEMP_DIR/cats_dogs_dataset.tar.gz" "gs://$BUCKET_NAME/$GCS_PREFIX/data/cats_dogs_dataset.tar.gz"
gcloud storage cp "$TEMP_DIR/pytorch_cnn.tar.gz" "gs://$BUCKET_NAME/$GCS_PREFIX/code/pytorch_cnn.tar.gz"

echo "========================================="
echo "🎉 All resources uploaded successfully to:"
echo "   gs://$BUCKET_NAME/$GCS_PREFIX/"
echo "========================================="
