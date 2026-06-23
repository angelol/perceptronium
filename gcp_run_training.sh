#!/bin/bash
set -euo pipefail

# GCP Provisioning and Training Script
# Coordinates:
# 1. Optional dataset/Code archiving and upload to GCS.
# 2. Dynamic discovery of latest active PyTorch Ubuntu image family.
# 3. Spot VM creation with 1x NVIDIA L4 GPU with robust multi-zone fallback.
# 4. Startup-script attachment and background execution.

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
    echo "👉 Please copy .env.template to .env and configure your GCP credentials."
    exit 1
fi

PROJECT_ID="${GCP_PROJECT_ID:-ems-spons}"
BUCKET_NAME="${GCP_BUCKET_NAME:-ai-studio-bucket-734017220015-us-west1}"
GCS_PREFIX="${GCP_GCS_PREFIX:-perceptronium}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-perceptronium-training-l4}"
MACHINE_TYPE="${GCP_MACHINE_TYPE:-g2-standard-4}"
IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"

# Candidate zones with L4 GPU availability under our quotas
ZONES=("us-west1-a" "us-west1-b" "us-west1-c" "us-central1-a" "us-central1-b" "us-central1-c" "us-east1-b" "us-east1-c" "us-east1-d" "us-east4-a" "us-east4-c" "us-west4-a" "us-west4-c")

echo "========================================="
echo "⚙️  Starting GCP Spot VM Migration Pipeline"
echo "========================================="

# Parse arguments
SKIP_UPLOAD=false
if [ "${1:-}" = "--skip-upload" ]; then
    SKIP_UPLOAD=true
fi

# Check active project
ACTIVE_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ "$ACTIVE_PROJECT" != "$PROJECT_ID" ]; then
    echo "⚠️ Warning: Active gcloud project is '$ACTIVE_PROJECT', expected '$PROJECT_ID'."
    echo "🔄 Setting active project to '$PROJECT_ID'..."
    gcloud config set project "$PROJECT_ID"
fi

# Step 1. Archive and upload dataset/code
if [ "$SKIP_UPLOAD" = true ]; then
    echo "⏭️  Skipping dataset and code packager/uploader (re-using existing GCS archives)..."
else
    if [ -f "./gcp_upload_data.sh" ]; then
        echo "📤 Running dataset and code packager/uploader..."
        ./gcp_upload_data.sh
    else
        echo "❌ Error: ./gcp_upload_data.sh not found!"
        exit 1
    fi
fi

# Step 2. Discover image family dynamically
echo "🔍 Querying latest active PyTorch Ubuntu GPU image family on GCP..."
IMAGE_FAMILY=$(gcloud compute images list --project="$IMAGE_PROJECT" --no-standard-images --filter="family:pytorch-*" --format="value(family)" | sort -V | tail -n 1)

if [ -z "$IMAGE_FAMILY" ]; then
    # Fallback to the known stable family if dynamic query failed
    IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2404-nvidia-580"
    echo "⚠️ Warning: Image family query returned empty. Using fallback: $IMAGE_FAMILY"
else
    echo "✓ Found latest active PyTorch image family: $IMAGE_FAMILY"
fi

# Step 3 & 4. Create the Spot VM instance with dynamic multi-zone fallback
SUCCESS=false
CHOSEN_ZONE=""

for ZONE in "${ZONES[@]}"; do
    echo "========================================="
    echo "🔄 Attempting to provision in zone: $ZONE"
    echo "========================================="

    # Check if instance already exists in this zone
    if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &>/dev/null; then
        echo "⚠️ Warning: Instance '$INSTANCE_NAME' already exists in zone '$ZONE'."
        echo "🔄 Deleting existing instance to ensure a fresh, clean training run..."
        gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
        echo "✓ Existing instance deleted successfully."
    fi

    # Try to create the Spot VM instance
    echo "🚀 Provisioning Spot VM instance '$INSTANCE_NAME' with 1x NVIDIA L4 GPU in '$ZONE'..."
    if gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --provisioning-model=SPOT \
        --preemptible \
        --scopes=cloud-platform \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-ssd \
        --metadata-from-file=startup-script=gcp_startup_script.sh \
        --metadata="bucket-name=$BUCKET_NAME,gcs-prefix=$GCS_PREFIX"; then
        
        CHOSEN_ZONE="$ZONE"
        SUCCESS=true
        break
    else
        echo "⚠️ Warning: Failed to provision Spot VM in zone '$ZONE' (possibly resource stockout)."
        echo "🔄 Proceeding to try next candidate zone..."
    fi
done

if [ "$SUCCESS" = false ]; then
    echo "❌ Error: Failed to provision Spot VM across all candidate zones!"
    echo "Please check your quotas or try again later."
    exit 1
fi

echo "========================================="
echo "🎉 Spot VM Instance Created Successfully!"
echo "========================================="
echo "Instance Name: $INSTANCE_NAME"
echo "Active Zone:   $CHOSEN_ZONE"
echo "Training has been initiated in the background via the startup-script."
echo ""
echo "📈 How to monitor progress:"
echo "1. View live VM startup logs using gcloud CLI:"
echo "   gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$CHOSEN_ZONE --project=$PROJECT_ID"
echo ""
echo "2. Once training is complete, the VM will automatically poweroff, and weights/logs"
echo "   will be uploaded to:"
echo "   gs://$BUCKET_NAME/$GCS_PREFIX/results/"
echo ""
echo "3. You can list the GCS results directory to check for final checkpoints:"
echo "   gcloud storage ls gs://$BUCKET_NAME/$GCS_PREFIX/results/"
echo "========================================="

