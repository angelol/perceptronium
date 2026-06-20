import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import CBAM_EfficientNet as CustomCNN

def generate_ascii_preview(image_path, width=64):
    """
    Renders an image as terminal-based ASCII art, maintaining correct
    aspect ratio with font-height correction.
    """
    try:
        img = Image.open(image_path).convert("L")
        aspect_ratio = img.height / img.width
        height = int(width * aspect_ratio * 0.45)
        
        # Resize to target terminal box
        img_resized = img.resize((width, height))
        
        # Grayscale character ramp from dark to bright
        ascii_chars = [" ", ".", ":", "-", "=", "+", "*", "%", "#", "@"]
        num_chars = len(ascii_chars)
        
        ascii_lines = []
        for y in range(height):
            line = "".join(ascii_chars[int((img_resized.getpixel((x, y)) / 255.0) * (num_chars - 1))] for x in range(width))
            ascii_lines.append(line)
        return "\n".join(ascii_lines)
    except Exception as e:
        return f"  [Error rendering ASCII preview: {e}]"


def load_temperature(weights_dir):
    """
    Loads optimal temperature T from temperature.txt if it exists.
    """
    temp_path = os.path.join(weights_dir, "temperature.txt")
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r") as f:
                T = float(f.read().strip())
                print(f"🌡️ Temperature scaling calibration loaded: T = {T:.4f}")
                return T
        except Exception as e:
            print(f"⚠️ Error reading temperature file: {e}")
    return 1.0


def predict_ensemble(image_path, models, device, T=1.0, image_size=224, use_tta=True):
    """
    Preprocesses an input image, runs CNN inference across an ensemble of models,
    averages predicted probabilities, and applies temperature scaling.
    """
    # 1. Show ASCII art preview
    print("\n📸 ASCII ART PREVIEW:")
    print("=" * 66)
    print(generate_ascii_preview(image_path))
    print("=" * 66)
    
    # 2. Image Preprocessing
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to load image at {image_path}: {e}")
        return
        
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device) # Shape: (1, 3, 224, 224)
    
    # 3. Model Inference (Ensemble & Calibrated)
    for model in models:
        model.eval()
        
    all_probs_uncalibrated = []
    all_probs_calibrated = []
    
    with torch.no_grad():
        # Standard input and flipped input (TTA)
        inputs = [input_tensor]
        if use_tta:
            inputs.append(torch.flip(input_tensor, dims=[3]))
            
        for inp in inputs:
            for model in models:
                logits = model(inp)
                
                # Uncalibrated probability
                prob_uncal = torch.sigmoid(logits).item()
                all_probs_uncalibrated.append(prob_uncal)
                
                # Calibrated probability using temperature T
                logits_calibrated = logits / T
                prob_cal = torch.sigmoid(logits_calibrated).item()
                all_probs_calibrated.append(prob_cal)
                
    # Average across all ensemble members and TTA views
    avg_uncalibrated_prob = sum(all_probs_uncalibrated) / len(all_probs_uncalibrated)
    avg_calibrated_prob = sum(all_probs_calibrated) / len(all_probs_calibrated)
    
    # Class determination using calibrated probability (threshold at 0.5)
    # The decision class remains unchanged, but the confidence reflects true calibration.
    if avg_calibrated_prob >= 0.5:
        predicted_class = "DOG"
        confidence_calibrated = avg_calibrated_prob * 100.0
        confidence_uncalibrated = avg_uncalibrated_prob * 100.0
    else:
        predicted_class = "CAT"
        confidence_calibrated = (1.0 - avg_calibrated_prob) * 100.0
        confidence_uncalibrated = (1.0 - avg_uncalibrated_prob) * 100.0
        
    print("\n🔬 ENSEMBLE & CALIBRATED CNN INFERENCE RESULTS:")
    print("=" * 66)
    print(f"  • Number of Ensemble Models: {len(models)}")
    print(f"  • Test-Time Augmentation:   {'ENABLED' if use_tta else 'DISABLED'}")
    print(f"  • Calibration Temperature T: {T:.4f}")
    print(f"  • Dog Probability (Uncal):   {avg_uncalibrated_prob * 100.0:.2f}%")
    print(f"  • Dog Probability (Calib):   {avg_calibrated_prob * 100.0:.2f}%")
    print(f"  • Predicted Class:           \033[1;36m{predicted_class}\033[0m")
    
    if T != 1.0:
        print(f"  • Uncalibrated Confidence:   {confidence_uncalibrated:.2f}%")
        print(f"  • Calibrated Confidence:     \033[1;32m{confidence_calibrated:.2f}%\033[0m (Highly stable & realistic)")
    else:
        print(f"  • Model Confidence:          \033[1;32m{confidence_uncalibrated:.2f}%\033[0m (Uncalibrated)")
    print("=" * 66 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble & Calibrated Inference Playground")
    parser.add_argument("--weights", type=str, nargs="+", default=["pytorch_cnn/model.pth"],
                        help="Paths to model weight files. Pass multiple files to build an ensemble!")
    parser.add_argument("--image", type=str, help="Optional path to a single Cat/Dog image to evaluate")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation (TTA)")
    parser.add_argument("--attention-type", type=str, default="se", choices=["se", "cbam"], help="Attention type in blocks")
    args = parser.parse_args()
    
    # Device Selection
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading custom CNN models on: {device.type.upper()}...")
    
    # Identify weights files
    resolved_paths = []
    for w in args.weights:
        if os.path.exists(w):
            resolved_paths.append(w)
        elif os.path.exists(os.path.join("pytorch_cnn", os.path.basename(w))):
            resolved_paths.append(os.path.join("pytorch_cnn", os.path.basename(w)))
            
    # Auto-detection of snapshots or SWA weights if none specified or standard is missing
    if len(resolved_paths) == 1 and not os.path.exists(resolved_paths[0]):
        # Check if SWA or snapshot files exist in directory
        dir_files = os.listdir("pytorch_cnn") if os.path.exists("pytorch_cnn") else []
        snapshots = [os.path.join("pytorch_cnn", f) for f in dir_files if f.startswith("model_epoch") and f.endswith(".pth")]
        swa_weight = os.path.join("pytorch_cnn", "model_swa.pth")
        
        if os.path.exists(swa_weight):
            resolved_paths = [swa_weight]
            print(f"🔄 Auto-detected and using SWA model checkpoint: {swa_weight}")
        elif len(snapshots) > 0:
            resolved_paths = sorted(snapshots)
            print(f"🔄 Auto-detected and ensembling {len(resolved_paths)} final-epoch snapshots: {resolved_paths}")
        else:
            print(f"❌ Error: Weights path '{args.weights[0]}' does not exist and no SWA/snapshots auto-detected.")
            sys.exit(1)
            
    print(f"📂 Active Model Checkpoints: {', '.join([os.path.basename(p) for p in resolved_paths])}")
    
    # Load all models in the ensemble
    models = []
    for path in resolved_paths:
        m = CustomCNN(attention_type=args.attention_type).to(device)
        try:
            m.load_state_dict(torch.load(path, map_location=device))
            models.append(m)
        except Exception as e:
            print(f"❌ Failed to load model weights from '{path}': {e}")
            sys.exit(1)
            
    print(f"✓ Loaded {len(models)} model(s) successfully!")
    
    # Load temperature from the first model's directory
    weights_dir = os.path.dirname(resolved_paths[0]) if len(resolved_paths) > 0 else "pytorch_cnn"
    T = load_temperature(weights_dir)
    
    # Execute prediction
    use_tta = not args.no_tta
    if args.image:
        if os.path.exists(args.image):
            predict_ensemble(args.image, models, device, T=T, use_tta=use_tta)
        else:
            print(f"❌ Error: Image file '{args.image}' does not exist.")
    else:
        print("\n🐾 Welcome to the Ensemble & Calibrated Inference Playground! 🐾")
        print("Enter 'exit' or 'quit' at any time to leave the playground.\n")
        while True:
            try:
                user_input = input("Enter path to a Cat/Dog image file: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                if os.path.exists(user_input):
                    predict_ensemble(user_input, models, device, T=T, use_tta=use_tta)
                else:
                    print(f"❌ Error: File '{user_input}' does not exist. Please check the path.")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()
