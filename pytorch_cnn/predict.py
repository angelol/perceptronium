import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import CustomCNN

def generate_ascii_preview(image_path, width=64):
    """
    Renders an image as terminal-based ASCII art, maintaining correct
    aspect ratio with font-height correction.
    """
    try:
        img = Image.open(image_path).convert("L")
        # Aspect ratio correction: terminal chars are about 2.2x taller than they are wide
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

def predict(image_path, model, device, image_size=224):
    """Preprocesses an input image, runs CNN inference, and outputs the class with confidence."""
    # 1. Show ASCII art preview
    print("\n📸 ASCII ART PREVIEW:")
    print("=" * 66)
    print(generate_ascii_preview(image_path))
    print("=" * 66)
    
    # 2. Image Preprocessing matching training validation
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
    
    # 3. Model Inference
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
        
    # Class determination (0.0 = Cat, 1.0 = Dog)
    if prob >= 0.5:
        predicted_class = "DOG"
        confidence = prob * 100.0
    else:
        predicted_class = "CAT"
        confidence = (1.0 - prob) * 100.0
        
    print("\n🔬 CNN INFERENCE RESULTS:")
    print("=" * 45)
    print(f"  • Raw Model Logit: {logits.item():.4f}")
    print(f"  • Dog Probability: {prob * 100.0:.2f}%")
    print(f"  • Predicted Class: \033[1;36m{predicted_class}\033[0m")
    print(f"  • Model Confidence: \033[1;32m{confidence:.2f}%\033[0m")
    print("=" * 45 + "\n")

def main():
    weights_path = "pytorch_cnn/model.pth"
    if not os.path.exists(weights_path) and os.path.exists("model.pth"):
        weights_path = "model.pth"
        
    # Check if weights file exists
    if not os.path.exists(weights_path):
        print(f"❌ Error: Model weights file '{weights_path}' not found.")
        print("Please run train.py first to train the network and save the weights.")
        sys.exit(1)
        
    # Device Selection
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading custom CNN on: {device.type.upper()}...")
    
    # Instantiate and load model
    model = CustomCNN().to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("✓ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        sys.exit(1)
        
    # Interactive predict loop
    if len(sys.argv) > 1:
        # Filepath passed via argument
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            predict(img_path, model, device)
        else:
            print(f"❌ Error: File '{img_path}' does not exist.")
    else:
        print("\n🐾 Welcome to the PyTorch Custom CNN Playground! 🐾")
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
                    predict(user_input, model, device)
                else:
                    print(f"❌ Error: File '{user_input}' does not exist. Please check the path.")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
