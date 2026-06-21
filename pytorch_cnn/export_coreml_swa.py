import os
import sys
import subprocess

# Ensure coremltools is installed in the virtual environment
try:
    import coremltools as ct
except ImportError:
    print("📦 'coremltools' not found. Installing now...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "coremltools"])
    import coremltools as ct

import torch
import torch.nn as nn

# Ensure imports work from project root or pytorch_cnn subdirectory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from model import CustomCNN

def main():
    print("🚀 Starting CoreML Export for SWA Model...", flush=True)
    
    device = torch.device("cpu") # Exporting on CPU is safer and fully supported by CoreML
    
    weights_path = "pytorch_cnn/model_swa.pth"
    if not os.path.exists(weights_path) and os.path.exists("model_swa.pth"):
        weights_path = "model_swa.pth"
        
    print(f"Loading SWA model weights from: {weights_path}", flush=True)
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: SWA checkpoint not found at {weights_path}.", flush=True)
        print("Please train a model first, or verify your weights path.", flush=True)
        sys.exit(1)
        
    # 1. Instantiate and load model
    model = CustomCNN(attention_type="cbam").to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 2. Create trace input
    dummy_input = torch.randn(1, 3, 224, 224)
    print("⚡ Tracing PyTorch model graph...", flush=True)
    traced_model = torch.jit.trace(model, dummy_input)
    
    # 3. Configure CoreML image inputs (with built-in ImageNet pre-processing)
    scale_r = 1.0 / (255.0 * 0.229)
    scale_g = 1.0 / (255.0 * 0.224)
    scale_b = 1.0 / (255.0 * 0.225)
    bias_r = -0.485 / 0.229
    bias_g = -0.456 / 0.224
    bias_b = -0.406 / 0.225
    
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, 224, 224),
        scale_red=scale_r, scale_green=scale_g, scale_blue=scale_b,
        bias_red=bias_r, bias_green=bias_g, bias_blue=bias_b,
        color_layout=ct.colorlayout.RGB
    )
    
    # 4. Convert traced PyTorch graph to CoreML
    print("⚡ Converting PyTorch graph to CoreML format...", flush=True)
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        classifier_config=None, # Output raw logits for flexibility
        convert_to="mlprogram"
    )
    
    # Save the CoreML model package
    out_path = "pytorch_cnn/model_swa.mlpackage"
    mlmodel.save(out_path)
    print(f"✓ CoreML model successfully exported and saved to: '{out_path}'", flush=True)

if __name__ == "__main__":
    main()
