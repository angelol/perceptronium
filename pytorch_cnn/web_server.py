import http.server
import json
import io
import os
import sys
import socket
import torch
import torchvision.transforms as transforms
from PIL import Image

# Ensure the model directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import CustomCNN

# Global Server Configuration
PORT_START = 8080
IMAGE_SIZE = 288 # Matching Run 11 progressive resolution
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

print(f"🚀 Initializing Perceptronium Web Server on: {DEVICE.type.upper()}", flush=True)

# Preprocessing Pipeline matching Run 11 training validation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and prepare model
print("⚙️ Instantiating CBAM-EfficientNet v4 (Ultra Hybrid)...", flush=True)
model = CustomCNN(attention_type="se").to(DEVICE)

weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_run11.pth")
if not os.path.exists(weights_path):
    print(f"❌ Error: Required weights '{weights_path}' not found locally.", flush=True)
    sys.exit(1)

print(f"⚙️ Loading weights from {weights_path}...", flush=True)
try:
    state_dict = torch.load(weights_path, map_location=DEVICE)
    # Automatically strip torch.compile '_orig_mod.' prefix if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('_orig_mod.', '')
        clean_state_dict[clean_key] = v
    model.load_state_dict(clean_state_dict)
    model.eval()
    print("✓ Model successfully initialized and placed in eval mode!", flush=True)
except Exception as e:
    print(f"❌ Failed to load model weights: {e}", flush=True)
    sys.exit(1)


# Find an available port robustly
def find_available_port(start_port):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                port += 1


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Perceptronium v4 — Neural Classifier</title>
  
  <!-- Premium Outfit Google Font -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  
  <style>
    :root {
      --bg-color: #060913;
      --panel-bg: rgba(17, 24, 39, 0.7);
      --border-color: rgba(255, 255, 255, 0.08);
      --text-primary: #f3f4f6;
      --text-secondary: #9ca3af;
      --accent-blue: #3b82f6;
      --accent-pink: #ec4899;
      --accent-green: #10b981;
      --neon-blue-glow: 0 0 25px rgba(59, 130, 246, 0.4);
      --neon-pink-glow: 0 0 25px rgba(236, 72, 153, 0.4);
      --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-color);
      color: var(--text-primary);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      background-image: 
        radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.12) 0%, transparent 40%);
      background-attachment: fixed;
      padding: 40px 20px;
    }

    /* Outer Wrapper */
    .app-container {
      width: 100%;
      max-width: 1100px;
      display: flex;
      flex-direction: column;
      gap: 30px;
    }

    /* Header Styling */
    header {
      text-align: center;
      margin-bottom: 10px;
    }

    h1 {
      font-size: 2.8rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, #60a5fa 0%, #f472b6 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 10px;
      text-shadow: 0 2px 20px rgba(96, 165, 250, 0.2);
    }

    .subtitle {
      color: var(--text-secondary);
      font-size: 1.1rem;
      font-weight: 300;
      max-width: 600px;
      margin: 0 auto;
      line-height: 1.5;
    }

    .badge-bar {
      display: flex;
      gap: 12px;
      justify-content: center;
      margin-top: 15px;
    }

    .meta-badge {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-color);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-secondary);
      backdrop-filter: blur(10px);
    }

    .meta-badge span {
      color: #60a5fa;
      font-weight: 600;
    }

    /* Grid Layout */
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 30px;
    }

    @media (min-width: 850px) {
      .grid {
        grid-template-columns: 1.1fr 1.2fr;
      }
    }

    /* Card Panels */
    .panel {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      border-radius: 24px;
      padding: 30px;
      backdrop-filter: blur(20px);
      box-shadow: var(--glass-shadow);
      display: flex;
      flex-direction: column;
      gap: 24px;
      position: relative;
      transition: border-color 0.3s ease;
    }

    /* Drag and Drop Zone */
    .dropzone {
      border: 2px dashed rgba(255, 255, 255, 0.12);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.01);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 60px 20px;
      text-align: center;
      position: relative;
      min-height: 320px;
    }

    .dropzone:hover {
      border-color: rgba(255, 255, 255, 0.25);
      background: rgba(255, 255, 255, 0.02);
      transform: translateY(-2px);
    }

    .dropzone.dragover {
      border-color: var(--accent-blue);
      background: rgba(59, 130, 246, 0.06);
      box-shadow: var(--neon-blue-glow);
      transform: scale(1.02);
    }

    .upload-icon {
      font-size: 3rem;
      margin-bottom: 16px;
      background: linear-gradient(135deg, #93c5fd, #f472b6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: pulse 2s infinite ease-in-out;
    }

    .dropzone p {
      font-size: 1.1rem;
      font-weight: 500;
      color: var(--text-primary);
      margin-bottom: 8px;
    }

    .dropzone span {
      font-size: 0.85rem;
      color: var(--text-secondary);
    }

    /* Hidden File Input */
    #file-input {
      display: none;
    }

    /* Preview Container */
    .preview-container {
      display: none;
      position: relative;
      width: 100%;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--border-color);
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
    }

    .preview-image {
      width: 100%;
      height: auto;
      max-height: 380px;
      object-fit: contain;
      display: block;
      background: #0b0f19;
    }

    .reupload-btn {
      position: absolute;
      top: 15px;
      right: 15px;
      background: rgba(17, 24, 39, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: var(--text-primary);
      padding: 8px 16px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      backdrop-filter: blur(5px);
      transition: all 0.2s ease;
      box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    .reupload-btn:hover {
      background: var(--text-primary);
      color: #111827;
      transform: translateY(-1px);
    }

    /* Dynamic Output Panel States */
    .output-empty {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      min-height: 320px;
      color: var(--text-secondary);
      text-align: center;
      font-weight: 300;
      gap: 12px;
    }

    .output-empty svg {
      width: 50px;
      height: 50px;
      stroke: rgba(255, 255, 255, 0.15);
      stroke-width: 1.5;
    }

    .output-active {
      display: none; /* Controlled by JS */
      animation: fadeInSlide 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Prediction Scale Slider (Cat vs Dog) */
    .classification-box {
      text-align: center;
      padding: 10px 0;
    }

    .class-label-main {
      font-size: 3rem;
      font-weight: 900;
      letter-spacing: -0.04em;
      margin-bottom: 4px;
      text-transform: uppercase;
      transition: text-shadow 0.3s ease;
    }

    .class-label-main.cat {
      color: var(--accent-pink);
      text-shadow: var(--neon-pink-glow);
    }

    .class-label-main.dog {
      color: var(--accent-blue);
      text-shadow: var(--neon-blue-glow);
    }

    .scale-container {
      position: relative;
      margin: 40px 0 25px 0;
      padding: 0 10px;
    }

    .scale-labels {
      display: flex;
      justify-content: space-between;
      font-weight: 700;
      font-size: 0.9rem;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
    }

    .scale-label-cat { color: var(--accent-pink); }
    .scale-label-dog { color: var(--accent-blue); }

    .scale-track {
      height: 12px;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent-pink) 0%, #a855f7 50%, var(--accent-blue) 100%);
      position: relative;
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }

    .scale-indicator {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: #ffffff;
      border: 4px solid var(--accent-pink);
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      transition: left 1s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.5s ease, box-shadow 0.5s ease;
      box-shadow: 0 0 15px rgba(255, 255, 255, 0.8);
    }

    /* Table Metrics */
    .metrics-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      background: rgba(255, 255, 255, 0.01);
      border-radius: 12px;
      overflow: hidden;
    }

    .metrics-table tr {
      border-bottom: 1px solid var(--border-color);
    }

    .metrics-table tr:last-child {
      border-bottom: none;
    }

    .metrics-table td {
      padding: 14px 16px;
      font-size: 0.95rem;
    }

    .metrics-table td:first-child {
      color: var(--text-secondary);
      font-weight: 400;
    }

    .metrics-table td:last-child {
      text-align: right;
      font-weight: 600;
      color: var(--text-primary);
    }

    .logit-val {
      font-family: monospace;
      font-size: 1.05rem;
    }

    /* Loading Overlay inside Output Panel */
    .loader-overlay {
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 16px;
      height: 100%;
      min-height: 320px;
    }

    .spinner {
      width: 48px;
      height: 48px;
      border: 3px solid rgba(255, 255, 255, 0.05);
      border-top-color: var(--accent-blue);
      border-bottom-color: var(--accent-pink);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    /* Footer */
    footer {
      text-align: center;
      margin-top: 50px;
      font-size: 0.85rem;
      color: rgba(255,255,255,0.25);
    }

    footer a {
      color: rgba(255,255,255,0.4);
      text-decoration: none;
    }

    footer a:hover {
      color: #60a5fa;
    }

    /* Keyframe Animations */
    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 0.95; }
      50% { transform: scale(1.05); opacity: 1; filter: drop-shadow(0 0 10px rgba(96, 165, 250, 0.3)); }
    }

    @keyframes fadeInSlide {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
  </style>
</head>
<body>

  <div class="app-container">
    <header>
      <h1>Perceptronium v4</h1>
      <div class="subtitle">Evaluate and analyze images using our custom 24.31M parameter Attention-Transformer hybrid model, trained entirely from scratch.</div>
      
      <div class="badge-bar">
        <div class="meta-badge">Architecture: <span>CBAM-EfficientNet v4</span></div>
        <div class="meta-badge">Hardware: <span id="hardware-tag">MPS (Metal)</span></div>
        <div class="meta-badge">Input Size: <span>288 × 288 px</span></div>
      </div>
    </header>

    <div class="grid">
      <!-- Upload Panel -->
      <div class="panel" id="upload-panel">
        <div class="dropzone" id="drop-zone">
          <div class="upload-icon">✦</div>
          <p>Drag & Drop Cat or Dog image</p>
          <span>or click to select file from your system</span>
        </div>
        <input type="file" id="file-input" accept="image/*">
        
        <div class="preview-container" id="preview-box">
          <img src="" alt="Selected Preview" class="preview-image" id="preview-img">
          <button class="reupload-btn" id="reupload-trigger">Analyze Another</button>
        </div>
      </div>

      <!-- Analysis Results Panel -->
      <div class="panel" id="results-panel">
        <!-- Overlay Loader -->
        <div class="loader-overlay" id="loader">
          <div class="spinner"></div>
          <div style="font-weight: 500; font-size: 1rem; color: var(--text-secondary);">Running inference on GPU...</div>
        </div>

        <!-- Initial Placeholder State -->
        <div class="output-empty" id="empty-state">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 16L8.586 11.414C9.367 10.633 10.633 10.633 11.414 11.414L16 16M14 14L15.586 12.414C16.367 11.633 17.633 11.633 18.414 12.414L20 14M14 8H14.01M6 20H18C19.1046 20 20 19.1046 20 18V6C20 4.89543 19.1046 4 18 4H6C4.89543 4 4 4.89543 4 6V18C4 19.1046 4.89543 20 6 20Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <div>Upload an image to start high-fidelity neural analysis</div>
        </div>

        <!-- Active Inference Result Area -->
        <div class="output-active" id="active-state">
          <div class="classification-box">
            <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); letter-spacing: 0.1em; margin-bottom: 4px;">PREDICTED CLASS</div>
            <div class="class-label-main" id="class-label">DOG</div>
            <div style="font-size: 1.1rem; font-weight: 400; color: var(--text-secondary);"><span style="color: var(--text-primary); font-weight:600;" id="conf-val">98.63%</span> Confidence</div>
          </div>

          <!-- Slider bar -->
          <div class="scale-container">
            <div class="scale-labels">
              <span class="scale-label-cat">🐈 CAT</span>
              <span class="scale-label-dog">🐕 DOG</span>
            </div>
            <div class="scale-track">
              <div class="scale-indicator" id="slider-indicator"></div>
            </div>
          </div>

          <!-- Quantitative Details Table -->
          <table class="metrics-table">
            <tbody>
              <tr>
                <td>Raw Model Logit</td>
                <td class="logit-val" id="metric-logit">+1.8975</td>
              </tr>
              <tr>
                <td>Continuous Probability</td>
                <td id="metric-prob">86.96% (Dog)</td>
              </tr>
              <tr>
                <td>Inference Device</td>
                <td id="metric-device">MPS (Metal GPU)</td>
              </tr>
              <tr>
                <td>Target Resolution</td>
                <td>288 × 288 px</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <footer>
      🤖 Built on the DeepMind advanced agentic coding framework. <a href="https://github.com/google/perceptronium" target="_blank">Perceptronium Project</a>
    </footer>
  </div>

  <script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewBox = document.getElementById('preview-box');
    const previewImg = document.getElementById('preview-img');
    const reuploadTrigger = document.getElementById('reupload-trigger');
    const emptyState = document.getElementById('empty-state');
    const activeState = document.getElementById('active-state');
    const loader = document.getElementById('loader');
    const hardwareTag = document.getElementById('hardware-tag');
    
    // UI Out
    const classLabel = document.getElementById('class-label');
    const confVal = document.getElementById('conf-val');
    const sliderIndicator = document.getElementById('slider-indicator');
    const metricLogit = document.getElementById('metric-logit');
    const metricProb = document.getElementById('metric-prob');
    const metricDevice = document.getElementById('metric-device');

    // Automatically detect hardware from backend on window load
    window.addEventListener('DOMContentLoaded', async () => {
      // Get the backend device type dynamically from the server
      try {
        const response = await fetch('/api/device');
        if (response.ok) {
          const data = await response.json();
          hardwareTag.innerText = data.device.toUpperCase();
          metricDevice.innerText = data.device === 'mps' ? 'MPS (Metal GPU)' : (data.device === 'cuda' ? 'NVIDIA GPU (CUDA)' : 'CPU');
        }
      } catch (e) {
        console.warn("Failed to fetch device details", e);
      }
    });

    // Wire up events
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleImageFile(e.target.files[0]);
      }
    });

    // Drag-over styling
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
      }, false);
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        handleImageFile(files[0]);
      }
    });

    reuploadTrigger.addEventListener('click', () => {
      // Reset view to original state
      fileInput.value = '';
      previewBox.style.display = 'none';
      previewImg.src = '';
      dropZone.style.display = 'flex';
      
      activeState.style.display = 'none';
      emptyState.style.display = 'flex';
    });

    function handleImageFile(file) {
      // 1. Show image preview
      const reader = new FileReader();
      reader.onload = function(e) {
        previewImg.src = e.target.result;
        dropZone.style.display = 'none';
        previewBox.style.display = 'block';
      };
      reader.readAsDataURL(file);

      // 2. Perform upload and inference
      runInference(file);
    }

    async function runInference(file) {
      // Reset UI state to Loading
      emptyState.style.display = 'none';
      activeState.style.display = 'none';
      loader.style.display = 'flex';

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          body: file // Send raw image bytes
        });

        if (!response.ok) {
          throw new Error("Prediction request failed");
        }

        const result = await response.json();
        
        // Hide loader
        loader.style.display = 'none';
        
        // Populate results
        displayPrediction(result);

      } catch (error) {
        loader.style.display = 'none';
        emptyState.style.display = 'flex';
        alert("❌ Error analyzing image: " + error.message);
      }
    }

    function displayPrediction(data) {
      // Show result section
      activeState.style.display = 'block';

      // 1. Set Class tag and colors
      classLabel.innerText = data.class;
      classLabel.className = 'class-label-main ' + data.class.toLowerCase();
      
      // 2. Set confidence values
      confVal.innerText = data.confidence.toFixed(2) + '%';
      
      // 3. Position and color slider indicator
      // data.probability is 0.0 (Pure Cat) to 1.0 (Pure Dog)
      const percent = data.probability * 100;
      sliderIndicator.style.left = percent + '%';
      
      if (data.class === 'CAT') {
        sliderIndicator.style.borderColor = 'var(--accent-pink)';
        sliderIndicator.style.boxShadow = '0 0 15px var(--accent-pink)';
      } else {
        sliderIndicator.style.borderColor = 'var(--accent-blue)';
        sliderIndicator.style.boxShadow = '0 0 15px var(--accent-blue)';
      }

      // 4. Fill Metrics Table
      metricLogit.innerText = (data.logit >= 0 ? '+' : '') + data.logit.toFixed(4);
      
      const probText = (data.probability * 100).toFixed(2) + '%';
      if (data.probability >= 0.5) {
        metricProb.innerText = probText + " (Dog Probability)";
      } else {
        metricProb.innerText = ((1.0 - data.probability) * 100).toFixed(2) + "% (Cat Probability)";
      }
    }
  </script>
</body>
</html>
"""


class WebUIServer(http.server.HTTPServer):
    pass


class WebUIRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        elif self.path == '/api/device':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"device": DEVICE.type}).encode('utf-8'))
        elif self.path == '/favicon.ico':
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')
            
    def do_POST(self):
        if self.path == '/api/predict':
            try:
                # Read content length
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    self.send_error_response("Empty request body")
                    return
                
                # Read raw binary image bytes
                img_bytes = self.rfile.read(content_length)
                
                # Load image in Pillow
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                
                # Preprocess image
                input_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                # Inference
                with torch.no_grad():
                    logits = model(input_tensor)
                    prob = torch.sigmoid(logits).item()
                    
                # Format output values (0.0 = Cat, 1.0 = Dog)
                if prob >= 0.5:
                    pred_class = "DOG"
                    confidence = prob * 100.0
                else:
                    pred_class = "CAT"
                    confidence = (1.0 - prob) * 100.0
                    
                response_data = {
                    "class": pred_class,
                    "probability": prob,
                    "confidence": confidence,
                    "logit": logits.item()
                }
                
                # Send JSON response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                
            except Exception as e:
                self.send_error_response(f"Inference error: {str(e)}")
        else:
            self.send_response(404)
            self.end_headers()
            
    def send_error_response(self, message):
        self.send_response(400)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))
        
    def log_message(self, format, *args):
        # Clean console log output format
        print(f"[HTTP] - {format%args}", flush=True)


def run():
    port = find_available_port(PORT_START)
    server_address = ('127.0.0.1', port)
    httpd = WebUIServer(server_address, WebUIRequestHandler)
    
    print("\n" + "=" * 66, flush=True)
    print("✨ PERCEPTRONIUM NEURAL EXPLORER IS READY! ✨", flush=True)
    print("=" * 66, flush=True)
    print(f"  • Model:        CBAM-EfficientNet v4 (Run 11, 24.31M params)", flush=True)
    print(f"  • Device:       {DEVICE.type.upper()}", flush=True)
    print(f"  • Address:      http://127.0.0.1:{port}", flush=True)
    print("=" * 66 + "\\n", flush=True)
    print("Please cmd+click the link above to open the Web UI in your browser.", flush=True)
    print("Press Ctrl+C to terminate the web server.\\n", flush=True)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\n🛑 Terminating Web Server...", flush=True)
        httpd.server_close()
        print("✓ Web Server shut down cleanly.", flush=True)


if __name__ == '__main__':
    run()
