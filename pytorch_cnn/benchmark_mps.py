import os
import sys
import time
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
    print("🚀 Starting Apple Silicon Metal GPU (MPS) Serving Benchmark...", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Benchmark running on: {device.type.upper()}", flush=True)
    
    # Instantiate v3 model
    model = CustomCNN(attention_type="cbam").to(device)
    model.eval()
    
    # Create dummy inputs: batch size of 1 for real-time latency, batch size of 32 for high throughput
    dummy_1 = torch.randn(1, 3, 224, 224).to(device)
    dummy_32 = torch.randn(32, 3, 224, 224).to(device)
    
    # We will test multiple serving modes
    modes = {}
    
    # Mode 1: PyTorch Eager (MPS)
    modes["PyTorch Eager (MPS)"] = model
    
    # Mode 2: TorchScript Traced (MPS)
    print("⚡ Tracing model to TorchScript JIT...", flush=True)
    try:
        traced_model = torch.jit.trace(model, dummy_1)
        modes["TorchScript Traced (MPS)"] = traced_model
        print("✓ TorchScript trace successful!", flush=True)
    except Exception as e:
        print(f"⚠️ TorchScript tracing failed: {e}", flush=True)
        
    # Mode 3: torch.compile (MPS)
    print("⚡ Compiling model with torch.compile...", flush=True)
    try:
        compiled_model = torch.compile(model, backend="aot_eager")
        # Warmup compiled model
        _ = compiled_model(dummy_1)
        modes["torch.compile aot_eager (MPS)"] = compiled_model
        print("✓ torch.compile successful!", flush=True)
    except Exception as e:
        print(f"⚠️ torch.compile failed or not supported on this version: {e}", flush=True)
        
    # Benchmark function
    def run_benchmark(net, dummy_input, label, num_runs=50, warmup_runs=10):
        # Warmup
        for _ in range(warmup_runs):
            _ = net(dummy_input)
        if device.type == "mps":
            torch.mps.synchronize()
            
        start_time = time.time()
        for _ in range(num_runs):
            _ = net(dummy_input)
        if device.type == "mps":
            torch.mps.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_latency = (total_time / num_runs) * 1000.0 # ms
        fps = (num_runs * dummy_input.size(0)) / total_time
        return avg_latency, fps

    print("\n=== LATENCY BENCHMARK (Batch Size = 1) ===", flush=True)
    for name, net in modes.items():
        try:
            latency, fps = run_benchmark(net, dummy_1, name, num_runs=50)
            print(f"  • {name:30s} | Avg Latency: {latency:8.2f} ms | Throughput: {fps:8.2f} FPS", flush=True)
        except Exception as e:
            print(f"  • {name:30s} | Error: {e}", flush=True)
            
    print("\n=== THROUGHPUT BENCHMARK (Batch Size = 32) ===", flush=True)
    for name, net in modes.items():
        try:
            latency, fps = run_benchmark(net, dummy_32, name, num_runs=15)
            print(f"  • {name:30s} | Avg Latency: {latency:8.2f} ms | Throughput: {fps:8.2f} FPS", flush=True)
        except Exception as e:
            print(f"  • {name:30s} | Error: {e}", flush=True)

if __name__ == "__main__":
    main()
