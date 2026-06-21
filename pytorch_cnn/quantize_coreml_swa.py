import os
import sys

# Ensure coremltools is imported
try:
    import coremltools as ct
except ImportError:
    print("❌ 'coremltools' is not installed. Please install coremltools or run export_coreml_swa.py first.", flush=True)
    sys.exit(1)

def main():
    print("🚀 Starting CoreML INT8 Post-Training Quantization...", flush=True)
    
    in_path = "pytorch_cnn/model_swa.mlpackage"
    out_path = "pytorch_cnn/model_swa_quantized.mlpackage"
    
    if not os.path.exists(in_path):
        print(f"❌ Error: Source CoreML package not found at '{in_path}'.", flush=True)
        print("Please run export_coreml_swa.py first to compile the base model.", flush=True)
        sys.exit(1)
        
    print(f"Loading base CoreML model from: '{in_path}'", flush=True)
    mlmodel = ct.models.MLModel(in_path)
    
    print("⚡ Compressing weights to 8-bit linear symmetric format...", flush=True)
    try:
        # Modern coremltools.optimize API (preferred for mlprogram structures)
        import coremltools.optimize.coreml as cto
        
        config = cto.LinearQuantizerConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                weight_threshold=512
            )
        )
        quantized_model = cto.quantize_weights(mlmodel, config)
        print("✓ Optimization using modern cto.quantize_weights completed!", flush=True)
    except Exception as e:
        print(f"⚠️ Modern cto.quantize_weights failed: {e}. Falling back to classical quantize_weights...", flush=True)
        try:
            # Classical weight quantization utility
            from coremltools.models.neural_network.quantization_utils import quantize_weights
            quantized_model = quantize_weights(mlmodel, nbits=8, quantization_mode="linear_symmetric")
            print("✓ Optimization using classical quantize_weights completed!", flush=True)
        except Exception as e2:
            print(f"❌ Classical quantization also failed: {e2}.", flush=True)
            sys.exit(1)
            
    quantized_model.save(out_path)
    print(f"✓ Quantized CoreML model saved to: '{out_path}'", flush=True)
    
    # Compare file sizes if possible
    def get_size(path):
        total_size = 0
        if os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        else:
            total_size = os.path.getsize(path)
        return total_size / (1024 * 1024) # MB
        
    base_size = get_size(in_path)
    quant_size = get_size(out_path)
    print(f"\n📈 Compression Summary:", flush=True)
    print(f"  • Base Model Size:      {base_size:6.2f} MB", flush=True)
    print(f"  • Quantized Model Size: {quant_size:6.2f} MB", flush=True)
    print(f"  • Savings Rate:         {((base_size - quant_size) / base_size) * 100.0:5.1f}%", flush=True)

if __name__ == "__main__":
    main()
