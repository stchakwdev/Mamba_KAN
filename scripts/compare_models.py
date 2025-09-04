"""Compare all four model variants."""

import torch
import argparse
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mamba_kan.models import create_model
from mamba_kan.utils.parameter_counter import compare_parameter_counts, validate_parameter_matching, estimate_memory_usage


def benchmark_inference_speed(model, device, seq_lengths=[128, 256, 512], num_runs=10):
    """Benchmark inference speed across different sequence lengths."""
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for seq_len in seq_lengths:
        times = []
        
        for _ in range(num_runs):
            dummy_input = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
                
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        results[seq_len] = avg_time
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare all model variants")
    parser.add_argument("--d_model", type=int, default=256,
                       help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4,
                       help="Number of layers")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--benchmark_speed", action="store_true",
                       help="Benchmark inference speed")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model config: d_model={args.d_model}, n_layers={args.n_layers}")
    print("\n" + "="*80)
    
    # Create all four models
    print("Creating all four model variants...")
    models = {}
    
    for model_type in ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]:
        print(f"  Creating {model_type}...")
        try:
            model = create_model(
                model_type,
                d_model=args.d_model,
                n_layers=args.n_layers,
                device=str(device)
            )
            models[model_type] = model
            print(f"  ✅ {model_type} created successfully")
        except Exception as e:
            print(f"  ❌ Failed to create {model_type}: {e}")
            continue
    
    if not models:
        print("❌ No models created successfully. Check dependencies.")
        return
    
    # Compare parameter counts
    print("\n" + "="*80)
    compare_parameter_counts(models)
    
    # Validate parameter matching
    if validate_parameter_matching(models, tolerance_pct=10.0):
        print("\n✅ Parameter counts are well-matched")
    else:
        print("\n⚠️  Parameter counts differ significantly")
    
    # Memory usage estimation
    print("\n" + "="*60)
    print("MEMORY USAGE ESTIMATES")
    print("="*60)
    
    for name, model in models.items():
        memory_est = estimate_memory_usage(model, batch_size=8, seq_len=512)
        print(f"{name:<20}: {memory_est['total_mb']:.1f} MB "
              f"({memory_est['total_gb']:.2f} GB)")
    
    # Benchmark inference speed
    if args.benchmark_speed and device.type == "cuda":
        print("\n" + "="*60)
        print("INFERENCE SPEED BENCHMARK")
        print("="*60)
        
        seq_lengths = [128, 256, 512]
        
        for name, model in models.items():
            print(f"\nBenchmarking {name}...")
            try:
                speeds = benchmark_inference_speed(model, device, seq_lengths, num_runs=5)
                for seq_len, time_taken in speeds.items():
                    print(f"  Seq len {seq_len}: {time_taken*1000:.2f}ms")
            except Exception as e:
                print(f"  ❌ Benchmark failed: {e}")
    
    # Test basic functionality
    print("\n" + "="*60)
    print("FUNCTIONALITY TEST")
    print("="*60)
    
    for name, model in models.items():
        try:
            model = model.to(device)
            model.eval()
            
            # Test forward pass
            dummy_input = torch.randint(0, model.config.vocab_size, (2, 64), device=device)
            with torch.no_grad():
                output = model(dummy_input)
            
            # Test generation
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=5, do_sample=False)
            
            print(f"✅ {name:<20}: Forward pass ✓, Generation ✓")
            
        except Exception as e:
            print(f"❌ {name:<20}: Failed - {e}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()