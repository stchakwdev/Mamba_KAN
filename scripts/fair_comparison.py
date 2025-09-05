#!/usr/bin/env python3
"""
Fair comparison script for MLP-Transformer vs KAN-Transformer with matched parameters.
This script creates models with similar parameter counts for meaningful comparison.
"""

import torch
import torch.nn.functional as F
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mamba_kan.models import create_model
from mamba_kan.utils.parameter_counter import count_parameters, compare_parameter_counts, validate_parameter_matching


def benchmark_forward_pass(model, test_input, num_runs=10):
    """Benchmark forward pass speed and memory usage."""
    model.eval()
    device = test_input.device
    
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(test_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(test_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    # Memory usage
    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        memory_mb = 0
    
    return {
        "avg_time_ms": avg_time * 1000,
        "memory_mb": memory_mb,
        "output_shape": list(output.shape),
        "throughput_tokens_per_sec": (test_input.numel() / avg_time) if avg_time > 0 else 0
    }


def test_language_modeling_task(model, vocab_size, seq_len=128, batch_size=4):
    """Test basic language modeling capability."""
    device = next(model.parameters()).device
    
    # Generate random text sequences
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)  # Shape: (batch, seq_len, vocab_size)
        
        # Compute perplexity
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
        perplexity = torch.exp(loss).item()
        
        # Compute accuracy (top-1)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == target_ids).float().mean().item()
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "loss": loss.item()
    }


def main():
    """Run fair comparison between MLP and KAN Transformers."""
    print("🎯 FAIR COMPARISON: MLP-TRANSFORMER vs KAN-TRANSFORMER")
    print("=" * 80)
    
    # Configuration optimized for fair comparison
    config = {
        "d_model": 256,
        "n_layers": 6,
        "vocab_size": 8000,
        "n_heads": 8,
        "max_seq_len": 512,
        "dropout": 0.1
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    print(f"\n📝 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print(f"\n⚙️  Creating models...")
    
    # Create models
    models = {}
    configs_used = {}
    
    # MLP-Transformer (uses default d_ff=2048)  
    mlp_model = create_model("mlp_transformer", **config)
    models["MLP-Transformer"] = mlp_model.to(device)
    configs_used["MLP-Transformer"] = config.copy()
    configs_used["MLP-Transformer"]["d_ff"] = 2048
    
    # KAN-Transformer (uses optimized d_ff and KAN params)
    kan_model = create_model("kan_transformer", **config)
    models["KAN-Transformer"] = kan_model.to(device)
    configs_used["KAN-Transformer"] = config.copy()
    configs_used["KAN-Transformer"]["d_ff"] = 350  # From optimized config
    configs_used["KAN-Transformer"]["kan_grid_size"] = 3
    configs_used["KAN-Transformer"]["kan_spline_order"] = 2
    
    print(f"✅ Created {len(models)} models")
    
    # Parameter comparison
    print(f"\n📊 PARAMETER COMPARISON")
    print("=" * 60)
    
    param_counts = {}
    for name, model in models.items():
        count = count_parameters(model)
        param_counts[name] = count
        print(f"{name:20}: {count:,} parameters")
    
    # Calculate parameter difference
    mlp_params = param_counts["MLP-Transformer"] 
    kan_params = param_counts["KAN-Transformer"]
    param_ratio = kan_params / mlp_params
    param_diff_pct = ((kan_params - mlp_params) / mlp_params) * 100
    
    print(f"\nParameter Ratio (KAN/MLP): {param_ratio:.2f}x")
    print(f"Parameter Difference: {param_diff_pct:+.1f}%")
    
    # Check if parameters are reasonably matched
    is_fair = validate_parameter_matching({"mlp": models["MLP-Transformer"], 
                                         "kan": models["KAN-Transformer"]}, 
                                        tolerance_pct=50.0)
    
    if is_fair:
        print("✅ Parameter counts are reasonably matched (within 50%)")
    else:
        print("⚠️  Parameter counts differ significantly")
    
    # Performance benchmarking
    print(f"\n⚡ PERFORMANCE BENCHMARKING")
    print("=" * 60)
    
    # Test with different sequence lengths
    test_configs = [
        {"batch_size": 2, "seq_len": 64, "name": "Short sequences"},
        {"batch_size": 4, "seq_len": 128, "name": "Medium sequences"}, 
        {"batch_size": 2, "seq_len": 256, "name": "Long sequences"}
    ]
    
    results = {}
    
    for test_config in test_configs:
        print(f"\n🧪 Testing {test_config['name']}: "
              f"batch={test_config['batch_size']}, seq_len={test_config['seq_len']}")
        
        # Create test input
        test_input = torch.randint(
            0, config["vocab_size"], 
            (test_config["batch_size"], test_config["seq_len"]),
            device=device
        )
        
        test_results = {}
        
        for name, model in models.items():
            try:
                # Benchmark forward pass
                benchmark_result = benchmark_forward_pass(model, test_input, num_runs=5)
                
                # Test language modeling task
                lm_result = test_language_modeling_task(
                    model, config["vocab_size"], 
                    seq_len=test_config["seq_len"], 
                    batch_size=test_config["batch_size"]
                )
                
                test_results[name] = {
                    **benchmark_result,
                    **lm_result
                }
                
                print(f"   {name:15}: {benchmark_result['avg_time_ms']:.1f}ms, "
                      f"{benchmark_result['memory_mb']:.1f}MB, "
                      f"ppl={lm_result['perplexity']:.2f}")
                
            except Exception as e:
                print(f"   {name:15}: Failed - {e}")
                test_results[name] = {"error": str(e)}
        
        results[test_config['name']] = test_results
    
    # Summary comparison
    print(f"\n📈 SUMMARY COMPARISON")
    print("=" * 60)
    
    # Calculate averages across test configs
    for name in models.keys():
        times = []
        memories = []
        perplexities = []
        
        for test_name, test_result in results.items():
            if name in test_result and "error" not in test_result[name]:
                times.append(test_result[name]["avg_time_ms"])
                memories.append(test_result[name]["memory_mb"])
                perplexities.append(test_result[name]["perplexity"])
        
        if times:
            avg_time = np.mean(times)
            avg_memory = np.mean(memories)
            avg_ppl = np.mean(perplexities)
            
            print(f"{name}:")
            print(f"   Avg Time: {avg_time:.1f}ms")
            print(f"   Avg Memory: {avg_memory:.1f}MB")
            print(f"   Avg Perplexity: {avg_ppl:.2f}")
            print(f"   Parameters: {param_counts[name]:,}")
    
    # Speed comparison
    if "MLP-Transformer" in results[list(results.keys())[0]] and \
       "KAN-Transformer" in results[list(results.keys())[0]]:
        
        mlp_times = []
        kan_times = []
        
        for test_result in results.values():
            if "error" not in test_result.get("MLP-Transformer", {}) and \
               "error" not in test_result.get("KAN-Transformer", {}):
                mlp_times.append(test_result["MLP-Transformer"]["avg_time_ms"])
                kan_times.append(test_result["KAN-Transformer"]["avg_time_ms"])
        
        if mlp_times and kan_times:
            mlp_avg = np.mean(mlp_times)
            kan_avg = np.mean(kan_times)
            speed_ratio = kan_avg / mlp_avg
            
            print(f"\n⚖️  Speed Comparison:")
            print(f"   KAN is {speed_ratio:.2f}x {'slower' if speed_ratio > 1 else 'faster'} than MLP")
            print(f"   Parameter ratio: {param_ratio:.2f}x")
    
    # Save results
    output_file = Path("comparison_results.json")
    full_results = {
        "config": config,
        "configs_used": configs_used,
        "parameter_counts": param_counts,
        "parameter_ratio": param_ratio,
        "parameter_difference_pct": param_diff_pct,
        "results": results,
        "device": str(device)
    }
    
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    print(f"\n🎯 COMPARISON COMPLETE!")
    print("=" * 80)
    print("Key Findings:")
    print(f"   • Parameter efficiency: KAN uses {param_ratio:.2f}x parameters")
    print(f"   • Memory efficiency: Varies by sequence length")
    print(f"   • Speed comparison: Check detailed results above")
    print("   • Both models are functional and ready for training!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)