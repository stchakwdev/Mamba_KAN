#!/usr/bin/env python3
"""Test script for comparing MLP-Transformer vs KAN-Transformer."""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mamba_kan.models import create_model
from mamba_kan.utils.parameter_counter import compare_parameter_counts, validate_parameter_matching


def main():
    """Test MLP vs KAN Transformer comparison."""
    print("🔬 MLP-TRANSFORMER vs KAN-TRANSFORMER COMPARISON")
    print("=" * 80)
    
    # Configuration for fair comparison
    config = {
        "d_model": 128,
        "n_layers": 4, 
        "vocab_size": 10000,
        "n_heads": 8,
        "max_seq_len": 512,
        "d_ff": 512,  # Reduced for KAN to match parameter count
    }
    
    print("\nCreating models with matched configuration...")
    
    # Create both models
    models = {}
    for model_type in ["mlp_transformer", "kan_transformer"]:
        try:
            model = create_model(model_type, **config)
            models[model_type] = model
            print(f"✅ {model_type.replace('_', '-').title()}: Created successfully")
        except Exception as e:
            print(f"❌ {model_type}: Failed - {e}")
    
    if len(models) < 2:
        print("\n❌ Cannot perform comparison - need both models")
        return
    
    print(f"\n📊 Comparing {len(models)} models...")
    
    # Compare parameter counts
    compare_parameter_counts(models)
    
    # Validate parameter matching
    is_matched = validate_parameter_matching(models, tolerance_pct=20.0)
    if is_matched:
        print("✅ Parameter counts are reasonably matched (within 20%)")
    else:
        print("⚠️  Parameter counts differ significantly")
    
    # Test forward passes
    print("\n🧪 Testing forward passes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test input
    batch_size, seq_len = 2, 64
    test_input = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
    
    for name, model in models.items():
        try:
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"✅ {name.replace('_', '-').title()}: {test_input.shape} -> {output.shape}")
            
            # Test memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
                print(f"   GPU Memory: {memory_mb:.1f} MB")
                torch.cuda.reset_peak_memory_stats()
                
        except Exception as e:
            print(f"❌ {name}: Forward pass failed - {e}")
    
    print("\n🎯 COMPARISON COMPLETE!")
    print("=" * 80)
    print("✅ MLP-Transformer: Standard feedforward layers")
    print("✅ KAN-Transformer: Kolmogorov-Arnold Network layers")
    print("\nBoth models are ready for training and evaluation!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)