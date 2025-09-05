"""Comprehensive system test with progressive dependency checking."""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """Test basic Python and torch imports."""
    print("="*60)
    print("TESTING BASIC IMPORTS")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        return False


def test_configs():
    """Test configuration system."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION SYSTEM")
    print("="*60)
    
    try:
        from mamba_kan.configs.base_config import BaseConfig, TransformerConfig, MambaConfig, KANConfig
        from mamba_kan.configs.model_configs import get_model_config
        
        # Test config creation
        for model_type in ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]:
            config = get_model_config(model_type, d_model=128, n_layers=2)
            print(f"✅ {model_type}: {type(config).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        traceback.print_exc()
        return False


def test_utilities():
    """Test parameter counting utilities.""" 
    print("\n" + "="*60)
    print("TESTING UTILITIES")
    print("="*60)
    
    try:
        import torch.nn as nn
        from mamba_kan.utils.parameter_counter import count_parameters, get_parameter_breakdown
        
        # Create simple test model
        model = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        param_count = count_parameters(model)
        breakdown = get_parameter_breakdown(model)
        
        print(f"✅ Parameter counting: {param_count:,} parameters")
        print(f"✅ Parameter breakdown: {len(breakdown)} components")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        traceback.print_exc()
        return False


def test_mlp_transformer():
    """Test MLP-Transformer (should work with just PyTorch)."""
    print("\n" + "="*60)
    print("TESTING MLP-TRANSFORMER (BASELINE)")
    print("="*60)
    
    try:
        import torch
        # First test components
        from mamba_kan.configs.model_configs import MLPTransformerConfig
        from mamba_kan.models.components.kan_layers import MLPBlock
        from mamba_kan.models.components.transformer_layers import MultiHeadAttention, TransformerBlock
        
        config = MLPTransformerConfig(d_model=64, n_layers=2, vocab_size=1000, n_heads=4, d_ff=256)
        
        # Test MLP block
        mlp = MLPBlock(config.d_model, config.d_ff, config.dropout)
        x = torch.randn(2, 10, 64)
        mlp_out = mlp(x)
        print(f"✅ MLPBlock: {x.shape} -> {mlp_out.shape}")
        
        # Test attention
        attention = MultiHeadAttention(config)
        attn_out = attention(x)
        print(f"✅ MultiHeadAttention: {x.shape} -> {attn_out.shape}")
        
        # Test transformer block
        block = TransformerBlock(config, use_kan=False)
        block_out = block(x)
        print(f"✅ TransformerBlock: {x.shape} -> {block_out.shape}")
        
        # Test full model
        from mamba_kan.models.mlp_transformer import MLPTransformer
        model = MLPTransformer(config)
        
        # Test forward pass
        dummy_input = torch.randint(0, 1000, (2, 10))
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        print(f"✅ Full MLPTransformer: {dummy_input.shape} -> {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ MLP-Transformer test failed: {e}")
        traceback.print_exc()
        return False


def test_kan_dependencies():
    """Test KAN dependencies and models."""
    print("\n" + "="*60)
    print("TESTING KAN DEPENDENCIES AND MODELS")
    print("="*60)
    
    try:
        import pykan
        print(f"✅ PyKAN available: version {getattr(pykan, '__version__', 'unknown')}")
        
        # Test KAN creation
        from pykan import KAN
        kan = KAN(width=[64, 128, 64], grid=5, k=3)
        x = torch.randn(10, 64)
        kan_out = kan(x)
        print(f"✅ Basic KAN: {x.shape} -> {kan_out.shape}")
        
        # Test KAN components
        from mamba_kan.models.components.kan_layers import KANBlock, KANProjection
        from mamba_kan.configs.base_config import KANConfig
        
        kan_config = KANConfig()
        kan_block = KANBlock(64, 256, kan_config)
        
        x_seq = torch.randn(2, 10, 64)
        kan_block_out = kan_block(x_seq)
        print(f"✅ KANBlock: {x_seq.shape} -> {kan_block_out.shape}")
        
        # Test KAN-Transformer
        from mamba_kan.models.kan_transformer import KANTransformer
        from mamba_kan.configs.model_configs import KANTransformerConfig
        
        config = KANTransformerConfig(d_model=64, n_layers=2, vocab_size=1000, n_heads=4, d_ff=128)
        kan_transformer = KANTransformer(config)
        
        dummy_input = torch.randint(0, 1000, (2, 10))
        kan_transformer.eval()
        with torch.no_grad():
            logits = kan_transformer(dummy_input)
        print(f"✅ KANTransformer: {dummy_input.shape} -> {logits.shape}")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyKAN not available: {e}")
        print("To install: pip install pykan")
        return False
    except Exception as e:
        print(f"❌ KAN test failed: {e}")
        traceback.print_exc()
        return False


def test_mamba_dependencies():
    """Test Mamba dependencies and models."""
    print("\n" + "="*60)
    print("TESTING MAMBA DEPENDENCIES AND MODELS")
    print("="*60)
    
    try:
        import torch
        import mamba_ssm
        print(f"✅ Mamba-SSM available")
        
        # Get device (prefer CUDA for Mamba)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Test basic Mamba
        from mamba_ssm import Mamba
        mamba = Mamba(d_model=64, d_state=16).to(device)
        
        x = torch.randn(2, 10, 64, device=device)
        mamba_out = mamba(x)
        print(f"✅ Basic Mamba: {x.shape} -> {mamba_out.shape}")
        
        # Test MLP-Mamba
        from mamba_kan.models.mlp_mamba import MLPMamba
        from mamba_kan.configs.model_configs import MLPMambaConfig
        
        config = MLPMambaConfig(d_model=64, n_layers=2, vocab_size=1000, d_state=16, expand=2.0)
        mlp_mamba = MLPMamba(config).to(device)
        
        dummy_input = torch.randint(0, 1000, (2, 10), device=device)
        mlp_mamba.eval()
        with torch.no_grad():
            logits = mlp_mamba(dummy_input)
        print(f"✅ MLPMamba: {dummy_input.shape} -> {logits.shape}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Mamba-SSM not available: {e}")
        print("To install: pip install mamba-ssm")
        return False
    except Exception as e:
        print(f"❌ Mamba test failed: {e}")
        traceback.print_exc()
        return False


def test_full_comparison():
    """Test full model comparison if all dependencies available."""
    print("\n" + "="*60)
    print("TESTING FULL MODEL COMPARISON")
    print("="*60)
    
    try:
        from mamba_kan.models import create_model
        from mamba_kan.utils.parameter_counter import compare_parameter_counts, validate_parameter_matching
        
        print("Creating all four model variants...")
        models = {}
        
        config_override = {
            "d_model": 128,
            "n_layers": 2,
            "vocab_size": 5000,
            "max_seq_len": 256
        }
        
        for model_type in ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]:
            try:
                model = create_model(model_type, **config_override)
                models[model_type] = model
                print(f"✅ {model_type} created")
            except Exception as e:
                print(f"❌ {model_type} failed: {e}")
        
        if len(models) >= 2:
            print(f"\nComparing {len(models)} models...")
            compare_parameter_counts(models)
            
            is_valid = validate_parameter_matching(models, tolerance_pct=15.0)
            if is_valid:
                print("✅ Parameter counts are reasonably matched")
            else:
                print("⚠️ Parameter counts differ significantly")
        
        return len(models) == 4
        
    except Exception as e:
        print(f"❌ Full comparison failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run comprehensive system test."""
    print("🧪 MAMBA-KAN SYSTEM TEST")
    print("=" * 80)
    
    test_results = {
        "basic_imports": test_basic_imports(),
        "configs": test_configs(),
        "utilities": test_utilities(),
        "mlp_transformer": test_mlp_transformer(),
        "kan_dependencies": test_kan_dependencies(),
        "mamba_dependencies": test_mamba_dependencies(),
    }
    
    # Only run full test if core components work
    if all([test_results["basic_imports"], test_results["configs"], test_results["utilities"]]):
        test_results["full_comparison"] = test_full_comparison()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    total_passed = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if test_results.get("basic_imports") and test_results.get("configs"):
        print("\n🎯 NEXT STEPS:")
        if not test_results.get("kan_dependencies"):
            print("1. Install PyKAN: pip install pykan")
        if not test_results.get("mamba_dependencies"):
            print("2. Install Mamba: pip install mamba-ssm")
        if test_results.get("kan_dependencies") and test_results.get("mamba_dependencies"):
            print("🎉 All dependencies available! Ready for full experiments.")
    
    return all(test_results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)