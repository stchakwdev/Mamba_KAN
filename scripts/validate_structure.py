"""Validate project structure without external dependencies."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def validate_imports():
    """Validate that core modules can be imported."""
    try:
        # Test config imports
        from mamba_kan.configs.base_config import BaseConfig, TransformerConfig, MambaConfig, KANConfig
        print("✅ Config imports successful")
        
        # Test that configs can be instantiated
        base_config = BaseConfig()
        transformer_config = TransformerConfig()
        mamba_config = MambaConfig() 
        kan_config = KANConfig()
        print("✅ Config instantiation successful")
        
        # Test model config factory
        from mamba_kan.configs.model_configs import get_model_config
        
        for model_type in ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]:
            config = get_model_config(model_type, d_model=64, n_layers=2)
            print(f"✅ {model_type} config created: d_model={config.d_model}")
        
        print("✅ All structure validation passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False


def check_project_structure():
    """Check that all expected files and directories exist."""
    expected_structure = [
        "mamba_kan/__init__.py",
        "mamba_kan/models/__init__.py",
        "mamba_kan/models/base.py",
        "mamba_kan/models/mlp_transformer.py",
        "mamba_kan/models/kan_transformer.py", 
        "mamba_kan/models/mlp_mamba.py",
        "mamba_kan/models/kan_mamba.py",
        "mamba_kan/models/components/__init__.py",
        "mamba_kan/models/components/kan_layers.py",
        "mamba_kan/models/components/mamba_layers.py",
        "mamba_kan/models/components/transformer_layers.py",
        "mamba_kan/configs/__init__.py",
        "mamba_kan/configs/base_config.py",
        "mamba_kan/configs/model_configs.py",
        "mamba_kan/utils/__init__.py",
        "mamba_kan/utils/parameter_counter.py",
        "mamba_kan/evaluation/__init__.py",
        "mamba_kan/evaluation/metrics.py",
        "scripts/train.py",
        "scripts/compare_models.py",
        "tests/test_models.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        "LICENSE",
        ".gitignore",
        "CLAUDE.md",
    ]
    
    missing_files = []
    for file_path in expected_structure:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ All expected files present")
        return True


def main():
    print("Validating Mamba-KAN project structure...")
    print("=" * 50)
    
    structure_ok = check_project_structure()
    imports_ok = validate_imports()
    
    if structure_ok and imports_ok:
        print("\n🎉 Project structure validation completed successfully!")
        print("\nTo get started:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run basic test: python scripts/train.py --model_type mlp_transformer --compare_all")
        print("3. Run full comparison: python scripts/compare_models.py")
    else:
        print("\n❌ Project structure validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()