#!/usr/bin/env python3
"""GPU testing pipeline for Mamba-KAN on RunPod."""

import os
import sys
import json
import time
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path("/workspace/Mamba_KAN")
sys.path.insert(0, str(project_root))


class GPUTestPipeline:
    """Comprehensive testing pipeline for RunPod GPU environment."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.get_environment_info(),
            "tests": {}
        }
        
    def get_environment_info(self):
        """Collect environment information."""
        info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device)
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            })
            
        return info
    
    def test_dependencies(self):
        """Test all critical dependencies."""
        print("🧪 Testing Dependencies")
        print("=" * 40)
        
        tests = {
            "pytorch": self.test_pytorch,
            "pykan": self.test_pykan,
            "mamba_ssm": self.test_mamba_ssm,
            "system_imports": self.test_system_imports
        }
        
        results = {}
        for test_name, test_func in tests.items():
            print(f"\nTesting {test_name}...")
            try:
                success = test_func()
                results[test_name] = {"success": success, "error": None}
                print(f"✅ {test_name}: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                results[test_name] = {"success": False, "error": str(e)}
                print(f"❌ {test_name}: FAIL - {e}")
                
        self.results["tests"]["dependencies"] = results
        return all(result["success"] for result in results.values())
    
    def test_pytorch(self):
        """Test PyTorch and CUDA."""
        # Basic tensor operations
        x = torch.randn(1000, 1000, device=self.device)
        y = torch.matmul(x, x.T)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            
        return True
    
    def test_pykan(self):
        """Test PyKAN import and basic operations."""
        try:
            import pykan
            from kan import KAN
            
            # Create simple KAN
            kan = KAN([10, 20, 10], grid=3)
            x = torch.randn(5, 10)
            y = kan(x)
            
            return y.shape == (5, 10)
        except ImportError:
            return False
    
    def test_mamba_ssm(self):
        """Test Mamba SSM import and basic operations."""
        try:
            import mamba_ssm
            from mamba_ssm import Mamba
            
            # Create simple Mamba layer
            mamba = Mamba(d_model=64, d_state=16).to(self.device)
            x = torch.randn(2, 100, 64, device=self.device)
            y = mamba(x)
            
            return y.shape == (2, 100, 64)
        except ImportError:
            return False
    
    def test_system_imports(self):
        """Test project system imports."""
        try:
            from mamba_kan.configs.base_config import BaseConfig
            from mamba_kan.utils.parameter_counter import count_parameters
            return True
        except ImportError:
            return False
    
    def test_model_creation(self):
        """Test creating all four model variants."""
        print("\n🏗️ Testing Model Creation")
        print("=" * 40)
        
        try:
            from mamba_kan.models import create_model
            
            model_types = ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]
            test_config = {
                "d_model": 128,
                "n_layers": 2,
                "vocab_size": 1000,
                "device": str(self.device)
            }
            
            models = {}
            results = {}
            
            for model_type in model_types:
                print(f"Creating {model_type}...")
                try:
                    model = create_model(model_type, **test_config)
                    models[model_type] = model
                    
                    # Test forward pass
                    model = model.to(self.device)
                    model.eval()
                    
                    dummy_input = torch.randint(0, 1000, (2, 32), device=self.device)
                    with torch.no_grad():
                        output = model(dummy_input)
                    
                    results[model_type] = {
                        "success": True,
                        "output_shape": list(output.shape),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "error": None
                    }
                    print(f"✅ {model_type}: {output.shape}")
                    
                except Exception as e:
                    results[model_type] = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"❌ {model_type}: {e}")
            
            self.results["tests"]["model_creation"] = results
            return len(models) > 0
            
        except Exception as e:
            self.results["tests"]["model_creation"] = {"error": str(e)}
            print(f"❌ Model creation failed: {e}")
            return False
    
    def run_benchmark(self):
        """Run performance benchmark."""
        print("\n⚡ Running Performance Benchmark")
        print("=" * 40)
        
        try:
            # Run the existing compare script
            os.chdir(project_root)
            exit_code = os.system("python scripts/compare_models.py --benchmark_speed --d_model 256 --n_layers 4")
            
            success = exit_code == 0
            self.results["tests"]["benchmark"] = {"success": success}
            
            if success:
                print("✅ Benchmark completed successfully")
            else:
                print("❌ Benchmark failed")
                
            return success
            
        except Exception as e:
            self.results["tests"]["benchmark"] = {"success": False, "error": str(e)}
            print(f"❌ Benchmark error: {e}")
            return False
    
    def save_results(self):
        """Save test results to file."""
        results_file = "/workspace/results/runpod_test_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def run_all_tests(self):
        """Run complete testing pipeline."""
        print("🚀 MAMBA-KAN GPU TESTING PIPELINE")
        print("=" * 60)
        
        print(f"🖥️ Device: {self.device}")
        if self.device.type == "cuda":
            print(f"🎯 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Run tests in sequence
        tests = [
            ("Dependencies", self.test_dependencies),
            ("Model Creation", self.test_model_creation),
            ("Performance Benchmark", self.run_benchmark)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            passed = test_func()
            all_passed = all_passed and passed
            
            if not passed:
                print(f"⚠️  {test_name} failed, continuing with next test...")
        
        # Save results
        self.save_results()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 TESTING COMPLETE")
        print("=" * 60)
        
        if all_passed:
            print("🎉 All tests passed! Pipeline is fully functional.")
        else:
            print("⚠️  Some tests failed. Check results for details.")
        
        print(f"\n📊 Environment: {self.results['environment']}")
        print(f"📝 Full results: /workspace/results/runpod_test_results.json")
        
        return all_passed


def main():
    """Main testing function."""
    pipeline = GPUTestPipeline()
    success = pipeline.run_all_tests()
    
    if success:
        print("\n🚀 Ready for full experiments!")
    else:
        print("\n🔧 Issues found - check logs for details")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())