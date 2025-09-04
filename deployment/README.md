# RunPod Deployment for Mamba-KAN

This directory contains all scripts and configurations needed to deploy and test the Mamba-KAN project on RunPod GPU instances.

## Quick Start

### Option 1: Automated Deployment
```bash
# From project root
./deployment/quick_deploy.sh
```

### Option 2: Manual RunPod Setup
1. Go to [RunPod GPU Instance](https://www.runpod.io/gpu-instance)
2. Choose RTX 3090 or RTX 4090 
3. Template: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`
4. Volume: 20-30GB
5. Expose ports: `8888,22`
6. Once started:
```bash
git clone https://github.com/stchakwdev/Mamba_KAN.git
cd Mamba_KAN  
bash deployment/setup_environment.sh
python deployment/run_tests.py
```

## Files

- **`deploy_runpod.py`** - Main deployment script using RunPod API
- **`setup_environment.sh`** - Environment setup for GPU instances  
- **`run_tests.py`** - Comprehensive testing pipeline
- **`monitor_training.py`** - GPU monitoring utilities
- **`quick_deploy.sh`** - One-command deployment
- **`requirements_gpu.txt`** - GPU-specific dependencies
- **`Dockerfile`** - Docker configuration for custom containers

## Testing Pipeline

The `run_tests.py` script performs:
1. ✅ Dependency verification (PyTorch, PyKAN, Mamba-SSM)
2. ✅ Model creation (all 4 variants)
3. ✅ Forward pass validation  
4. ✅ Performance benchmarking
5. ✅ Results logging to `/workspace/results/`

## Expected Results

With CUDA environment, you should see:
- ✅ All dependencies install successfully (including mamba-ssm)
- ✅ All 4 model variants create without errors
- ✅ Parameter counts match within 5% tolerance
- ✅ GPU utilization during benchmarks

## Troubleshooting

If mamba-ssm still fails:
- Check CUDA version: `nvcc --version`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Manual install: `pip install mamba-ssm --no-cache-dir --force-reinstall`