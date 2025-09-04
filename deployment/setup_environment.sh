#!/bin/bash
"""
RunPod environment setup script for Mamba-KAN project.
This script installs all dependencies with CUDA support.
"""

set -e  # Exit on any error

echo "🚀 Setting up Mamba-KAN environment on RunPod GPU"
echo "=" * 60

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
nvidia-smi
nvcc --version

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y git build-essential cmake ninja-build

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Upgrade pip and install build tools
echo "🛠️ Installing build tools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA (force reinstall for compatibility)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --force-reinstall

# Test PyTorch CUDA
echo "🧪 Testing PyTorch CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name()}')
"

# Install core ML libraries
echo "📚 Installing ML libraries..."
pip install \
    pytorch-lightning>=2.0.0 \
    transformers>=4.30.0 \
    datasets>=2.14.0 \
    evaluate>=0.4.0 \
    wandb>=0.15.0 \
    numpy pandas scipy scikit-learn \
    matplotlib seaborn \
    jupyter jupyterlab ipywidgets

# Install KAN dependencies
echo "🔢 Installing KAN dependencies..."
pip install pykan>=0.2.0

# Install efficient KAN from GitHub
echo "⚡ Installing Efficient KAN..."
pip install git+https://github.com/Blealtan/efficient-kan.git

# Install Mamba SSM with CUDA support (this should work now!)
echo "🐍 Installing Mamba SSM with CUDA..."
pip install mamba-ssm>=1.2.0

# Install development and utility packages
echo "🔧 Installing development tools..."
pip install \
    pytest black isort \
    python-dotenv requests \
    tqdm rich \
    tensorboard

# Test all critical imports
echo "🧪 Testing critical imports..."
python -c "
print('Testing imports...')
import torch
print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')

try:
    import pykan
    from kan import KAN
    print('✅ PyKAN available')
except ImportError as e:
    print(f'❌ PyKAN failed: {e}')

try:
    import mamba_ssm
    from mamba_ssm import Mamba
    print('✅ Mamba-SSM available')
except ImportError as e:
    print(f'❌ Mamba-SSM failed: {e}')

try:
    import pytorch_lightning as pl
    print(f'✅ PyTorch Lightning {pl.__version__}')
except ImportError as e:
    print(f'❌ PyTorch Lightning failed: {e}')

print('Import testing complete!')
"

# Create results directory
mkdir -p /workspace/results

echo "✅ Environment setup complete!"
echo "📍 Project will be available at: /workspace/Mamba_KAN"
echo "🚀 Ready to run tests and benchmarks!"