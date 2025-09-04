#!/bin/bash
"""
Quick deployment script for Mamba-KAN on RunPod.
One-command deployment and testing.
"""

set -e  # Exit on error

echo "🚀 MAMBA-KAN RUNPOD QUICK DEPLOY"
echo "=" * 50

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create .env file with RUNPOD_API_KEY"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "CLAUDE.md" ]; then
    echo "❌ Not in project root directory!"
    echo "Please run from Mamba_KAN project root"
    exit 1
fi

echo "✅ Environment checks passed"

# Install RunPod Python client locally if not available
echo "📦 Installing RunPod client..."
pip install runpod python-dotenv requests --quiet

echo "🚀 Starting deployment..."
python deployment/deploy_runpod.py

echo "✅ Deployment script completed!"

# Alternative manual deployment instructions
echo ""
echo "🔧 ALTERNATIVE: Manual RunPod Setup"
echo "=================================="
echo "1. Go to: https://www.runpod.io/gpu-instance"
echo "2. Choose: RTX 3090 or RTX 4090"
echo "3. Template: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
echo "4. Volume: 20GB"
echo "5. Expose ports: 8888, 22"
echo "6. Once started, run these commands:"
echo ""
echo "   git clone https://github.com/stchakwdev/Mamba_KAN.git"
echo "   cd Mamba_KAN"
echo "   bash deployment/setup_environment.sh"
echo "   python deployment/run_tests.py"
echo ""