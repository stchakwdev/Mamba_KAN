# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a comprehensive research project comparing neural network architectures through a 2×2 factorial experimental design:

- **Feedforward Components**: Multi-Layer Perceptron (MLP) vs. Kolmogorov-Arnold Network (KAN)  
- **Sequence Modeling**: Transformer vs. Mamba (State Space Model)

The goal is to implement and empirically evaluate four model variants:
1. **MLP-Transformer** (baseline)
2. **KAN-Transformer** 
3. **MLP-Mamba**
4. **KAN-Mamba**

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n kan_mamba_comparison python=3.9
conda activate kan_mamba_comparison

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning wandb transformers datasets evaluate
pip install numpy pandas matplotlib seaborn scikit-learn

# Install architecture-specific libraries
pip install pykan                    # KAN implementation
pip install mamba-ssm               # Mamba state space model
pip install git+https://github.com/Blealtan/efficient-kan.git  # Efficient KAN
```

### Testing and Validation
```bash
# Verify installations
python -c "import torch, pytorch_lightning as pl; from pykan import KAN; from mamba_ssm import Mamba; print('All dependencies installed successfully')"

# Run parameter counting utilities
python utils/count_parameters.py

# Test model implementations
python test/test_model_variants.py

# Validate parameter matching (ensure all variants have similar parameter counts)
python scripts/validate_parameter_matching.py
```

## Architecture Design

### Core Experimental Framework

The project uses a **modular architecture** where feedforward components (MLP/KAN) and sequence modeling components (Transformer/Mamba) can be mixed and matched:

```python
class UnifiedModel(nn.Module):
    def __init__(self, 
                 sequence_model_type: str,  # "transformer" or "mamba"
                 feedforward_type: str,     # "mlp" or "kan"
                 **kwargs):
        self.sequence_model = self._build_sequence_model(sequence_model_type)
        self.feedforward = self._build_feedforward(feedforward_type)
```

### Key Integration Points

- **Transformer + KAN**: Replace standard FFN blocks (2-layer MLP) with KAN layers
- **Mamba + KAN**: Replace linear input/output projections with KAN functions

### Parameter Matching Strategy

All model variants must maintain similar parameter counts (±5% tolerance):
- **KAN variants**: Reduce hidden dimensions by ~30% to account for spline coefficient overhead
- **Mamba variants**: Adjust expand ratio and state size to match baseline

## Evaluation Protocol

### Statistical Analysis Framework
- **Primary Analysis**: Factorial ANOVA to test main effects and interactions
- **Performance Metrics**: Task-specific (accuracy, perplexity, exact match)
- **Efficiency Metrics**: FLOPs, memory usage, throughput
- **Multiple Seeds**: Run experiments with different random seeds for statistical validity

### Task Categories for Evaluation
1. **Mathematical/Symbolic Reasoning**: GSM8K, MATH, symbolic regression (expected KAN strength)
2. **Long Sequence Modeling**: Long Range Arena, BookCorpus (expected Mamba strength)  
3. **General Language Understanding**: SuperGLUE, HellaSwag (neutral comparison)
4. **Code Understanding**: HumanEval, MBPP (mixed expectations)

## Implementation Notes

### Critical Dependencies
- **PyKAN**: Official KAN implementation with B-spline basis functions
- **mamba-ssm**: Official Mamba implementation with CUDA kernels
- **PyTorch Lightning**: For standardized training loops and multi-GPU support

### Architecture-Specific Considerations
- **KAN models**: Higher memory usage due to spline functions, requires careful dimension tuning
- **Mamba models**: Linear scaling advantage for long sequences, different parameter distribution than Transformers
- **Parameter counting**: Use `count_parameters()` utility to ensure fair comparison across variants

### Training Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3080+ recommended)
- **Memory**: 16GB+ RAM, 8GB+ VRAM minimum
- **Reproducibility**: Fixed seeds, consistent preprocessing, identical training schedules

## Project Structure Context

- **Docs/**: Contains comprehensive research plans and implementation guides
- **No source code yet**: This is a planning/documentation phase project
- **Expected implementation**: Modular Python codebase with PyTorch/Lightning
- **Evaluation focus**: Empirical comparison with rigorous statistical analysis