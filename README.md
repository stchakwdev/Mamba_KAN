# Mamba-KAN Neural Architecture Comparison

A comprehensive research project comparing neural network architectures through a 2×2 factorial experimental design:

- **Feedforward Components**: Multi-Layer Perceptron (MLP) vs. Kolmogorov-Arnold Network (KAN)  
- **Sequence Modeling**: Transformer vs. Mamba (State Space Model)

## Model Variants

This project implements and compares four model architectures:

1. **MLP-Transformer** - Standard transformer with MLP feedforward (baseline)
2. **KAN-Transformer** - Transformer with KAN feedforward blocks
3. **MLP-Mamba** - Mamba with standard MLP projections
4. **KAN-Mamba** - Mamba with KAN-based projections

## Quick Start

### Installation

```bash
# Create environment
conda create -n mamba_kan python=3.9
conda activate mamba_kan

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from mamba_kan.models import create_model
from mamba_kan.utils.parameter_counter import compare_parameter_counts

# Create any of the four model variants
model = create_model("mlp_transformer")  # or kan_transformer, mlp_mamba, kan_mamba

# Compare all models
models = {
    name: create_model(name) 
    for name in ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]
}
compare_parameter_counts(models)
```

### Command Line Interface

```bash
# Test a specific model variant
python scripts/train.py --model_type mlp_transformer

# Compare parameter counts of all variants
python scripts/train.py --model_type mlp_transformer --compare_all

# Customize model dimensions
python scripts/train.py --model_type kan_transformer --d_model 768 --n_layers 12
```

## Project Structure

```
mamba_kan/
├── models/                     # Model implementations
│   ├── base.py                # Base model class
│   ├── mlp_transformer.py     # MLP-Transformer (baseline)
│   ├── kan_transformer.py     # KAN-Transformer
│   ├── mlp_mamba.py          # MLP-Mamba
│   ├── kan_mamba.py          # KAN-Mamba
│   └── components/           # Reusable components
│       ├── kan_layers.py     # KAN building blocks
│       ├── mamba_layers.py   # Mamba building blocks
│       └── transformer_layers.py # Transformer blocks
├── configs/                   # Configuration classes
│   ├── base_config.py        # Base configurations
│   └── model_configs.py      # Model-specific configs
├── utils/                    # Utilities
│   └── parameter_counter.py  # Parameter analysis tools
├── scripts/                  # Entry point scripts
│   └── train.py             # Training and testing script
└── tests/                    # Test suite
```

## Research Design

### Factorial Experimental Design

This project uses a 2×2 factorial design to isolate the effects of:

- **Feedforward Architecture** (MLP vs KAN)
- **Sequence Modeling** (Transformer vs Mamba)  
- **Interactions** between feedforward and sequence modeling components

### Parameter Matching

All four model variants are designed to have similar parameter counts (±5% tolerance) to ensure fair comparison:

- **KAN variants**: Use reduced hidden dimensions (~30% smaller) to account for spline coefficient overhead
- **Mamba variants**: Adjust expand ratios to match transformer parameter counts

### Evaluation Tasks

The models will be evaluated on diverse tasks to test different capabilities:

1. **Mathematical/Symbolic Reasoning** (expected KAN advantage)
2. **Long Sequence Modeling** (expected Mamba advantage)
3. **General Language Understanding** (neutral comparison)
4. **Code Understanding** (mixed expectations)

## Key Dependencies

- **PyTorch**: Core deep learning framework
- **PyKAN**: Official KAN implementation with B-spline basis functions
- **mamba-ssm**: Official Mamba implementation with optimized CUDA kernels
- **PyTorch Lightning**: For standardized training and multi-GPU support

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 8GB+ VRAM, 16GB+ RAM
- **Recommended**: RTX 3080+ or A100, 32GB+ RAM
- **For large-scale experiments**: Multi-GPU setup recommended

## Contributing

This is a research project comparing neural architectures. Contributions welcome for:

- Additional evaluation tasks
- Architecture improvements
- Efficiency optimizations
- Analysis and visualization tools

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mamba_kan_comparison,
    title={Mamba-KAN: A Factorial Comparison of Neural Network Architectures},
    author={[Samuel T Chakwera]},
    year={2025},
    url={https://github.com/[your-username]/Mamba_KAN}
}
```

## License

MIT License - see LICENSE file for details.

## References

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [PyKAN Implementation](https://github.com/KindXiaoming/pykan)
- [Official Mamba Implementation](https://github.com/state-spaces/mamba)
