"""Base configuration classes for all models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    """Base configuration for all model variants."""
    
    # Model architecture
    d_model: int = 512
    n_layers: int = 6
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dropout: float = 0.1
    
    # Training parameters  
    learning_rate: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Evaluation
    eval_interval: int = 1000
    eval_iters: int = 200
    
    # Hardware
    device: str = "cuda"
    compile: bool = True
    
    # Reproducibility
    seed: int = 42


@dataclass
class TransformerConfig(BaseConfig):
    """Configuration for Transformer-based models."""
    
    n_heads: int = 8
    d_ff: int = 2048
    layer_norm_eps: float = 1e-5
    use_bias: bool = True


@dataclass
class MambaConfig(BaseConfig):
    """Configuration for Mamba-based models."""
    
    d_state: int = 16
    d_conv: int = 4
    expand: float = 2.0
    dt_rank: str = "auto"  # Can be int or "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False


@dataclass
class KANConfig:
    """Configuration for KAN layers."""
    
    grid_size: int = 5
    spline_order: int = 3
    noise_scale: float = 0.1
    base_activation: str = "silu"  # Base activation function
    grid_eps: float = 0.02
    grid_range: list = None  # Will default to [-1, 1]
    
    def __post_init__(self):
        if self.grid_range is None:
            self.grid_range = [-1, 1]