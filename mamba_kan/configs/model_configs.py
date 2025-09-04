"""Model-specific configurations for all four variants."""

from dataclasses import dataclass, field
from .base_config import TransformerConfig, MambaConfig, KANConfig


@dataclass
class MLPTransformerConfig(TransformerConfig):
    """Configuration for MLP-Transformer (baseline)."""
    
    model_type: str = "mlp_transformer"
    
    # Standard transformer parameters
    n_heads: int = 8
    d_ff: int = 2048


@dataclass  
class KANTransformerConfig(TransformerConfig):
    """Configuration for KAN-Transformer."""
    
    model_type: str = "kan_transformer"
    
    # Reduced dimensions to match parameter count
    n_heads: int = 8
    d_ff: int = 1400  # ~30% reduction for parameter matching
    
    # KAN-specific parameters
    kan_config: KANConfig = field(default_factory=KANConfig)


@dataclass
class MLPMambaConfig(MambaConfig):
    """Configuration for MLP-Mamba."""
    
    model_type: str = "mlp_mamba"
    
    # Standard Mamba parameters
    d_state: int = 16
    expand: float = 2.0


@dataclass
class KANMambaConfig(MambaConfig):
    """Configuration for KAN-Mamba."""
    
    model_type: str = "kan_mamba"
    
    # Adjusted parameters for parameter matching
    d_state: int = 16
    expand: float = 1.4  # Reduced expand ratio
    
    # KAN-specific parameters
    kan_config: KANConfig = field(default_factory=KANConfig)


# Factory function to get the right config
def get_model_config(model_type: str, **kwargs):
    """Factory function to create model configurations."""
    
    configs = {
        "mlp_transformer": MLPTransformerConfig,
        "kan_transformer": KANTransformerConfig,
        "mlp_mamba": MLPMambaConfig,
        "kan_mamba": KANMambaConfig,
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(configs.keys())}")
    
    config_class = configs[model_type]
    
    # Override defaults with provided kwargs
    return config_class(**kwargs)