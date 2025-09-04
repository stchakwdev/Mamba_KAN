"""Model factory and imports."""

from .mlp_transformer import MLPTransformer
from .kan_transformer import KANTransformer
from .mlp_mamba import MLPMamba
from .kan_mamba import KANMamba

from ..configs.model_configs import (
    MLPTransformerConfig,
    KANTransformerConfig, 
    MLPMambaConfig,
    KANMambaConfig,
    get_model_config
)


def create_model(model_type: str, **config_kwargs):
    """Factory function to create any of the four model variants.
    
    Args:
        model_type: One of "mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"
        **config_kwargs: Configuration overrides
        
    Returns:
        Instantiated model
    """
    # Get appropriate configuration
    config = get_model_config(model_type, **config_kwargs)
    
    # Model mapping
    models = {
        "mlp_transformer": MLPTransformer,
        "kan_transformer": KANTransformer,
        "mlp_mamba": MLPMamba,
        "kan_mamba": KANMamba,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    model_class = models[model_type]
    return model_class(config)


# Make all models easily accessible
__all__ = [
    "MLPTransformer",
    "KANTransformer", 
    "MLPMamba",
    "KANMamba",
    "create_model",
    "MLPTransformerConfig",
    "KANTransformerConfig",
    "MLPMambaConfig", 
    "KANMambaConfig",
    "get_model_config",
]