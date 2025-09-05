"""Model factory and imports."""

from .mlp_transformer import MLPTransformer
from .kan_transformer import KANTransformer

from ..configs.model_configs import (
    MLPTransformerConfig,
    KANTransformerConfig, 
    get_model_config
)

# Try to import Mamba models (optional)
try:
    from .mlp_mamba import MLPMamba
    from .kan_mamba import KANMamba
    from ..configs.model_configs import MLPMambaConfig, KANMambaConfig
    MAMBA_AVAILABLE = True
except ImportError:
    MLPMamba = None
    KANMamba = None
    MLPMambaConfig = None
    KANMambaConfig = None
    MAMBA_AVAILABLE = False


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
    }
    
    # Add Mamba models if available
    if MAMBA_AVAILABLE:
        models.update({
            "mlp_mamba": MLPMamba,
            "kan_mamba": KANMamba,
        })
    
    if model_type not in models:
        available_models = list(models.keys())
        if not MAMBA_AVAILABLE and model_type in ["mlp_mamba", "kan_mamba"]:
            raise ValueError(f"Mamba models not available due to missing dependencies. Available: {available_models}")
        raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
    
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