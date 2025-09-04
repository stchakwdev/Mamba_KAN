"""Mamba-KAN neural architecture comparison package."""

__version__ = "0.1.0"
__author__ = "Mamba-KAN Research Team"

# Core model exports
from .models import (
    MLPTransformer,
    KANTransformer,
    MLPMamba, 
    KANMamba,
    create_model,
)

# Configuration exports
from .configs.model_configs import (
    MLPTransformerConfig,
    KANTransformerConfig,
    MLPMambaConfig,
    KANMambaConfig,
    get_model_config,
)

# Utility exports
from .utils.parameter_counter import (
    count_parameters,
    compare_parameter_counts,
    validate_parameter_matching,
)

__all__ = [
    # Models
    "MLPTransformer",
    "KANTransformer", 
    "MLPMamba",
    "KANMamba",
    "create_model",
    
    # Configs
    "MLPTransformerConfig",
    "KANTransformerConfig",
    "MLPMambaConfig",
    "KANMambaConfig", 
    "get_model_config",
    
    # Utils
    "count_parameters",
    "compare_parameter_counts",
    "validate_parameter_matching",
]