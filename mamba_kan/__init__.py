"""Mamba-KAN neural architecture comparison package."""

__version__ = "0.1.0"
__author__ = "Mamba-KAN Research Team"

# Core model exports (Mamba models optional)
from .models import (
    MLPTransformer,
    KANTransformer,
    create_model,
)

# Try to import Mamba models (optional)
try:
    from .models import MLPMamba, KANMamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    MLPMamba = None
    KANMamba = None

# Configuration exports
from .configs.model_configs import (
    MLPTransformerConfig,
    KANTransformerConfig,
    get_model_config,
)

# Try to import Mamba configs (optional)
try:
    from .configs.model_configs import MLPMambaConfig, KANMambaConfig
except ImportError:
    MLPMambaConfig = None
    KANMambaConfig = None

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