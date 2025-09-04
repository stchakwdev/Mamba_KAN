"""Utilities for counting and matching model parameters."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import OrderedDict


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        only_trainable: Whether to count only trainable parameters
        
    Returns:
        Total number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """Get detailed breakdown of parameters by module.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping module names to parameter counts
    """
    breakdown = OrderedDict()
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if param_count > 0:
                breakdown[name] = param_count
                
    return breakdown


def compare_parameter_counts(models: Dict[str, nn.Module]) -> None:
    """Compare parameter counts across multiple models.
    
    Args:
        models: Dictionary mapping model names to models
    """
    print("\n" + "="*60)
    print("MODEL PARAMETER COMPARISON")
    print("="*60)
    
    counts = {}
    for name, model in models.items():
        counts[name] = count_parameters(model)
        
    # Find baseline (MLP-Transformer)
    baseline_name = "mlp_transformer"
    if baseline_name in counts:
        baseline_count = counts[baseline_name]
        print(f"Baseline ({baseline_name}): {baseline_count:,} parameters\n")
        
        for name, count in counts.items():
            if name != baseline_name:
                diff_abs = count - baseline_count
                diff_pct = ((count - baseline_count) / baseline_count) * 100
                print(f"{name:<20}: {count:,} parameters "
                      f"({diff_pct:+.2f}%, {diff_abs:+,})")
    else:
        for name, count in counts.items():
            print(f"{name:<20}: {count:,} parameters")
    
    print("="*60)


def validate_parameter_matching(models: Dict[str, nn.Module], 
                              tolerance_pct: float = 5.0) -> bool:
    """Validate that all models have similar parameter counts.
    
    Args:
        models: Dictionary mapping model names to models
        tolerance_pct: Allowed percentage difference from baseline
        
    Returns:
        True if all models are within tolerance
    """
    counts = {name: count_parameters(model) for name, model in models.items()}
    
    # Use first model as baseline
    baseline_count = list(counts.values())[0]
    
    all_valid = True
    for name, count in counts.items():
        diff_pct = abs((count - baseline_count) / baseline_count) * 100
        if diff_pct > tolerance_pct:
            print(f"WARNING: {name} parameter count differs by {diff_pct:.2f}% "
                  f"(>{tolerance_pct}% tolerance)")
            all_valid = False
            
    return all_valid


def estimate_memory_usage(model: nn.Module, batch_size: int = 32, 
                         seq_len: int = 1024, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """Estimate memory usage for a model.
    
    Args:
        model: PyTorch model
        batch_size: Batch size for estimation
        seq_len: Sequence length
        dtype: Data type for tensors
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Bytes per element based on dtype
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }[dtype]
    
    # Model parameters
    param_memory = count_parameters(model) * bytes_per_element
    
    # Input tensor
    input_memory = batch_size * seq_len * model.config.d_model * bytes_per_element
    
    # Rough estimate of activations (varies by architecture)
    # This is a conservative estimate
    activation_memory = input_memory * model.config.n_layers * 4
    
    # Gradients (same size as parameters during training)
    gradient_memory = param_memory
    
    total_memory = param_memory + input_memory + activation_memory + gradient_memory
    
    return {
        "parameters_mb": param_memory / (1024 * 1024),
        "input_mb": input_memory / (1024 * 1024), 
        "activations_mb": activation_memory / (1024 * 1024),
        "gradients_mb": gradient_memory / (1024 * 1024),
        "total_mb": total_memory / (1024 * 1024),
        "total_gb": total_memory / (1024 * 1024 * 1024),
    }


def get_flop_estimate(model: nn.Module, batch_size: int = 32, 
                     seq_len: int = 1024) -> int:
    """Rough FLOP estimate for a forward pass.
    
    This is a simplified estimate - for precise measurements use 
    profiling tools like pytorch-flop-counter.
    
    Args:
        model: PyTorch model
        batch_size: Batch size
        seq_len: Sequence length
        
    Returns:
        Estimated FLOPs
    """
    param_count = count_parameters(model)
    
    # Very rough estimate: 2 * params * batch_size * seq_len
    # This assumes each parameter is used once per token per batch
    estimated_flops = 2 * param_count * batch_size * seq_len
    
    return estimated_flops