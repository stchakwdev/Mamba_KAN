"""KAN layer implementations for integration with Transformer and Mamba."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from kan import KAN

from ...configs.base_config import KANConfig


class KANBlock(nn.Module):
    """KAN block that can replace MLP feedforward layers."""
    
    def __init__(self, d_model: int, d_ff: int, kan_config: KANConfig):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Create KAN with specified architecture
        # [input_dim, hidden_dim, output_dim]
        # Note: KAN may create checkpoint directories - this is normal behavior
        self.kan = KAN(
            width=[d_model, d_ff, d_model],
            grid=kan_config.grid_size,
            k=kan_config.spline_order,
            noise_scale=kan_config.noise_scale,
            save_act=False,  # Disable activation saving for efficiency
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through KAN block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # KAN expects 2D input: (batch_size * seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        # Apply KAN
        output_flat = self.kan(x_flat)
        
        # Reshape back to sequence format
        output = output_flat.view(batch_size, seq_len, d_model)
        
        return output


class KANProjection(nn.Module):
    """Single KAN layer for projections (used in Mamba)."""
    
    def __init__(self, input_dim: int, output_dim: int, kan_config: KANConfig):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Single layer KAN
        self.kan = KAN(
            width=[input_dim, output_dim],
            grid=kan_config.grid_size,
            k=kan_config.spline_order,
            noise_scale=kan_config.noise_scale,
            save_act=False,  # Disable activation saving for efficiency
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through KAN projection.
        
        Args:
            x: Input tensor of any shape (..., input_dim)
            
        Returns:
            Output tensor of shape (..., output_dim)
        """
        original_shape = x.shape
        
        # Flatten to 2D for KAN
        x_flat = x.view(-1, self.input_dim)
        
        # Apply KAN
        output_flat = self.kan(x_flat)
        
        # Reshape back to original shape (except last dim)
        output_shape = original_shape[:-1] + (self.output_dim,)
        output = output_flat.view(output_shape)
        
        return output


class MultiKANProjection(nn.Module):
    """Multiple parallel KAN projections (for efficiency)."""
    
    def __init__(self, input_dim: int, output_dims: List[int], kan_config: KANConfig):
        super().__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        
        # Create separate KAN for each output
        self.kans = nn.ModuleList([
            KAN(
                width=[input_dim, output_dim],
                grid=kan_config.grid_size,
                k=kan_config.spline_order,
                noise_scale=kan_config.noise_scale,
                save_act=False,  # Disable activation saving for efficiency
            ) for output_dim in output_dims
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through multiple KAN projections.
        
        Args:
            x: Input tensor (..., input_dim)
            
        Returns:
            List of output tensors with shapes (..., output_dims[i])
        """
        original_shape = x.shape[:-1]
        x_flat = x.view(-1, self.input_dim)
        
        outputs = []
        for kan, output_dim in zip(self.kans, self.output_dims):
            output_flat = kan(x_flat)
            output_shape = original_shape + (output_dim,)
            outputs.append(output_flat.view(output_shape))
            
        return outputs


class MLPBlock(nn.Module):
    """Standard MLP block for comparison."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP block."""
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x