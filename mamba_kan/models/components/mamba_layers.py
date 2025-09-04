"""Mamba layer components with optional KAN integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from mamba_ssm import Mamba

from ...configs.base_config import MambaConfig, KANConfig
from .kan_layers import KANProjection


class MambaBlock(nn.Module):
    """Mamba block with configurable projection layers."""
    
    def __init__(self, config: MambaConfig, use_kan: bool = False, kan_config: Optional[KANConfig] = None):
        super().__init__()
        self.config = config
        self.use_kan = use_kan
        
        # Layer norm
        self.norm = nn.RMSNorm(config.d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(config.d_model)
        
        # Mamba core with appropriate projections
        if use_kan:
            if kan_config is None:
                raise ValueError("KAN config required when use_kan=True")
            self.mamba = KANMambaCore(config, kan_config)
        else:
            self.mamba = MLPMambaCore(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Residual connection with normalization
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + x


class MLPMambaCore(nn.Module):
    """Mamba core with standard MLP projections."""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_inner = int(config.expand * config.d_model)
        
        # Input projection
        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=config.bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
            bias=config.conv_bias,
        )
        self.activation = nn.SiLU()
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, 
                               config.d_state * 2 + (config.dt_rank if isinstance(config.dt_rank, int) else self.d_inner),
                               bias=False)
        self.dt_proj = nn.Linear(config.dt_rank if isinstance(config.dt_rank, int) else self.d_inner, 
                                self.d_inner, 
                                bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=config.bias)
        
        # Initialize Mamba SSM
        self.ssm = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using official Mamba implementation."""
        return self.ssm(x)


class KANMambaCore(nn.Module):
    """Mamba core with KAN projections."""
    
    def __init__(self, config: MambaConfig, kan_config: KANConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_inner = int(config.expand * config.d_model)
        self.d_state = config.d_state
        
        # KAN-based projections
        self.in_proj = KANProjection(config.d_model, self.d_inner * 2, kan_config)
        
        # Convolution (unchanged)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner, 
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
            bias=config.conv_bias,
        )
        self.activation = nn.SiLU()
        
        # SSM parameters with KAN
        dt_rank = config.dt_rank if isinstance(config.dt_rank, int) else self.d_inner
        self.x_proj = KANProjection(self.d_inner, config.d_state * 2 + dt_rank, kan_config)
        self.dt_proj = KANProjection(dt_rank, self.d_inner, kan_config)
        
        # Output projection with KAN
        self.out_proj = KANProjection(self.d_inner, config.d_model, kan_config)
        
        # SSM parameters (learnable)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, config.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through KAN-Mamba.
        
        This is a simplified implementation - for production use,
        the official Mamba CUDA kernels would be preferred.
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, res = x_and_res.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seq_len]  # Trim padding
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = self.activation(x)
        
        # SSM parameters
        x_ssm = self.x_proj(x)  # (B, L, d_state * 2 + dt_rank)
        dt, B, C = x_ssm.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        
        # Delta projection
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)
        
        # Simplified SSM (not optimized)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize
        dt = dt.unsqueeze(-1)  # (B, L, d_inner, 1)
        A_discrete = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt)  # (B, L, d_inner, d_state)
        B_discrete = dt * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # SSM scan (simplified - not optimized)
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            h = A_discrete[:, t] * h + B_discrete[:, t] * x[:, t:t+1].unsqueeze(-1)
            y_t = (h * C[:, t:t+1].unsqueeze(1)).sum(-1)  # (B, d_inner)
            outputs.append(y_t)
        
        x = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        # Skip connection with residual
        x = x * self.activation(res)
        
        # Output projection
        x = self.out_proj(x)
        
        return x


