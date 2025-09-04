"""Transformer layer components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from ...configs.base_config import TransformerConfig, KANConfig
from .kan_layers import KANBlock, MLPBlock


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        # Combined projection for Q, K, V
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.use_bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                                     .view(1, 1, config.max_seq_len, config.max_seq_len))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-head attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        B, T, C = x.size()  # batch_size, seq_len, d_model
        
        # Calculate Q, K, V
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        
        # Attention
        if self.flash:
            # Use flash attention if available
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=True
            )
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, n_heads, T, d_head)
        
        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class TransformerBlock(nn.Module):
    """Transformer block with configurable feedforward layer."""
    
    def __init__(self, config: TransformerConfig, use_kan: bool = False, kan_config: Optional[KANConfig] = None):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        
        # Choose feedforward layer type
        if use_kan:
            if kan_config is None:
                raise ValueError("KAN config required when use_kan=True")
            self.mlp = KANBlock(config.d_model, config.d_ff, kan_config)
        else:
            self.mlp = MLPBlock(config.d_model, config.d_ff, config.dropout, config.use_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Attention block with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # Feedforward block with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x