"""MLP-Mamba model variant."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseModel
from .components.mamba_layers import MambaBlock
from ..configs.model_configs import MLPMambaConfig


class MLPMamba(BaseModel):
    """Mamba model with MLP projections."""
    
    def __init__(self, config: MLPMambaConfig):
        super().__init__(config)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(config, use_kan=False)
            for _ in range(config.n_layers)
        ])
        
        print(f"MLPMamba with {self.get_num_params():,} parameters")
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through MLP-Mamba.
        
        Args:
            idx: Input token indices (batch_size, seq_len)
            targets: Target indices for loss computation (batch_size, seq_len)
            
        Returns:
            If targets provided: loss scalar
            Otherwise: logits (batch_size, seq_len, vocab_size)
        """
        device = idx.device
        b, t = idx.size()
        
        # Token embeddings (no positional encoding for Mamba)
        tok_emb = self.token_embedding(idx)  # (b, t, d_model)
        x = self.dropout(tok_emb)
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        if targets is not None:
            # Training mode - compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return loss
        else:
            # Inference mode - return logits
            logits = self.lm_head(x)
            return logits