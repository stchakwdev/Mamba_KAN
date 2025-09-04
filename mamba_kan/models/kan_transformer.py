"""KAN-Transformer model variant."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseModel
from .components.transformer_layers import TransformerBlock
from ..configs.model_configs import KANTransformerConfig


class KANTransformer(BaseModel):
    """Transformer model with KAN feedforward blocks."""
    
    def __init__(self, config: KANTransformerConfig):
        super().__init__(config)
        
        # Transformer blocks with KAN feedforward
        self.blocks = nn.ModuleList([
            TransformerBlock(config, use_kan=True, kan_config=config.kan_config)
            for _ in range(config.n_layers)
        ])
        
        print(f"KANTransformer with {self.get_num_params():,} parameters")
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through KAN-Transformer.
        
        Args:
            idx: Input token indices (batch_size, seq_len)
            targets: Target indices for loss computation (batch_size, seq_len)
            
        Returns:
            If targets provided: loss scalar
            Otherwise: logits (batch_size, seq_len, vocab_size)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Sequence length {t} > max length {self.config.max_seq_len}"
        
        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        
        tok_emb = self.token_embedding(idx)  # (b, t, d_model)
        pos_emb = self.position_embedding(pos)  # (1, t, d_model)
        x = self.dropout(tok_emb + pos_emb)
        
        # Apply transformer blocks with KAN feedforward
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