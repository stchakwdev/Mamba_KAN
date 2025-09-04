"""Base model class for all architecture variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..configs.base_config import BaseConfig
from ..utils.parameter_counter import count_parameters


class BaseModel(nn.Module, ABC):
    """Abstract base class for all model variants."""
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Final layer norm and head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters in the model."""
        n_params = count_parameters(self)
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params
    
    @abstractmethod
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        pass
    
    def configure_optimizers(self):
        """Configure optimizers for training."""
        # Separate parameters that should and shouldn't have weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        # Validate that we've covered all parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both decay/no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters not in either set: {param_dict.keys() - union_params}"
        
        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], 
             "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], 
             "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate)
        return optimizer
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100, 
                 temperature: float = 1.0, do_sample: bool = True) -> torch.Tensor:
        """Generate new tokens from the model."""
        for _ in range(max_new_tokens):
            # Crop context if it gets too long
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Sample or take most likely
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx