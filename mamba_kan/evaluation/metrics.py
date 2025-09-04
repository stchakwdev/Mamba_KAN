"""Evaluation metrics for model comparison."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> float:
    """Compute perplexity from logits and targets."""
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        targets.view(-1), 
        ignore_index=ignore_index
    )
    return torch.exp(loss).item()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> float:
    """Compute token-level accuracy."""
    predictions = torch.argmax(logits, dim=-1)
    
    # Mask out ignored tokens
    valid_mask = (targets != ignore_index)
    correct = (predictions == targets) & valid_mask
    
    accuracy = correct.sum().float() / valid_mask.sum().float()
    return accuracy.item()


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5, ignore_index: int = -1) -> float:
    """Compute top-k accuracy."""
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    
    # Expand targets to match top_k_preds shape
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    
    # Check if true target is in top-k predictions
    correct = (top_k_preds == targets_expanded).any(dim=-1)
    
    # Mask out ignored tokens
    valid_mask = (targets != ignore_index)
    correct_masked = correct & valid_mask
    
    accuracy = correct_masked.sum().float() / valid_mask.sum().float()
    return accuracy.item()


class MetricsTracker:
    """Track and aggregate metrics during evaluation."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.step_count = 0
    
    def update(self, batch_metrics: Dict[str, float]):
        """Update metrics with batch results."""
        for name, value in batch_metrics.items():
            self.metrics[name].append(value)
        self.step_count += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics across all batches."""
        return {
            name: np.mean(values) 
            for name, values in self.metrics.items()
        }
    
    def get_std(self) -> Dict[str, float]:
        """Get standard deviation of metrics."""
        return {
            name: np.std(values)
            for name, values in self.metrics.items()
        }
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.step_count = 0
    
    def summary(self) -> str:
        """Get formatted summary of metrics."""
        averages = self.get_averages()
        stds = self.get_std()
        
        lines = [f"Metrics over {self.step_count} steps:"]
        for name in sorted(averages.keys()):
            lines.append(f"  {name}: {averages[name]:.4f} ± {stds[name]:.4f}")
        
        return "\n".join(lines)


def evaluate_model_on_batch(model: torch.nn.Module, batch_input: torch.Tensor, 
                           batch_targets: torch.Tensor) -> Dict[str, float]:
    """Evaluate model on a single batch and return metrics.
    
    Args:
        model: Model to evaluate
        batch_input: Input token indices (batch_size, seq_len)
        batch_targets: Target token indices (batch_size, seq_len)
        
    Returns:
        Dictionary of computed metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Get logits
        logits = model(batch_input)
        
        # Compute metrics
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch_targets.view(-1),
            ignore_index=-1
        )
        
        perplexity = torch.exp(loss)
        accuracy = compute_accuracy(logits, batch_targets)
        top5_accuracy = compute_top_k_accuracy(logits, batch_targets, k=5)
        
        return {
            "loss": loss.item(),
            "perplexity": perplexity.item(),
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
        }


class ModelComparator:
    """Compare multiple models on the same task."""
    
    def __init__(self, models: Dict[str, torch.nn.Module]):
        self.models = models
        self.trackers = {name: MetricsTracker() for name in models.keys()}
    
    def evaluate_batch(self, batch_input: torch.Tensor, batch_targets: torch.Tensor):
        """Evaluate all models on a batch."""
        for name, model in self.models.items():
            metrics = evaluate_model_on_batch(model, batch_input, batch_targets)
            self.trackers[name].update(metrics)
    
    def get_comparison_summary(self) -> str:
        """Get formatted comparison of all models."""
        lines = ["Model Comparison Results:"]
        lines.append("=" * 50)
        
        for name, tracker in self.trackers.items():
            lines.append(f"\n{name}:")
            averages = tracker.get_averages()
            for metric_name, value in averages.items():
                lines.append(f"  {metric_name}: {value:.4f}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all trackers."""
        for tracker in self.trackers.values():
            tracker.reset()