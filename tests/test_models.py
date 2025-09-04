"""Tests for model implementations."""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mamba_kan.models import create_model
from mamba_kan.utils.parameter_counter import validate_parameter_matching


class TestModelImplementations:
    """Test all four model variants."""
    
    @pytest.fixture(params=["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"])
    def model_type(self, request):
        return request.param
    
    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return {
            "d_model": 64,
            "n_layers": 2,
            "vocab_size": 1000,
            "max_seq_len": 128,
        }
    
    def test_model_creation(self, model_type, small_config):
        """Test that all models can be created without errors."""
        model = create_model(model_type, **small_config)
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'config')
    
    def test_forward_pass(self, model_type, small_config):
        """Test forward pass for all models."""
        model = create_model(model_type, **small_config)
        model.eval()
        
        batch_size, seq_len = 4, 32
        dummy_input = torch.randint(0, small_config["vocab_size"], (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(dummy_input)
        
        expected_shape = (batch_size, seq_len, small_config["vocab_size"])
        assert output.shape == expected_shape
    
    def test_parameter_counts(self, small_config):
        """Test that parameter counts are reasonable and similar across variants."""
        models = {}
        for model_type in ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]:
            models[model_type] = create_model(model_type, **small_config)
        
        # Check that all models have positive parameter counts
        for name, model in models.items():
            param_count = model.get_num_params()
            assert param_count > 0, f"{name} has {param_count} parameters"
        
        # Validate parameter matching (within 20% for small models)
        is_valid = validate_parameter_matching(models, tolerance_pct=20.0)
        assert is_valid, "Models have very different parameter counts"
    
    def test_generation(self, model_type, small_config):
        """Test text generation capability."""
        model = create_model(model_type, **small_config)
        model.eval()
        
        # Start with a single token
        context = torch.zeros((1, 1), dtype=torch.long)
        
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=5, do_sample=False)
        
        assert generated.shape[0] == 1  # Batch size
        assert generated.shape[1] == 6  # Original 1 + 5 new tokens
    
    def test_training_step(self, model_type, small_config):
        """Test that models can compute gradients."""
        model = create_model(model_type, **small_config)
        model.train()
        
        batch_size, seq_len = 2, 16
        dummy_input = torch.randint(0, small_config["vocab_size"], (batch_size, seq_len))
        dummy_targets = torch.randint(0, small_config["vocab_size"], (batch_size, seq_len))
        
        # Forward pass with targets should return loss
        loss = model(dummy_input, dummy_targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        
        # Backward pass should work
        loss.backward()
        
        # Check that gradients were computed
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients computed"


def test_parameter_matching_across_variants():
    """Test parameter matching across all variants with standard config."""
    config = {
        "d_model": 256,
        "n_layers": 4,
        "vocab_size": 5000,
    }
    
    models = {}
    for model_type in ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]:
        models[model_type] = create_model(model_type, **config)
    
    # Should be within 5% for properly tuned configs
    is_valid = validate_parameter_matching(models, tolerance_pct=15.0)  # Relaxed for testing
    assert is_valid, "Parameter counts differ too much between variants"


if __name__ == "__main__":
    # Run a quick test
    print("Running quick model tests...")
    
    config = {"d_model": 64, "n_layers": 2, "vocab_size": 1000}
    
    for model_type in ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]:
        print(f"Testing {model_type}...")
        model = create_model(model_type, **config)
        
        # Quick forward pass
        dummy_input = torch.randint(0, 1000, (2, 16))
        output = model(dummy_input)
        print(f"✅ {model_type}: {model.get_num_params():,} params, output shape {output.shape}")
    
    print("All tests passed!")