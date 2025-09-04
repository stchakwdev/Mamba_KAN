"""Training script for model comparison."""

import torch
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mamba_kan.models import create_model
from mamba_kan.utils.parameter_counter import compare_parameter_counts, validate_parameter_matching


def main():
    parser = argparse.ArgumentParser(description="Train Mamba-KAN comparison models")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"],
                       help="Model variant to train")
    parser.add_argument("--compare_all", action="store_true", 
                       help="Compare parameter counts of all models")
    parser.add_argument("--d_model", type=int, default=512,
                       help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6,
                       help="Number of layers")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.compare_all:
        # Compare all four models
        print("Creating all four model variants for comparison...")
        
        models = {}
        for model_type in ["mlp_transformer", "kan_transformer", "mlp_mamba", "kan_mamba"]:
            print(f"Creating {model_type}...")
            model = create_model(
                model_type,
                d_model=args.d_model,
                n_layers=args.n_layers,
                device=str(device)
            )
            models[model_type] = model
        
        # Compare parameter counts
        compare_parameter_counts(models)
        
        # Validate parameter matching
        if validate_parameter_matching(models, tolerance_pct=5.0):
            print("\n✅ All models have similar parameter counts (within 5% tolerance)")
        else:
            print("\n❌ Models have significantly different parameter counts")
    
    else:
        # Create and test single model
        print(f"Creating {args.model_type}...")
        
        model = create_model(
            args.model_type,
            d_model=args.d_model,
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            device=str(device)
        )
        
        model = model.to(device)
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = args.batch_size
        seq_len = 128
        
        # Create dummy input
        dummy_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
            print(f"✅ Forward pass successful! Output shape: {logits.shape}")
        
        # Test generation
        print("Testing text generation...")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=10)
        print(f"✅ Generation successful! Generated {generated.shape[1]} tokens")
        
        print(f"\nModel summary:")
        print(f"- Type: {args.model_type}")
        print(f"- Parameters: {model.get_num_params():,}")
        print(f"- Model dimension: {model.config.d_model}")
        print(f"- Layers: {model.config.n_layers}")


if __name__ == "__main__":
    main()