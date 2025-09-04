# Implementation Guide: Four Model Variants

## 1. Architecture Breakdown

### 1.1. Component Identification

To implement the 2×2 factorial design, we need to identify where feedforward components (MLP vs KAN) are used in each sequence modeling architecture:

#### Transformer Architecture Components
- **Attention mechanism**: Multi-head self-attention (unchanged)
- **Feedforward blocks**: MLP layers after attention (KAN replacement target)
- **Layer normalization**: Pre/post normalization (unchanged)
- **Positional encoding**: Position embeddings (unchanged)

#### Mamba Architecture Components  
- **State space mechanism**: Selective SSM (unchanged)
- **Input/output projections**: Linear layers for x→B,C,Δ (KAN replacement target)
- **Activation functions**: SiLU/Swish activations (unchanged in SSM core)
- **Convolution**: 1D conv for local dependencies (unchanged)

### 1.2. KAN Integration Points

| Architecture | MLP Components to Replace | KAN Integration Strategy |
|--------------|---------------------------|-------------------------|
| **Transformer** | FFN blocks (2-layer MLP) | Replace entire FFN with KAN |
| **Mamba** | Input/output projections | Replace linear projections with KAN |

## 2. Model Implementation Details

### 2.1. MLP-Transformer (Baseline)

```python
class MLPTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Standard MLP feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        # Attention block
        x = x + self.attention(self.norm1(x))
        # MLP block  
        x = x + self.ffn(self.norm2(x))
        return x
```

### 2.2. KAN-Transformer

```python
class KANTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, kan_grid_size=5):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # KAN feedforward block
        self.ffn = KAN([d_model, d_ff, d_model], 
                       grid_size=kan_grid_size,
                       spline_order=3)
    
    def forward(self, x):
        # Attention block (unchanged)
        x = x + self.attention(self.norm1(x))
        # KAN block
        x = x + self.ffn(self.norm2(x))
        return x
```

### 2.3. MLP-Mamba

```python
class MLPMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        # Input projection (MLP)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters projection (MLP)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # Output projection (MLP)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        # Standard Mamba forward pass with MLP projections
        return self.selective_scan(x)
```

### 2.4. KAN-Mamba

```python
class KANMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, kan_grid_size=5):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        # Input projection (KAN)
        self.in_proj = KAN([d_model, self.d_inner * 2], 
                          grid_size=kan_grid_size)
        
        # Convolution (unchanged)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters projection (KAN)
        self.x_proj = KAN([self.d_inner, d_state * 2], 
                         grid_size=kan_grid_size)
        self.dt_proj = KAN([self.d_inner, self.d_inner], 
                          grid_size=kan_grid_size)
        
        # Output projection (KAN)
        self.out_proj = KAN([self.d_inner, d_model], 
                           grid_size=kan_grid_size)
        
    def forward(self, x):
        # Mamba forward pass with KAN projections
        return self.selective_scan(x)
```

## 3. Parameter Matching Strategy

### 3.1. Parameter Counting

To ensure fair comparison, we need to match parameters across all four variants:

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def match_parameters(target_params, architecture_type, feedforward_type):
    """
    Adjust model dimensions to match target parameter count
    """
    if feedforward_type == "kan":
        # KAN uses more parameters due to spline coefficients
        # Reduce hidden dimensions accordingly
        adjustment_factor = 0.7  # Empirically determined
    else:
        adjustment_factor = 1.0
        
    if architecture_type == "mamba":
        # Mamba has different parameter distribution
        # Adjust expand ratio and state size
        pass
    
    return adjusted_config
```

### 3.2. Dimension Scaling Rules

| Model Variant | Parameter Adjustment Strategy |
|---------------|------------------------------|
| **MLP-Transformer** | Baseline (no adjustment) |
| **KAN-Transformer** | Reduce `d_ff` by ~30% to account for KAN overhead |
| **MLP-Mamba** | Match by adjusting `expand` ratio |
| **KAN-Mamba** | Reduce `expand` ratio and adjust KAN grid size |

## 4. Training Configuration

### 4.1. Unified Training Loop

```python
class UnifiedTrainer:
    def __init__(self, model_type, feedforward_type, config):
        self.model = self.build_model(model_type, feedforward_type, config)
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        
    def build_model(self, model_type, feedforward_type, config):
        if model_type == "transformer" and feedforward_type == "mlp":
            return MLPTransformer(config)
        elif model_type == "transformer" and feedforward_type == "kan":
            return KANTransformer(config)
        elif model_type == "mamba" and feedforward_type == "mlp":
            return MLPMamba(config)
        elif model_type == "mamba" and feedforward_type == "kan":
            return KANMamba(config)
        else:
            raise ValueError(f"Unknown combination: {model_type}, {feedforward_type}")
    
    def train_step(self, batch):
        # Unified training step for all model types
        loss = self.model(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

### 4.2. Hyperparameter Configuration

```python
# Base configuration (adjust for each variant)
base_config = {
    "d_model": 512,
    "n_layers": 6,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "max_seq_len": 1024,
    "dropout": 0.1,
    "weight_decay": 0.01,
}

# Variant-specific adjustments
variant_configs = {
    "mlp_transformer": {
        **base_config,
        "n_heads": 8,
        "d_ff": 2048,
    },
    "kan_transformer": {
        **base_config,
        "n_heads": 8,
        "d_ff": 1400,  # Reduced for parameter matching
        "kan_grid_size": 5,
    },
    "mlp_mamba": {
        **base_config,
        "d_state": 16,
        "expand": 2,
        "d_conv": 4,
    },
    "kan_mamba": {
        **base_config,
        "d_state": 16,
        "expand": 1.4,  # Reduced for parameter matching
        "d_conv": 4,
        "kan_grid_size": 5,
    }
}
```

## 5. Evaluation Pipeline

### 5.1. Standardized Evaluation

```python
class EvaluationSuite:
    def __init__(self):
        self.tasks = {
            "math": [GSM8K(), MATH(), SymbolicRegression()],
            "long_seq": [LongRangeArena(), BookCorpus()],
            "language": [SuperGLUE(), HellaSwag()],
            "code": [HumanEval(), MBPP()]
        }
    
    def evaluate_model(self, model, model_name):
        results = {}
        for task_type, tasks in self.tasks.items():
            results[task_type] = {}
            for task in tasks:
                score = task.evaluate(model)
                results[task_type][task.name] = score
        
        return results
    
    def run_full_evaluation(self, models):
        """Run all four models on all tasks"""
        all_results = {}
        for model_name, model in models.items():
            all_results[model_name] = self.evaluate_model(model, model_name)
        
        return all_results
```

### 5.2. Statistical Analysis

```python
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

def analyze_results(results_dict):
    """
    Convert results to DataFrame and run factorial ANOVA
    """
    # Convert to long format
    data = []
    for model_name, task_results in results_dict.items():
        feedforward_type = "kan" if "kan" in model_name else "mlp"
        sequence_type = "mamba" if "mamba" in model_name else "transformer"
        
        for task_type, scores in task_results.items():
            for task_name, score in scores.items():
                data.append({
                    'model': model_name,
                    'feedforward': feedforward_type,
                    'sequence': sequence_type,
                    'task_type': task_type,
                    'task_name': task_name,
                    'score': score
                })
    
    df = pd.DataFrame(data)
    
    # Run factorial ANOVA
    model = ols('score ~ C(feedforward) * C(sequence) * C(task_type)', data=df).fit()
    anova_results = anova_lm(model, typ=2)
    
    return df, anova_results
```

## 6. Implementation Checklist

### Phase 1: Setup
- [ ] Install all required libraries (PyTorch, PyKAN, Mamba-SSM)
- [ ] Set up unified project structure
- [ ] Implement parameter counting utilities
- [ ] Create configuration management system

### Phase 2: Model Implementation
- [ ] Implement MLP-Transformer (validate against NanoGPT)
- [ ] Implement KAN-Transformer (test KAN integration)
- [ ] Implement MLP-Mamba (validate against official implementation)
- [ ] Implement KAN-Mamba (novel integration)

### Phase 3: Parameter Matching
- [ ] Calculate baseline parameter counts
- [ ] Adjust dimensions for each variant
- [ ] Validate parameter counts match within 5%
- [ ] Test all models train successfully

### Phase 4: Evaluation Setup
- [ ] Implement evaluation pipeline
- [ ] Set up data loaders for all tasks
- [ ] Create metrics calculation utilities
- [ ] Test evaluation on small examples

### Phase 5: Experiments
- [ ] Run pilot experiments (small scale)
- [ ] Run full experiments (multiple seeds)
- [ ] Collect efficiency metrics (FLOPs, memory, time)
- [ ] Save all results and model checkpoints

### Phase 6: Analysis
- [ ] Statistical analysis (ANOVA, effect sizes)
- [ ] Visualization of results
- [ ] Interpretability analysis (for KAN variants)
- [ ] Write comprehensive report

This implementation guide provides the technical foundation for conducting the rigorous 2×2 factorial comparison of neural network architectures.

