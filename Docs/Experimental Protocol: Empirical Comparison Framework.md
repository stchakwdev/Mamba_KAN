# Experimental Protocol: Empirical Comparison Framework

## 1. Experimental Design Principles

### 1.1. Factorial Design Structure

This study employs a 2×2×4 factorial design:
- **Factor A**: Feedforward Architecture (MLP vs KAN)
- **Factor B**: Sequence Modeling (Transformer vs Mamba)  
- **Factor C**: Task Type (Math, Long-Sequence, Language, Code)

**Total Conditions**: 2 × 2 × 4 = 16 experimental conditions
**Replications**: 5 random seeds per condition = 80 total experiments

### 1.2. Control Variables

To ensure empirical rigor, the following variables will be strictly controlled:

#### Model Architecture Controls
- **Parameter Count**: All models matched within ±5% (target: ~124M parameters)
- **Layer Count**: Same number of layers across architectures
- **Hidden Dimensions**: Adjusted to maintain parameter parity
- **Attention Heads**: Same for transformer variants (8 heads)

#### Training Controls
- **Dataset**: Identical training/validation/test splits
- **Preprocessing**: Same tokenization and normalization
- **Batch Size**: Same effective batch size (may adjust micro-batch for memory)
- **Learning Rate**: Same base learning rate with architecture-specific scaling
- **Optimization**: Same optimizer (AdamW) and hyperparameters
- **Regularization**: Same dropout and weight decay
- **Training Steps**: Same number of training steps for all models

#### Environmental Controls
- **Hardware**: Same GPU type (A100 40GB) for all experiments
- **Software**: Same PyTorch version and CUDA version
- **Random Seeds**: Controlled and documented for reproducibility
- **Precision**: Same numerical precision (fp16 with automatic mixed precision)

## 2. Parameter Matching Protocol

### 2.1. Baseline Parameter Calculation

Starting with MLP-Transformer as baseline (similar to GPT-2 124M):

```python
baseline_config = {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,  # 4 * n_embd
}

# Baseline parameters: ~124M
```

### 2.2. Parameter Adjustment Rules

#### KAN Parameter Overhead Calculation
```python
def calculate_kan_overhead(input_dim, output_dim, grid_size=5, spline_order=3):
    """
    KAN uses (grid_size + spline_order) coefficients per input-output pair
    Plus scale and bias parameters
    """
    coefficients = (grid_size + spline_order) * input_dim * output_dim
    scale_bias = 2 * input_dim * output_dim
    return coefficients + scale_bias

def calculate_mlp_params(input_dim, output_dim):
    """Standard linear layer parameters"""
    return input_dim * output_dim + output_dim  # weights + bias
```

#### Dimension Adjustment Algorithm
```python
def match_parameters(target_params, architecture, feedforward_type):
    """
    Iteratively adjust dimensions to match target parameter count
    """
    config = base_config.copy()
    
    while True:
        current_params = count_model_parameters(config, architecture, feedforward_type)
        
        if abs(current_params - target_params) / target_params < 0.05:
            break
            
        if current_params > target_params:
            # Reduce dimensions
            if feedforward_type == "kan":
                config["kan_grid_size"] = max(3, config["kan_grid_size"] - 1)
            config["n_inner"] = int(config["n_inner"] * 0.95)
        else:
            # Increase dimensions
            if feedforward_type == "kan":
                config["kan_grid_size"] = min(7, config["kan_grid_size"] + 1)
            config["n_inner"] = int(config["n_inner"] * 1.05)
    
    return config
```

### 2.3. Final Parameter Verification

Before training, verify parameter counts:

| Model Variant | Target Params | Actual Params | Difference |
|---------------|---------------|---------------|------------|
| MLP-Transformer | 124M | 124.0M | 0.0% |
| KAN-Transformer | 124M | 123.2M | -0.6% |
| MLP-Mamba | 124M | 124.8M | +0.6% |
| KAN-Mamba | 124M | 123.5M | -0.4% |

## 3. Training Protocol

### 3.1. Hyperparameter Configuration

#### Base Hyperparameters (Same for All Models)
```python
training_config = {
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 2000,
    "max_steps": 100000,
    "eval_interval": 1000,
    "save_interval": 5000,
    "batch_size": 32,  # Per GPU
    "gradient_accumulation": 4,  # Effective batch size: 128
}
```

#### Architecture-Specific Adjustments
```python
architecture_adjustments = {
    "kan_models": {
        "learning_rate_scale": 0.8,  # KAN may need lower LR
        "warmup_steps": 3000,        # Longer warmup for stability
    },
    "mamba_models": {
        "learning_rate_scale": 1.2,  # Mamba may handle higher LR
        "gradient_accumulation": 2,   # Better memory efficiency
    }
}
```

### 3.2. Training Procedure

#### Phase 1: Model Initialization
1. **Initialize models** with same random seed for weight initialization
2. **Verify parameter counts** match within tolerance
3. **Test forward pass** on dummy data to ensure no errors
4. **Calculate theoretical FLOPs** for each model

#### Phase 2: Training Execution
```python
def train_model(model, config, seed):
    """Standardized training procedure"""
    
    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), **config.optimizer)
    scheduler = CosineAnnealingLR(optimizer, **config.scheduler)
    
    # Training loop with standardized logging
    for step in range(config.max_steps):
        batch = next(train_loader)
        
        # Forward pass
        loss = model(batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        if (step + 1) % config.gradient_accumulation == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging and evaluation
        if step % config.eval_interval == 0:
            eval_loss = evaluate_model(model, val_loader)
            log_metrics(step, loss, eval_loss, learning_rate)
```

### 3.3. Early Stopping and Convergence Criteria

```python
early_stopping_config = {
    "patience": 10,           # Number of evaluations without improvement
    "min_delta": 0.001,      # Minimum change to qualify as improvement
    "monitor": "val_loss",   # Metric to monitor
    "mode": "min",           # Lower is better for loss
}
```

## 4. Evaluation Protocol

### 4.1. Task Selection and Rationale

#### Mathematical Reasoning Tasks (KAN Expected Advantage)
- **GSM8K**: Grade school math word problems
  - *Rationale*: Tests symbolic manipulation and multi-step reasoning
  - *Metric*: Exact match accuracy
  - *Sample Size*: 1,319 test examples

- **MATH**: Competition mathematics problems  
  - *Rationale*: Tests advanced mathematical reasoning
  - *Metric*: Exact match accuracy
  - *Sample Size*: 5,000 test examples

- **Symbolic Regression**: Function discovery from data points
  - *Rationale*: Direct test of function approximation ability
  - *Metric*: Mean squared error on held-out points
  - *Sample Size*: 1,000 synthetic functions

#### Long Sequence Tasks (Mamba Expected Advantage)
- **Long Range Arena**: Standardized long sequence benchmark
  - *Rationale*: Tests ability to model long-range dependencies
  - *Metric*: Task-specific accuracy/F1
  - *Sequence Lengths*: 1K to 16K tokens

- **BookCorpus Long**: Long-form text modeling
  - *Rationale*: Tests language modeling on extended contexts
  - *Metric*: Perplexity on sequences >2K tokens
  - *Sample Size*: 10,000 long passages

#### General Language Understanding (Neutral Ground)
- **SuperGLUE**: Language understanding benchmark
  - *Rationale*: Standard evaluation for language models
  - *Metric*: Task-specific accuracy/F1
  - *Tasks*: 8 diverse language understanding tasks

- **HellaSwag**: Commonsense reasoning
  - *Rationale*: Tests world knowledge and reasoning
  - *Metric*: Accuracy
  - *Sample Size*: 10,042 test examples

#### Code Understanding (Mixed Expectations)
- **HumanEval**: Python code generation
  - *Rationale*: Tests structured reasoning and syntax knowledge
  - *Metric*: Pass@1, Pass@10
  - *Sample Size*: 164 programming problems

### 4.2. Evaluation Procedure

#### Standardized Evaluation Pipeline
```python
def evaluate_all_models(models, tasks, seeds):
    """
    Evaluate all model variants on all tasks with multiple seeds
    """
    results = {}
    
    for model_name, model_path in models.items():
        results[model_name] = {}
        
        for seed in seeds:
            # Load model checkpoint
            model = load_model(model_path, seed)
            
            for task_name, task in tasks.items():
                # Evaluate with fixed random seed
                torch.manual_seed(seed)
                score = task.evaluate(model)
                
                if task_name not in results[model_name]:
                    results[model_name][task_name] = []
                results[model_name][task_name].append(score)
    
    return results
```

#### Efficiency Measurements
```python
def measure_efficiency(model, test_data):
    """Measure computational efficiency metrics"""
    
    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    _ = model(test_data)
    peak_memory = torch.cuda.max_memory_allocated()
    
    # Throughput
    start_time = time.time()
    for _ in range(100):
        _ = model(test_data)
    throughput = 100 / (time.time() - start_time)
    
    # FLOPs (theoretical)
    flops = calculate_flops(model, test_data.shape)
    
    return {
        "peak_memory_mb": peak_memory / 1024**2,
        "throughput_samples_per_sec": throughput,
        "flops_per_sample": flops
    }
```

## 5. Statistical Analysis Plan

### 5.1. Primary Analysis: Factorial ANOVA

```python
# Statistical model
model_formula = """
score ~ C(feedforward) * C(sequence) * C(task_type) + 
        C(feedforward) * C(sequence) * C(model_size) +
        Error(C(seed))
"""

# Effect size calculations
def calculate_effect_sizes(anova_results, data):
    """Calculate eta-squared and Cohen's d for significant effects"""
    
    effects = {}
    
    # Main effects
    effects["feedforward_eta2"] = calculate_eta_squared(anova_results, "feedforward")
    effects["sequence_eta2"] = calculate_eta_squared(anova_results, "sequence")
    
    # Interaction effects  
    effects["interaction_eta2"] = calculate_eta_squared(anova_results, "feedforward:sequence")
    
    # Cohen's d for pairwise comparisons
    effects["kan_vs_mlp_d"] = calculate_cohens_d(data, "feedforward", "kan", "mlp")
    effects["mamba_vs_transformer_d"] = calculate_cohens_d(data, "sequence", "mamba", "transformer")
    
    return effects
```

### 5.2. Secondary Analyses

#### Task-Specific Analysis
```python
def analyze_by_task_type(data):
    """Separate analysis for each task type"""
    
    task_results = {}
    
    for task_type in ["math", "long_seq", "language", "code"]:
        task_data = data[data["task_type"] == task_type]
        
        # 2x2 ANOVA for this task type
        model = ols("score ~ C(feedforward) * C(sequence)", data=task_data).fit()
        anova_result = anova_lm(model, typ=2)
        
        task_results[task_type] = {
            "anova": anova_result,
            "means": task_data.groupby(["feedforward", "sequence"])["score"].mean(),
            "effect_sizes": calculate_effect_sizes(anova_result, task_data)
        }
    
    return task_results
```

#### Efficiency Analysis
```python
def analyze_efficiency(efficiency_data):
    """Analyze computational efficiency trade-offs"""
    
    # Performance vs efficiency scatter plots
    # Pareto frontier analysis
    # Efficiency ratios by architecture type
    
    return efficiency_analysis
```

### 5.3. Multiple Comparisons Correction

```python
def correct_multiple_comparisons(p_values, method="holm"):
    """Apply multiple comparisons correction"""
    
    from statsmodels.stats.multitest import multipletests
    
    corrected = multipletests(p_values, method=method)
    
    return {
        "corrected_p_values": corrected[1],
        "significant": corrected[0],
        "alpha_corrected": corrected[3] if len(corrected) > 3 else None
    }
```

## 6. Quality Assurance Checklist

### Pre-Experiment Validation
- [ ] All models have parameter counts within ±5% of target
- [ ] All models can successfully complete forward and backward passes
- [ ] Training infrastructure tested on small-scale runs
- [ ] Evaluation pipeline tested on subset of data
- [ ] Random seed management verified
- [ ] Hardware resources confirmed available

### During Experiment Monitoring
- [ ] Training loss curves monitored for anomalies
- [ ] Memory usage tracked to prevent OOM errors
- [ ] Checkpoint saving verified
- [ ] Evaluation metrics logged correctly
- [ ] No NaN or infinite values in outputs

### Post-Experiment Verification
- [ ] All planned experiments completed successfully
- [ ] Results files contain expected number of entries
- [ ] Statistical assumptions checked (normality, homoscedasticity)
- [ ] Effect sizes calculated and interpreted
- [ ] Results replicated with different random seeds
- [ ] Code and data archived for reproducibility

This experimental protocol ensures maximum empirical rigor while maintaining practical feasibility for the comparison study.

