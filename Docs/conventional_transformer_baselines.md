# Conventional Transformer Baselines Research Notes

## NanoGPT - Key Baseline Implementation

### Overview
- **Repository**: https://github.com/karpathy/nanoGPT
- **Author**: Andrej Karpathy
- **Description**: "The simplest, fastest repository for training/finetuning medium-sized GPTs"
- **Purpose**: Rewrite of minGPT that prioritizes teeth over education
- **Stars**: 44k+ (highly popular)
- **Status**: Still under active development

### Architecture Characteristics
1. **Simplicity**: 
   - `train.py`: ~300-line boilerplate training loop
   - `model.py`: ~300-line GPT model definition
   - Total codebase is extremely minimal and readable

2. **GPT-2 Reproduction**:
   - Reproduces GPT-2 (124M) on OpenWebText
   - Runs on single 8XA100 40GB node
   - Training time: ~4 days
   - Achieves validation loss of ~2.85 (matches GPT-2 performance)

3. **Model Configurations**:
   - **Small (Shakespeare)**: 6-layer Transformer, 6 heads, 384 feature channels, 256 context size
   - **CPU Version**: 4 layers, 4 heads, 128 embedding size, 64 context size
   - **GPT-2 124M**: Full GPT-2 architecture reproduction

### Performance Benchmarks
1. **Shakespeare Character-Level**:
   - **GPU (A100)**: 3 minutes training, validation loss 1.4697
   - **CPU**: ~3 minutes training, validation loss 1.88
   - **Apple Silicon (MPS)**: 2-3X acceleration over CPU

2. **GPT-2 Reproduction**:
   - **Training**: 4 days on 8XA100 40GB
   - **Validation Loss**: ~2.85 (matches original GPT-2)
   - **Dataset**: OpenWebText (open reproduction of WebText)

### Technical Features
1. **Dependencies**:
   - PyTorch
   - NumPy
   - Transformers (HuggingFace)
   - Datasets (HuggingFace)
   - Tiktoken (OpenAI's BPE)
   - Wandb (logging)
   - Tqdm (progress bars)

2. **Training Features**:
   - PyTorch Distributed Data Parallel (DDP) support
   - Multi-node training capability
   - Configurable hyperparameters
   - Checkpoint saving/loading
   - Optional GPT-2 weight loading from OpenAI

3. **Flexibility**:
   - Easy to hack and modify
   - Train from scratch or finetune pretrained models
   - Supports various model sizes
   - CPU, GPU, and Apple Silicon support

### Baseline Performance Numbers
- **GPT-2 Small (124M)**: val loss ~2.85 on OpenWebText
- **GPT-2 Medium**: [baseline numbers available in repo]
- **GPT-2 Large**: [baseline numbers available in repo]
- **GPT-2 XL**: [baseline numbers available in repo]

### Key Advantages as Baseline
1. **Simplicity**: Minimal, readable codebase
2. **Reproducibility**: Matches original GPT-2 results
3. **Flexibility**: Easy to modify for experiments
4. **Documentation**: Well-documented with clear examples
5. **Community**: Large community (44k stars, 7.5k forks)
6. **Educational**: Great for understanding transformer internals

