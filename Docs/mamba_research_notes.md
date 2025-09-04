# Mamba Architecture Research Notes

## Key Information from Original Paper (arXiv:2312.00752)

### Core Concept
- **Authors**: Albert Gu, Tri Dao
- **Title**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- **Submission**: December 1, 2023 (revised May 31, 2024)

### Problem Addressed
- Foundation models are almost universally based on Transformer architecture and attention modules
- Many subquadratic-time architectures (linear attention, gated convolution, recurrent models, structured state space models) have been developed to address Transformers' computational inefficiency on long sequences
- However, these alternatives have not performed as well as attention on important modalities such as language

### Key Innovation: Selective State Space Models (SSMs)
1. **Core Weakness Identified**: Previous models' inability to perform content-based reasoning
2. **Solution**: Make SSM parameters functions of the input
   - Allows model to selectively propagate or forget information along sequence length dimension depending on current token
   - Addresses weakness with discrete modalities

### Architecture Features
- **Simplified Design**: End-to-end neural network architecture WITHOUT attention or even MLP blocks
- **Hardware-Aware**: Designed a hardware-aware parallel algorithm in recurrent mode
- **Linear Scaling**: Linear scaling in sequence length (vs quadratic for Transformers)

### Performance Claims
1. **Speed**: 5× higher throughput than Transformers
2. **Long Sequences**: Performance improves on real data up to million-length sequences
3. **Model Performance**: Mamba-3B model:
   - Outperforms Transformers of the same size
   - Matches Transformers twice its size
   - Both in pretraining and downstream evaluation

### Modalities Tested
- **Language modeling** (primary focus)
- **Audio processing**
- **Genomics**
- Achieves state-of-the-art performance across several modalities

### Key Technical Advantage
- **Linear Time Complexity**: Unlike Transformers' quadratic complexity with sequence length
- **Content-Based Reasoning**: Selective mechanism allows dynamic information processing


## Key Insights from Visual Guide

### Transformer Problems
1. **Training vs Inference Trade-off**:
   - **Training**: Fast (parallelizable) - can calculate all attention weights simultaneously
   - **Inference**: Slow (scales quadratically) - must recalculate attention for entire sequence when generating each new token

2. **Computational Complexity**:
   - Generating tokens for sequence of length L needs roughly L² computations
   - Major bottleneck for long sequences
   - Need to recalculate entire sequence even for previously generated tokens

### RNN Advantages and Limitations
1. **Advantages**:
   - **Linear Scaling**: Inference scales linearly with sequence length
   - **Infinite Context**: Theoretically can have infinite context length
   - **Efficient Generation**: Only needs previous hidden state and current input

2. **Limitations**:
   - **Forgetting Problem**: Tend to forget information over time (only consider one previous state)
   - **Compressed View**: Hidden state is aggregation of all previous states, loses specific information
   - **Accuracy**: Lacked accuracy that Transformer models could offer

### State Space Models (SSM) Motivation
- Designed to efficiently use RNNs (and sometimes convolutions)
- Process sequences of information like text and signals
- Bridge between RNN efficiency and Transformer accuracy

### State Space Concept
- **Definition**: Contains minimum number of variables that fully describe a system
- **Mathematical Representation**: Way to track where you are, where you can go, and how you can get there
- **Neural Network Context**: "State" typically refers to hidden state, crucial for generating new tokens


### State Space Model (SSM) Mathematical Foundations

#### Core SSM Equations
1. **State Equation**: h'(t) = Ah(t) + Bx(t)
   - Describes how state changes based on:
     - Matrix A: How the state evolves over time
     - Matrix B: How input influences the state

2. **Output Equation**: y(t) = Ch(t) + Dx(t)
   - Describes how to derive output from:
     - Matrix C: How state maps to output
     - Matrix D: Direct input-to-output mapping

#### Key Concepts
- **Continuous Sequences**: Unlike discrete sequences, SSMs work with continuous input/output
- **State Representation h(t)**: The goal is to find optimal state representation for input-to-output mapping
- **Statistical Principles**: By solving these equations, uncover principles to predict system state from observed data

#### SSM Process Flow
1. Map input sequence x(t) → latent state representation h(t) → predicted output sequence y(t)
2. Process sequences of information (text, signals, etc.)
3. Track system state and predict next state based on input

#### Connection to Neural Networks
- **State**: Typically the hidden state in neural networks
- **LLM Context**: One of most important aspects for generating new tokens
- **Embeddings**: Vectors frequently used to describe "state" of input sequence

