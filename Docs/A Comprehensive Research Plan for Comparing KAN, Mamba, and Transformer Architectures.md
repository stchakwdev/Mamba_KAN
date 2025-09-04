# A Comprehensive Research Plan for Comparing KAN, Mamba, and Transformer Architectures

**Date:** September 4, 2025

## 1. Introduction

The field of deep learning is characterized by rapid innovation, with new neural network architectures constantly emerging to challenge the dominance of established models. For years, the Transformer architecture has been the undisputed king of natural language processing and has made significant inroads into other domains. However, its quadratic scaling with sequence length and its black-box nature have motivated the search for more efficient and interpretable alternatives.

This research plan outlines a comprehensive study to compare two promising new architectures, Kolmogorov-Arnold Networks (KANs) and Mamba (a state-space model), against the conventional Transformer architecture, using NanoGPT as a representative baseline. The goal is to provide a detailed analysis of the advantages and drawbacks of each architecture, identify suitable datasets and evaluation metrics for a fair comparison, and compile the necessary implementation resources to carry out the study.

This report is structured as follows:

*   **Section 2: Kolmogorov-Arnold Networks (KANs)** provides a detailed overview of the KAN architecture, its theoretical underpinnings, and its potential advantages in terms of interpretability and accuracy.
*   **Section 3: Mamba and State-Space Models (SSMs)** explores the Mamba architecture, its linear-time scaling, and its suitability for long-sequence modeling tasks.
*   **Section 4: Conventional Transformer Baselines** analyzes the NanoGPT implementation as a representative of the conventional Transformer architecture, highlighting its strengths and weaknesses.
*   **Section 5: Datasets for Comparison Study** identifies a diverse set of datasets for evaluating the different architectures, including tasks that are expected to favor each model.
*   **Section 6: Evaluation Metrics and Testing Methodologies** details the performance metrics, computational efficiency measures, and statistical testing approaches that will be used to compare the models.
*   **Section 7: Implementation Resources and Code Repositories** compiles a list of the necessary software, libraries, and code repositories for implementing and evaluating the different architectures.

By the end of this report, we will have a clear and actionable plan for conducting a rigorous and comprehensive comparison of these three exciting neural network architectures.



## 2. Kolmogorov-Arnold Networks (KANs)

Kolmogorov-Arnold Networks (KANs) have recently emerged as a promising alternative to traditional Multi-Layer Perceptrons (MLPs). Inspired by the Kolmogorov-Arnold representation theorem, KANs introduce a fundamental shift in neural network design: instead of having fixed activation functions on nodes and learnable weights on edges, KANs have learnable activation functions on edges and no linear weights at all. This seemingly simple change has profound implications for the accuracy, interpretability, and scalability of neural networks.

### 2.1. Theoretical Foundations

The theoretical underpinning of KANs is the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be represented as a finite composition of continuous functions of a single variable and the binary operation of addition. This theorem suggests that it is possible to build a neural network that can approximate any continuous function with a relatively simple architecture.

In a KAN, each weight parameter in a traditional MLP is replaced by a univariate function, which is typically parameterized as a B-spline. This allows the network to learn the optimal activation function for each connection, rather than being constrained to a fixed set of activation functions. This flexibility is the key to KANs' potential advantages.

### 2.2. Architectural Comparison: KAN vs. MLP

The architectural differences between KANs and MLPs can be summarized as follows:

| Feature | Multi-Layer Perceptron (MLP) | Kolmogorov-Arnold Network (KAN) |
| :--- | :--- | :--- |
| **Activation Functions** | Fixed, on nodes | Learnable, on edges |
| **Weights** | Learnable, on edges | No linear weights |
| **Structure** | Fully-connected | Fully-connected |
| **Interpretability** | Black-box | Intuitively visualizable |
| **Scaling** | Slower neural scaling laws | Faster neural scaling laws |

### 2.3. Claimed Advantages and Drawbacks

The original KAN paper [1] claims several advantages over MLPs:

*   **Accuracy**: KANs can achieve comparable or better accuracy than much larger MLPs on a variety of tasks, including data fitting and solving partial differential equations.
*   **Interpretability**: The learnable activation functions in KANs can be visualized, making it easier to understand how the network is making its decisions. This is a significant advantage over the black-box nature of MLPs.
*   **Scientific Discovery**: The interpretability of KANs makes them a useful tool for scientists to discover or rediscover mathematical and physical laws from data.

However, the paper also acknowledges some potential drawbacks:

*   **Computational Cost**: Since each weight parameter is replaced by a spline function, KANs can be more computationally expensive than MLPs.
*   **Curse of Dimensionality**: Splines are known to suffer from the curse of dimensionality, which could limit the scalability of KANs to high-dimensional problems.

### 2.4. A Fairer Comparison: KAN vs. MLP

A more recent study [2] provides a fairer comparison between KANs and MLPs by controlling for the number of parameters and FLOPs. The key findings of this study are:

*   **General Performance**: MLPs generally outperform KANs across most tasks, including machine learning, computer vision, audio processing, and natural language processing.
*   **Symbolic Formula Representation**: KANs only excel in tasks that involve symbolic formula representation.
*   **The Role of B-splines**: The advantage of KANs in symbolic tasks seems to stem from their use of B-spline activation functions. When B-splines are used as activation functions in MLPs, their performance in symbolic tasks significantly improves, often matching or surpassing that of KANs.
*   **Continual Learning**: KANs seem to suffer from a more severe forgetting problem than MLPs in continual learning settings, which contradicts the findings of the original KAN paper.

These findings suggest that the advantages of KANs may be more limited than initially claimed, and that the choice between KANs and MLPs should be carefully considered based on the specific task at hand.




## 3. Mamba and State-Space Models (SSMs)

Mamba has emerged as a powerful new architecture for sequence modeling, offering a compelling alternative to the dominant Transformer architecture. Based on the principles of State-Space Models (SSMs), Mamba addresses the quadratic scaling limitations of Transformers while introducing a selective mechanism that enables content-based reasoning. This section provides a detailed overview of the Mamba architecture, its theoretical foundations, and its potential advantages for long-sequence modeling tasks.

### 3.1. The Limitations of Transformers

While Transformers have revolutionized natural language processing, they suffer from a key limitation: their computational complexity scales quadratically with the sequence length. This is due to the self-attention mechanism, which requires every token to attend to every other token in the sequence. This quadratic scaling makes it computationally expensive to process long sequences, which is a significant bottleneck for many applications, such as high-resolution image generation, long-form document understanding, and genomic data analysis.

### 3.2. The Promise of State-Space Models

State-Space Models (SSMs) offer a promising solution to the limitations of Transformers. SSMs are a class of models that have been used for decades in control theory and signal processing. They are designed to model systems that evolve over time, and they have a natural ability to handle long sequences with linear-time complexity.

An SSM is defined by two core equations:

*   **State Equation**: `h'(t) = Ah(t) + Bx(t)`
*   **Output Equation**: `y(t) = Ch(t) + Dx(t)`

where:

*   `h(t)` is the hidden state of the system at time `t`
*   `x(t)` is the input to the system at time `t`
*   `y(t)` is the output of the system at time `t`
*   `A`, `B`, `C`, and `D` are matrices that define the dynamics of the system

In the context of neural networks, the hidden state `h(t)` can be thought of as a compressed representation of the sequence up to time `t`. The state equation describes how this hidden state is updated based on the current input and the previous hidden state. The output equation describes how the output is generated from the hidden state.

### 3.3. Mamba: Selective State-Space Models

The key innovation of Mamba is the introduction of a selective mechanism that allows the SSM to dynamically control the flow of information along the sequence. This is achieved by making the `A`, `B`, and `C` matrices functions of the input, which allows the model to selectively propagate or forget information depending on the current token. This selective mechanism is what gives Mamba its ability to perform content-based reasoning, which is a key weakness of previous SSM-based models.

### 3.4. Architectural Features and Performance

The Mamba architecture is designed to be simple and efficient. It consists of a stack of Mamba blocks, each of which contains a selective SSM and a small MLP. The entire architecture is end-to-end differentiable and can be trained with standard backpropagation.

The performance of Mamba is impressive. The original Mamba paper [3] reports that:

*   Mamba achieves 5x higher throughput than Transformers of the same size.
*   Mamba's performance improves on real data up to million-length sequences.
*   A Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size on both pretraining and downstream evaluation.

These results suggest that Mamba is a powerful and efficient architecture for sequence modeling, and it has the potential to replace Transformers on a wide range of tasks.




## 4. Conventional Transformer Baselines

To provide a meaningful comparison for KAN and Mamba, it is essential to establish a strong baseline using a conventional Transformer architecture. For this study, we have selected Andrej Karpathy's NanoGPT [4] as the primary baseline. NanoGPT is a minimalist implementation of the GPT-2 architecture that is designed for simplicity, readability, and hackability. These characteristics make it an ideal choice for a research setting, as it allows for easy experimentation and modification.

### 4.1. NanoGPT: A Minimalist GPT Implementation

NanoGPT is a ~600-line implementation of the GPT-2 architecture, with a ~300-line training loop and a ~300-line model definition. This simplicity makes it easy to understand the inner workings of the Transformer architecture and to modify the code for experimental purposes. Despite its minimalist design, NanoGPT is capable of reproducing the performance of the original GPT-2 (124M) model on the OpenWebText dataset, achieving a validation loss of ~2.85 after four days of training on an 8x A100 40GB node.

### 4.2. Architectural Features and Performance

NanoGPT supports a variety of model configurations, from a small character-level model for testing on a CPU to a full-fledged GPT-2 reproduction for large-scale experiments. It also includes support for multi-GPU training using PyTorch Distributed Data Parallel (DDP), which allows for efficient scaling to multiple GPUs and nodes.

The performance of NanoGPT is well-documented, with clear benchmarks for different model sizes and hardware configurations. This makes it easy to establish a reliable baseline for comparison against KAN and Mamba.

### 4.3. Advantages as a Baseline

NanoGPT offers several advantages as a baseline for this comparison study:

*   **Simplicity**: The minimalist codebase makes it easy to understand and modify.
*   **Reproducibility**: NanoGPT can reproduce the performance of the original GPT-2, providing a reliable baseline.
*   **Flexibility**: The code is easy to hack, allowing for a wide range of experiments.
*   **Documentation**: The repository is well-documented, with clear examples and instructions.
*   **Community**: NanoGPT has a large and active community, which provides a valuable resource for support and collaboration.

By using NanoGPT as our baseline, we can ensure that our comparison is fair, rigorous, and meaningful.




## 5. Datasets for Comparison Study

To provide a comprehensive and fair comparison of KAN, Mamba, and Transformer architectures, it is essential to select a diverse set of datasets that are designed to test the specific strengths and weaknesses of each model. This section outlines a detailed dataset selection strategy, including tasks that are expected to favor each architecture, as well as general language understanding benchmarks for a baseline comparison.

### 5.1. Mathematical and Symbolic Reasoning (KAN's Potential Strength)

Given KAN's theoretical advantages in function approximation and interpretability, it is expected to excel on tasks that require mathematical and symbolic reasoning. The following datasets will be used to test this hypothesis:

*   **Mathematical Problem Solving**: Datasets like GSM8K, MATH, and the DeepMind Mathematics Dataset will be used to evaluate the models' ability to solve mathematical word problems and competition-level mathematics problems.
*   **Symbolic Formula Representation**: Symbolic regression datasets and formula-based numerical reasoning tasks will be used to assess the models' ability to discover and represent mathematical equations from data.
*   **Advanced Mathematical Reasoning**: Datasets like OpenMathReasoning and DeepMath-103K will be used to test the models' ability to solve advanced mathematical problems that require multi-step reasoning.

### 5.2. Long-Sequence Modeling (Mamba's Potential Strength)

Mamba's linear-time scaling and selective state-space mechanism are expected to give it an advantage on tasks that involve long sequences. The following datasets will be used to test this hypothesis:

*   **Long-Sequence Language Modeling**: Datasets like OpenWebText, BookCorpus, and Wikipedia will be used to evaluate the models' ability to model long-range dependencies in text.
*   **Time Series and Sequential Data**: Audio processing datasets and genomics datasets will be used to assess the models' ability to handle long sequences in other modalities.
*   **Code Generation**: Programming tasks that require long context, such as those found in the CodeForces dataset, will be used to test the models' ability to generate coherent and functional code.

### 5.3. General Language Understanding (Baseline Comparison)

To provide a baseline comparison of the three architectures, a set of general language understanding benchmarks will be used. These benchmarks are designed to test a wide range of language understanding skills, including sentence classification, similarity, and inference.

*   **SuperGLUE Benchmark**: The SuperGLUE benchmark is a more difficult version of the GLUE benchmark, with a new set of more challenging language understanding tasks.
*   **GLUE Benchmark**: The GLUE benchmark is a widely used benchmark for evaluating the performance of language models on a variety of language understanding tasks.

### 5.4. Reasoning and Chain-of-Thought

To further probe the reasoning capabilities of the different architectures, a set of datasets that require multi-step reasoning and chain-of-thought will be used.

*   **Multi-Step Reasoning**: Datasets like OpenThoughts-114k and Bespoke-Stratos-17k will be used to evaluate the models' ability to perform multi-step reasoning on a variety of tasks, including math, science, code, and puzzles.
*   **Code Reasoning**: Datasets like OpenCodeReasoning and KodCode-V1 will be used to test the models' ability to reason about code and solve programming problems.

By using this diverse set of datasets, we can provide a comprehensive and fair comparison of KAN, Mamba, and Transformer architectures, and gain a deeper understanding of their respective strengths and weaknesses.




## 6. Evaluation Metrics and Testing Methodologies

To ensure a fair and rigorous comparison of KAN, Mamba, and Transformer architectures, it is essential to use a comprehensive set of evaluation metrics and testing methodologies. This section outlines a detailed evaluation framework that covers task-specific performance, computational efficiency, scalability, and statistical significance.

### 6.1. Performance Metrics

A multi-faceted approach to performance evaluation will be used, including:

*   **Task-Specific Metrics**: For classification tasks, we will use accuracy, precision, recall, F1-score, and AUC-ROC. For language modeling tasks, we will use perplexity, BLEU score, and ROUGE score. For mathematical reasoning tasks, we will use exact match, step-by-step accuracy, and formula accuracy.
*   **Computational Efficiency Metrics**: We will measure the computational efficiency of each architecture using metrics such as FLOPs, parameter count, peak memory usage, throughput, and latency.
*   **Scalability Metrics**: We will assess the scalability of each architecture by measuring its performance on varying sequence lengths, batch sizes, and model sizes.

### 6.2. Statistical Testing

To ensure that the observed differences in performance are statistically significant, we will use a variety of statistical tests, including:

*   **Parametric Tests**: Paired t-tests and ANOVA will be used to compare the means of the different models.
*   **Non-Parametric Tests**: Wilcoxon signed-rank tests and Kruskal-Wallis tests will be used when the assumptions of parametric tests are not met.
*   **Permutation Tests**: Randomization tests and bootstrap resampling will be used to generate confidence intervals and test the null hypothesis that the models have similar performance.

### 6.3. Experimental Design

A rigorous experimental design will be used to ensure that the comparison is fair and unbiased. This will include:

*   **Cross-Validation**: k-fold cross-validation will be used to obtain robust performance estimates.
*   **Hyperparameter Optimization**: A systematic approach to hyperparameter optimization will be used to ensure that each model is performing at its best.
*   **Baseline Comparisons**: A set of appropriate baselines will be used to provide a meaningful comparison, including random baselines, majority class baselines, and previous state-of-the-art models.

### 6.4. Specialized Evaluation

In addition to the general evaluation framework, a set of specialized evaluation metrics will be used to assess the specific strengths and weaknesses of each architecture:

*   **For KANs**: We will use metrics such as symbolic accuracy, interpolation vs. extrapolation performance, and function approximation error to assess their ability to learn symbolic functions.
*   **For Mamba**: We will use metrics such as long-sequence performance, memory efficiency, and selective attention quality to assess its ability to handle long sequences.
*   **For Transformers**: We will use metrics such as attention pattern analysis and position encoding effectiveness to assess their ability to learn long-range dependencies.

By using this comprehensive evaluation framework, we can provide a fair, rigorous, and meaningful comparison of KAN, Mamba, and Transformer architectures, and gain a deeper understanding of their respective strengths and weaknesses.




## 7. Implementation Resources and Code Repositories

To facilitate the implementation of this comparison study, a comprehensive set of open-source libraries, code repositories, and benchmarking frameworks are available. This section provides a curated list of the most important resources for KAN, Mamba, and Transformer implementations.

### 7.1. KAN Implementations

*   **PyKAN**: The official implementation of KANs, providing a complete and well-documented library for building and training KANs.
*   **Efficient KAN**: An optimized implementation of KANs that offers better performance and reduced memory usage.
*   **FastKAN**: A very fast implementation of KANs that uses optimized basis functions for significant speed improvements.

### 7.2. Mamba Implementations

*   **Official Mamba Implementation**: The official implementation of the Mamba architecture, including pre-trained model checkpoints and CUDA kernels for efficiency.
*   **Mamba Minimal**: A simple and minimal implementation of Mamba in a single PyTorch file, ideal for educational purposes.
*   **Hugging Face Integration**: A collection of Mamba models that are compatible with the Hugging Face Transformers library, allowing for easy model loading and inference.

### 7.3. Transformer Baselines

*   **NanoGPT**: A minimalist implementation of the GPT-2 architecture that is designed for simplicity, readability, and hackability.
*   **MinGPT**: The predecessor to NanoGPT, providing a clear and well-documented implementation of the GPT architecture for educational purposes.

### 7.4. Comparison and Benchmarking Frameworks

*   **PyTorch Lightning**: A powerful and flexible framework for organizing PyTorch code, which simplifies multi-GPU training, logging, and checkpointing.
*   **Hugging Face Evaluate**: A library for easily evaluating machine learning models and datasets, with a comprehensive set of standardized evaluation metrics.
*   **MLCommons Benchmarks**: A collection of industry-standard benchmarks and performance measurement tools for fair and reproducible model comparison.

By leveraging these open-source resources, we can significantly reduce the implementation effort and focus on the core research questions of this study.




## 8. References

[1] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark. "KAN: Kolmogorov-Arnold Networks." arXiv preprint arXiv:2404.19756 (2024).
[https://arxiv.org/abs/2404.19756](https://arxiv.org/abs/2404.19756)

[2] Zhi-Fan Wu, et al. "KAN or MLP: A Fairer Comparison." arXiv preprint arXiv:2407.16674 (2024).
[https://arxiv.org/abs/2407.16674](https://arxiv.org/abs/2407.16674)

[3] Albert Gu, Tri Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv preprint arXiv:2312.00752 (2023).
[https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)

[4] Andrej Karpathy. "nanoGPT." GitHub repository (2022).
[https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)


