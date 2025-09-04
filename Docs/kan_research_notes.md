# KAN (Kolmogorov-Arnold Networks) Research Notes

## Key Information from Abstract (arXiv:2404.19756)

### Core Concept
- KANs are proposed as alternatives to Multi-Layer Perceptrons (MLPs)
- Inspired by the Kolmogorov-Arnold representation theorem
- Key difference: MLPs have fixed activation functions on nodes ("neurons"), KANs have learnable activation functions on edges ("weights")
- KANs have NO linear weights at all - every weight parameter is replaced by a univariate function parametrized as a spline

### Claimed Advantages
1. **Accuracy**: Much smaller KANs can achieve comparable or better accuracy than much larger MLPs in:
   - Data fitting
   - PDE solving
   - Theoretically and empirically possess faster neural scaling laws than MLPs

2. **Interpretability**: 
   - Can be intuitively visualized
   - Can easily interact with human users
   - Useful collaborators helping scientists (re)discover mathematical and physical laws

### Publication Status
- Accepted by International Conference on Learning Representations (ICLR) 2025
- Original submission: April 30, 2024
- Latest revision: February 9, 2025 (v5)
- Authors: Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark

### Code Availability
- Codes are available (mentioned in paper)

## URLs to Investigate Further
- Original paper: https://arxiv.org/abs/2404.19756
- Medium explanation: https://medium.com/data-science/understanding-kolmogorov-arnold-networks-kan-e317b1b4d075
- Comparison paper: https://arxiv.org/abs/2407.16674 (KAN or MLP: A Fairer Comparison)
- Reddit discussions: https://www.reddit.com/r/MachineLearning/comments/1clcu5i/d_kolmogorovarnold_network_is_just_an_mlp/



## Technical Details from Original Paper

### Architecture Comparison (MLP vs KAN)
From Figure 0.1 in the paper:
- **MLP**: Fixed activation functions on nodes, learnable weights on edges
- **KAN**: Learnable activation functions on edges, no linear weights

### Key Technical Points
1. **Spline-based Functions**: KANs replace every weight parameter with a univariate function parametrized as a spline
2. **No Linear Weights**: KANs have no linear weight matrices at all
3. **Fully-connected Structure**: Like MLPs, KANs have fully-connected structures
4. **Computational Considerations**: 
   - KANs might be more expensive since each weight parameter becomes a spline function
   - However, KANs usually allow much smaller computation graphs than MLPs

### Theoretical Foundation
- Based on Kolmogorov-Arnold representation theorem
- The theorem states that any multivariate continuous function can be represented as a finite composition of continuous functions of a single variable and the binary operation of addition
- Mathematical representation shown: f(x₁, ..., xₙ) = exp(1/N ∑ᵢ₌₁ᴺ sin²(xᵢ))

### Advantages Claimed
1. **Accuracy**: Smaller KANs can achieve comparable/better accuracy than larger MLPs
2. **Interpretability**: Can be intuitively visualized and interact with human users
3. **Scientific Discovery**: Useful for helping scientists (re)discover mathematical and physical laws
4. **Scaling Laws**: Possess faster neural scaling laws than MLPs

### Potential Drawbacks Mentioned
1. **Computational Cost**: Each weight parameter becomes a spline function, potentially more expensive
2. **Curse of Dimensionality**: Splines have serious curse of dimensionality problems
3. **Limited Modern Techniques**: Many modern techniques (e.g., back propagation) couldn't leverage the original depth-2 width-(2n + 1) representation


## Critical Findings from "KAN or MLP: A Fairer Comparison" (arXiv:2407.16674)

### Key Results (Controlled for Parameters and FLOPs)
1. **General Performance**: MLP generally outperforms KAN across most tasks
2. **Exception**: KAN only excels in symbolic formula representation tasks
3. **Source of KAN's Advantage**: Mainly stems from its B-spline activation function

### Task-Specific Performance
- **Machine Learning**: MLP > KAN
- **Computer Vision**: MLP > KAN  
- **Audio Processing**: MLP > KAN
- **Natural Language Processing**: MLP > KAN
- **Symbolic Formula Representation**: KAN > MLP

### Important Ablation Study Results
1. **B-spline in MLP**: When B-spline activation is applied to MLP, performance in symbolic formula representation significantly improves, surpassing or matching KAN
2. **Limited Benefit**: In tasks where MLP already excels over KAN, B-spline does not substantially enhance MLP's performance

### Continual Learning Issues
- **KAN's Forgetting Problem**: KAN's forgetting issue is more severe than MLP in standard class-incremental continual learning settings
- **Contradicts Original Claims**: This differs from findings reported in the original KAN paper

### Implications
- KAN's advantages may be more limited than initially claimed
- The B-spline activation function appears to be the key differentiator
- For most practical ML tasks, MLPs remain superior

