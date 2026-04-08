# JMLR Experimental Suite: Capability-Risk Frontier Analysis

This directory contains the complete experimental implementation for the JMLR paper:
**"Universal Catastrophic Safety Undecidability and Capability-Risk Upper Bound Frontier"**

## Overview

The experimental suite validates three main theoretical contributions:

1. **Unified Capability-Risk Framework** (Theorem 5.3-5.4)
2. **Data-Dependent Lipschitz Bounds** (Theorem 5.4)
3. **SSR Safety Framework** (Algorithm 1 + Section 8)

## Directory Structure

```
jmlr_reproduction/
├── README_EXPERIMENTS.md          # This file
├── run_all_experiments.py         # Master script to run everything
│
├── Axis 1: Multi-Dataset Multi-Method Frontier
│   ├── cifar_capability_risk.py       # CIFAR-10 baseline (already exists)
│   ├── cifar100_frontier.py           # CIFAR-100 experiments
│   ├── adversarial_training.py        # PGD-AT robust training
│   └── spectral_regularization.py     # Spectral norm regularization
│
├── Axis 2: Lipschitz Surrogate Analysis
│   └── lipschitz_surrogates.py        # Compare 3 Lipschitz estimators
│
├── Axis 3: Complex Safe RL + SSR
│   ├── gridworld_shield.py            # Original 8×8 toy (already exists)
│   └── complex_gridworld_ssr.py       # 16×16 complex environment + full SSR
│
└── Plotting & Visualization
    ├── plot_frontier.py               # Original plots (already exists)
    └── plot_unified_frontier.py       # Unified publication figures
```

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision numpy matplotlib seaborn scipy

# For GPU acceleration (recommended):
# CUDA 11.8+ with compatible PyTorch
```

### Run All Experiments

**Full pipeline** (24-48 hours on single GPU):
```bash
python run_all_experiments.py
```

**Quick test** (2-4 hours, reduced epochs):
```bash
python run_all_experiments.py --quick
```

### Run Individual Experiments

```bash
# Axis 1: Multi-dataset frontier
python cifar100_frontier.py              # ~8 hours
python adversarial_training.py           # ~12 hours
python spectral_regularization.py        # ~8 hours

# Axis 2: Lipschitz analysis
python lipschitz_surrogates.py           # ~30 minutes

# Axis 3: Safe RL
python complex_gridworld_ssr.py          # ~2 hours

# Generate all figures
python plot_unified_frontier.py          # ~5 minutes
```

## Experimental Details

### Axis 1: Multi-Dataset Multi-Method Frontier

**Purpose**: Validate Theorem 5.3 (Unified Bound) across:
- Multiple datasets (CIFAR-10, CIFAR-100)
- Multiple training methods (ERM, PGD-AT, Spectral Norm)
- Multiple distribution shifts (Gaussian noise ρ ∈ {0, 0.25, 0.5})

**Key Metrics**:
- Clean error vs Lipschitz constant $L$
- Robust error vs $L$ (log-log plot)
- PAC-Bayes bound tightness
- Capability-risk bound validation

**Expected Outcomes**:
- Linear scaling: $R_{\text{rob}} \propto L\rho$ on log scale
- Bounds are valid (no violations) but loose (2-5× empirical error)
- Different methods exhibit different frontier positions

### Axis 2: Lipschitz Surrogate Analysis

**Purpose**: Validate Theorem 5.4 (Data-Dependent Bound)

**Three Estimation Methods**:
1. **Global Spectral**: $L_{\text{global}} = \prod_{\text{layers}} \sigma_{\max}$
   - Fastest, loosest
   - Typical: $L \sim 10^{10}$ to $10^{13}$

2. **Gradient-Based**: $\bar{L} = \mathbb{E}[\|\nabla_x \ell(f(x), y)\|]$
   - Medium cost
   - Typical: $\bar{L} \sim 10^2$ to $10^4$

3. **Finite-Difference**: $L_{\text{FD}} = \max_{\delta} |\ell(x+\delta) - \ell(x)|/\|\delta\|$
   - Slowest, tightest
   - Typical: $L_{\text{FD}} \sim 10$ to $10^3$

**Expected Improvement**: $L_{\text{global}} / \bar{L} \approx 10^6$ to $10^{10}$

This validates that data-dependent bounds are **orders of magnitude tighter** than global bounds.

### Axis 3: Complex GridWorld + SSR Pipeline

**Purpose**: Demonstrate SSR Framework (Section 8, Algorithm 1)

**Environment**: 16×16 grid with:
- Multiple hazard zones (scattered + rectangular regions)
- Wall obstacles
- Long-horizon planning (max 200 steps)

**SSR Layers**:
1. **Layer 1 (Scope)**: Restrict actions in hazard zones
2. **Layer 2 (Shield)**: One-step lookahead safety filter
3. **Layer 3 (Risk Budget)**: Capability-risk bound gating

**Comparison**:
- Baseline: Q-learning without shield
- Full SSR: All three layers active

**Expected Results**:
- Catastrophic rate: ~70% → ~0% (complete elimination)
- Intervention rate: ~20-30% of actions
- Performance cost: ~10-20% reduction in return

## Output Files

### JSON Results
- `cifar_frontier_results.json` - CIFAR-10 baseline
- `cifar100_frontier_results.json` - CIFAR-100 experiments
- `pgd_at_results.json` - Adversarial training
- `spectral_norm_results.json` - Spectral regularization
- `lipschitz_surrogate_analysis.json` - Lipschitz comparison
- `complex_gridworld_ssr_results.json` - SSR pipeline

### Figures (PDF + PNG)
- `multi_dataset_frontier.pdf` - Figure for Section 7.2 (Axis 1)
- `lipschitz_surrogate_comparison.pdf` - Figure for Section 7.4 (Axis 2)
- `ssr_pipeline_results.pdf` - Figure for Section 7.5 (Axis 3)

## Integration with LaTeX Paper

### Section 7: Experiments

The experiments directly support these sections:

- **Section 7.2**: Multi-Dataset Capability-Risk Frontier
  - Use: `multi_dataset_frontier.pdf`
  - Tables: Extract from `*_results.json` files

- **Section 7.3**: PAC-Bayes vs MI Ablation
  - Use: Existing `frontier_plot_pacbayes.png`

- **Section 7.4**: Lipschitz Surrogate Analysis (NEW)
  - Use: `lipschitz_surrogate_comparison.pdf`
  - Table: Tightness ratios from `lipschitz_surrogate_analysis.json`

- **Section 7.5**: Complex Safe RL + SSR (NEW)
  - Use: `ssr_pipeline_results.pdf`
  - Table: Catastrophic rates from `complex_gridworld_ssr_results.json`

### LaTeX Integration Example

```latex
\subsection{Lipschitz Surrogate Analysis}
\label{subsec:lipschitz_ablation}

To validate Theorem~\ref{thm:data_dependent_bound}, we compare three 
Lipschitz estimation methods...

\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{jmlr_reproduction/lipschitz_surrogate_comparison.pdf}
\caption{Comparison of Lipschitz estimators. (a) shows absolute estimates; 
(b) shows tightness improvement. Data-dependent $\bar{L}$ achieves 
$10^6$-$10^{10}$× tightening over global spectral norm.}
\label{fig:lipschitz_comparison}
\end{figure}
```

## Reproducibility

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 16 GB
- GPU: 1× NVIDIA GPU with 8GB VRAM
- Storage: 50 GB

**Recommended**:
- CPU: 8+ cores
- RAM: 32 GB
- GPU: 1× NVIDIA RTX 3090 or A100
- Storage: 100 GB

### Software Environment

```bash
# Exact versions used
python==3.10.12
torch==2.0.1+cu118
torchvision==0.15.2+cu118
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
```

Create environment:
```bash
conda create -n jmlr_experiments python=3.10
conda activate jmlr_experiments
pip install -r requirements.txt
```

### Random Seeds

All experiments use fixed seeds for reproducibility:
- Training: seeds {42, 43, 44} for 3 independent runs
- Evaluation: seed 0

## Computational Budget

| Experiment | Time (GPU) | Samples | Cost (A100-hours) |
|------------|-----------|---------|-------------------|
| CIFAR-100 | 8 hrs | 50K train | 8 |
| PGD-AT (CIFAR-10) | 12 hrs | 50K train | 12 |
| Spectral Norm | 8 hrs | 50K train | 8 |
| Lipschitz Analysis | 0.5 hrs | 10K test | 0.5 |
| Complex GridWorld | 2 hrs | 2K episodes | 0.1 (CPU) |
| **Total** | **~30 hrs** | **~160K samples** | **~29** |

At $2/A100-hour: **~$60 total**

## Troubleshooting

### Out of Memory

Reduce batch size:
```python
# In config files
batch_size: int = 64  # Instead of 128
```

### Slow Training

Enable mixed precision:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
```

### Missing Results Files

Run individual experiments first:
```bash
python cifar_capability_risk.py  # Generate CIFAR-10 baseline
python gridworld_shield.py       # Generate GridWorld baseline
```

## Citation

If you use this experimental suite, please cite:

```bibtex
@article{author2025universal,
  title={Universal Catastrophic Safety Undecidability and Capability-Risk Upper Bound Frontier},
  author={Author Name},
  journal={Journal of Machine Learning Research},
  year={2025}
}
```

## Contact

For questions about the experiments:
- Open an issue on GitHub
- Email: [author@institution.edu]

---

**Last Updated**: 2025-11-30  
**Version**: 2.0 (JMLR Submission)  
**Status**: Ready for submission

