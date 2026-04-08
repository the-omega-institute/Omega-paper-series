"""
Unified plotting script for all frontier experiments.
Generates publication-quality figures for JMLR paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import matplotlib as mpl


# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 13


def load_results(result_file: Path) -> List[Dict[str, Any]]:
    """Load experiment results from JSON file."""
    if not result_file.exists():
        print(f"Warning: {result_file} not found")
        return []
    
    with open(result_file, 'r') as f:
        return json.load(f)


def plot_multi_dataset_frontier(output_dir: Path):
    """
    Figure for Axis 1: Multi-Dataset Multi-Method Frontier
    Shows capability-risk frontier across datasets and training methods.
    """
    # Helper function to safely extract robust error
    def get_robust_error(r, rho="rho_0.25"):
        """Extract robust error from various JSON formats."""
        if "robust_errors" in r and isinstance(r["robust_errors"], dict):
            return r["robust_errors"].get(rho, r["robust_errors"].get("0.25", 0))
        elif "robust_error" in r:
            return r["robust_error"]
        elif "clean_error" in r:
            return r["clean_error"]  # Fallback
        else:
            return 0.0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Load results from different experiments
    cifar10_results = load_results(output_dir / "cifar_frontier_results.json")
    cifar100_results = load_results(output_dir / "cifar100_frontier_results.json")
    pgd_results = load_results(output_dir / "pgd_at_results.json")
    spectral_results = load_results(output_dir / "spectral_norm_results.json")
    
    # Combine all results
    all_results = {
        "CIFAR-10 ERM": cifar10_results,
        "CIFAR-100 ERM": cifar100_results,
        "CIFAR-10 PGD-AT": [r for r in pgd_results if r.get("config", {}).get("dataset") == "cifar10"],
        "CIFAR-10 Spectral": [r for r in spectral_results if r.get("method") == "Spectral-Norm"]
    }
    
    # Plot 1: Robust Error vs Lipschitz Constant (log scale)
    ax1 = axes[0]
    markers = ['o', 's', '^', 'd']
    colors = plt.cm.tab10(range(len(all_results)))
    
    for (method_name, results), marker, color in zip(all_results.items(), markers, colors):
        if not results:
            continue
        
        lipschitz_vals = []
        robust_errors = []
        
        for r in results:
            L = r.get("lipschitz_constant", r.get("L_est", 1e10))
            rob_err = get_robust_error(r)
            
            lipschitz_vals.append(L)
            robust_errors.append(rob_err)
        
        if lipschitz_vals:
            ax1.scatter(lipschitz_vals, robust_errors, label=method_name,
                       marker=marker, s=100, alpha=0.7, color=color, edgecolors='black', linewidths=1)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Lipschitz Constant $L$ (log scale)')
    ax1.set_ylabel('Robust Error ($\\rho=0.25$)')
    ax1.set_title('(a) Multi-Method Capability-Risk Frontier')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bound Tightness Comparison
    ax2 = axes[1]
    
    method_names = []
    bound_ratios = []
    
    for method_name, results in all_results.items():
        if not results:
            continue
        
        for r in results:
            clean_err = r.get("clean_error", 0)
            bound = r.get("capability_risk_bounds", {}).get("bound_rho_0.25",
                         r.get("pac_bayes_bound", 1.0))
            
            if clean_err > 0 and bound < 1.0:
                ratio = bound / clean_err
                method_names.append(method_name[:15])  # Truncate for display
                bound_ratios.append(ratio)
    
    if bound_ratios:
        x_pos = np.arange(len(method_names))
        bars = ax2.bar(x_pos, bound_ratios, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Color bars
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])
        
        ax2.set_ylabel('Bound / Empirical Error')
        ax2.set_title('(b) Bound Tightness Across Methods')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(method_names, rotation=45, ha='right')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect tightness')
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: Dataset Consistency
    ax3 = axes[2]
    
    datasets = ["CIFAR-10", "CIFAR-100"]
    clean_errors = []
    robust_errors = []
    
    # Helper function to safely extract robust error
    def get_robust_error(r, rho="rho_0.25"):
        # Try multiple field names for compatibility
        if "robust_errors" in r and isinstance(r["robust_errors"], dict):
            return r["robust_errors"].get(rho, r["robust_errors"].get("0.25", 0))
        elif "robust_error" in r:
            return r["robust_error"]
        else:
            return 0.0
    
    # CIFAR-10
    if cifar10_results:
        c10_clean = np.mean([r["clean_error"] for r in cifar10_results])
        c10_robust = np.mean([get_robust_error(r) for r in cifar10_results])
        clean_errors.append(c10_clean)
        robust_errors.append(c10_robust)
    else:
        clean_errors.append(0)
        robust_errors.append(0)
    
    # CIFAR-100
    if cifar100_results:
        c100_clean = np.mean([r["clean_error"] for r in cifar100_results])
        c100_robust = np.mean([get_robust_error(r) for r in cifar100_results])
        clean_errors.append(c100_clean)
        robust_errors.append(c100_robust)
    else:
        clean_errors.append(0)
        robust_errors.append(0)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax3.bar(x - width/2, clean_errors, width, label='Clean Error', alpha=0.8, edgecolor='black')
    ax3.bar(x + width/2, robust_errors, width, label='Robust Error ($\\rho=0.25$)', 
           alpha=0.8, edgecolor='black')
    
    ax3.set_ylabel('Classification Error')
    ax3.set_title('(c) Cross-Dataset Consistency')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "multi_dataset_frontier.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "multi_dataset_frontier.png", dpi=300, bbox_inches='tight')
    print(f"Saved: multi_dataset_frontier.pdf/png")
    plt.close()


def plot_lipschitz_surrogate_comparison(output_dir: Path):
    """
    Figure for Axis 2: Lipschitz Surrogate Analysis
    Compares three Lipschitz estimation methods.
    """
    results_file = output_dir / "lipschitz_surrogate_analysis.json"
    if not results_file.exists():
        print(f"Warning: {results_file} not found, skipping Lipschitz plot")
        return
    
    results = load_results(results_file)
    
    if not results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Comparison of three methods
    ax1 = axes[0]
    
    for i, result in enumerate(results):
        dataset = result["dataset"]
        
        methods = ["Global\nSpectral", "Gradient\nBased", "Finite\nDifference"]
        L_values = [
            result["global_spectral"]["L_estimate"],
            result["gradient_based"]["L_estimate"],
            result["finite_difference"]["L_estimate"]
        ]
        
        x = np.arange(len(methods)) + i * 0.25
        ax1.bar(x, L_values, width=0.2, label=dataset.upper(), alpha=0.8, edgecolor='black')
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Lipschitz Constant Estimate (log scale)')
    ax1.set_title('(a) Lipschitz Estimation Methods Comparison')
    ax1.set_xticks(np.arange(len(methods)))
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Tightness improvement
    ax2 = axes[1]
    
    for result in results:
        dataset = result["dataset"]
        ratios = result["ratios"]
        
        improvement_labels = ["Global/Grad", "Global/FD"]
        improvements = [
            ratios["global_over_grad"],
            ratios["global_over_fd"]
        ]
        
        x = np.arange(len(improvement_labels))
        ax2.bar(x, improvements, width=0.6, alpha=0.8, edgecolor='black')
        ax2.set_yscale('log')
        ax2.set_ylabel('Tightness Improvement Factor')
        ax2.set_title('(b) Bound Tightening via Data-Dependent $\\bar{L}$')
        ax2.set_xticks(x)
        ax2.set_xticklabels(improvement_labels)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No improvement')
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lipschitz_surrogate_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "lipschitz_surrogate_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: lipschitz_surrogate_comparison.pdf/png")
    plt.close()


def plot_ssr_pipeline_results(output_dir: Path):
    """
    Figure for Axis 3: SSR Pipeline Results
    Shows effectiveness of SSR layers.
    """
    results_file = output_dir / "complex_gridworld_ssr_results.json"
    if not results_file.exists():
        print(f"Warning: {results_file} not found, skipping SSR plot")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Extract metrics
    no_shield = results["no_shield"]
    with_shield = results["with_shield"]
    
    # Plot 1: Catastrophic Rate Comparison
    ax1 = axes[0]
    
    methods = ["No Shield", "With SSR"]
    catastrophic_rates = [
        no_shield["catastrophic_rate"] * 100,
        with_shield["catastrophic_rate"] * 100
    ]
    
    bars = ax1.bar(methods, catastrophic_rates, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars[0].set_color('coral')
    bars[1].set_color('lightgreen')
    
    ax1.set_ylabel('Catastrophic Rate (%)')
    ax1.set_title('(a) Catastrophic Event Reduction')
    ax1.set_ylim([0, max(catastrophic_rates) * 1.2])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, catastrophic_rates)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Success Rate and Return
    ax2 = axes[1]
    
    x = np.arange(2)
    width = 0.35
    
    success_rates = [no_shield["success_rate"] * 100, with_shield["success_rate"] * 100]
    avg_returns = [no_shield["avg_return"], with_shield["avg_return"]]
    
    # Normalize returns to percentage for visualization
    max_return = max(abs(min(avg_returns)), max(avg_returns))
    normalized_returns = [(r / max_return) * 100 for r in avg_returns]
    
    ax2.bar(x - width/2, success_rates, width, label='Success Rate (%)', alpha=0.8, edgecolor='black')
    ax2.bar(x + width/2, normalized_returns, width, label='Avg Return (normalized)', 
           alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('Percentage')
    ax2.set_title('(b) Task Performance Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: Training curves
    ax3 = axes[2]
    
    # Plot catastrophic events over training
    training_no_shield = results["training_curves"]["no_shield"]
    training_with_shield = results["training_curves"]["with_shield"]
    
    # Compute moving average of catastrophic rate
    window = 50
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    if len(training_no_shield["catastrophic"]) >= window:
        cat_no_shield = moving_average(training_no_shield["catastrophic"], window)
        cat_with_shield = moving_average(training_with_shield["catastrophic"], window)
        
        x_axis = np.arange(len(cat_no_shield))
        
        ax3.plot(x_axis, cat_no_shield, label='No Shield', linewidth=2, alpha=0.8)
        ax3.plot(x_axis, cat_with_shield, label='With SSR', linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Training Episode')
        ax3.set_ylabel(f'Catastrophic Rate (MA-{window})')
        ax3.set_title('(c) Learning Curves: Safety During Training')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ssr_pipeline_results.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "ssr_pipeline_results.png", dpi=300, bbox_inches='tight')
    print(f"Saved: ssr_pipeline_results.pdf/png")
    plt.close()


def main():
    """Generate all publication figures."""
    output_dir = Path(__file__).parent
    
    print("="*70)
    print("Generating Unified Frontier Plots for JMLR Paper")
    print("="*70)
    
    # Generate all figures
    print("\n1. Multi-Dataset Multi-Method Frontier...")
    plot_multi_dataset_frontier(output_dir)
    
    print("\n2. Lipschitz Surrogate Comparison...")
    plot_lipschitz_surrogate_comparison(output_dir)
    
    print("\n3. SSR Pipeline Results...")
    plot_ssr_pipeline_results(output_dir)
    
    print("\n" + "="*70)
    print("All plots generated successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

