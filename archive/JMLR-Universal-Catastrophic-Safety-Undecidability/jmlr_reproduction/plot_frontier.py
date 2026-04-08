import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results():
    results_path = os.path.join(os.path.dirname(__file__), "cifar_frontier_results.json")
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("No results found. Run cifar_frontier.py first.")
        return

    # Extract data
    l_ests = np.array([r["L_est"] for r in results])
    bounds = np.array([r["pacbayes_bound"] for r in results])
    clean_errs = np.array([r["clean_error"] for r in results])
    robust_errs = np.array([r["robust_error"] for r in results])
    bound_plus_lrho = np.array([r["bound_plus_Lrho"] for r in results])
    weight_decays = [r["config"]["weight_decay"] for r in results]
    
    print(f"Lipschitz constants: {l_ests}")
    print(f"PAC-Bayes bounds: {bounds}")
    print(f"Clean errors: {clean_errs}")
    print(f"Robust errors: {robust_errs}")
    
    # Plot 1: Frontier with L_est (LOG SCALE to handle huge values)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(l_ests, clean_errs, label="Clean Error", marker='o', s=100, alpha=0.7)
    ax.scatter(l_ests, robust_errs, label="Robust Error (Corrupted)", marker='x', s=100, alpha=0.7)
    
    # For upper bound, cap it at 1.0 for visualization since error is in [0,1]
    bound_capped = np.minimum(bound_plus_lrho, 1.0)
    ax.scatter(l_ests, bound_capped, label="Upper Bound (Capped at 1.0)", marker='^', s=100, alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_xlabel("Lipschitz Constant $L$ (log scale)", fontsize=12)
    ax.set_ylabel("Error / Risk", fontsize=12)
    ax.set_title("Capability-Risk Frontier: Error vs Lipschitz Constant", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_ylim([0, 1.0])
    
    # Annotate weight decay values
    for i, wd in enumerate(weight_decays):
        ax.annotate(f"WD={wd}", (l_ests[i], clean_errs[i]), 
                   textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
    
    output_file1 = os.path.join(os.path.dirname(__file__), "frontier_plot_lipschitz.png")
    plt.tight_layout()
    plt.savefig(output_file1, dpi=150)
    plt.close()
    
    # Plot 2: Frontier with PAC-Bayes Bound (Information/Complexity)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(bounds, clean_errs, label="Clean Error", marker='o', s=100, alpha=0.7)
    ax.scatter(bounds, robust_errs, label="Robust Error (Corrupted)", marker='x', s=100, alpha=0.7)
    ax.scatter(bounds, bounds, label="PAC-Bayes Bound (Identity)", marker='s', s=100, alpha=0.7)
    
    ax.set_xlabel("PAC-Bayes Bound (Information Complexity)", fontsize=12)
    ax.set_ylabel("Error / Risk", fontsize=12)
    ax.set_title("Capability-Risk Frontier vs Information Complexity", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_xlim([0.7, 1.3])
    ax.set_ylim([0.5, 1.0])
    
    # Annotate weight decay values
    for i, wd in enumerate(weight_decays):
        ax.annotate(f"WD={wd}", (bounds[i], clean_errs[i]), 
                   textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
    
    output_file2 = os.path.join(os.path.dirname(__file__), "frontier_plot_pacbayes.png")
    plt.tight_layout()
    plt.savefig(output_file2, dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_file1} and {output_file2}")

if __name__ == "__main__":
    plot_results()

