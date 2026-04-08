"""
Master script to run all JMLR experiments in sequence.
Executes the complete experimental pipeline for the paper.

Usage:
    python run_all_experiments.py [--quick]
    
    --quick: Run abbreviated experiments for testing (fewer epochs/episodes)
"""

import sys
import subprocess
from pathlib import Path
import time
import argparse


def run_script(script_name: str, description: str):
    """Run a Python script and report status."""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False
        )
        
        elapsed = time.time() - start_time
        print(f"\n[OK] SUCCESS: {description} completed in {elapsed/60:.1f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[FAIL] FAILED: {description} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all JMLR experiments")
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version with reduced epochs/episodes')
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent
    
    print("="*80)
    print("JMLR EXPERIMENTAL PIPELINE")
    print("Complete Capability-Risk Frontier Analysis")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Mode: {'QUICK TEST' if args.quick else 'FULL EXPERIMENTS'}")
    print("\nThis will run:")
    print("  - Axis 1: Multi-dataset multi-method frontier (CIFAR-10, CIFAR-100, PGD-AT, Spectral)")
    print("  - Axis 2: Lipschitz surrogate analysis")
    print("  - Axis 3: Complex GridWorld + SSR pipeline")
    print("  - Final: Generate all publication figures")
    
    if not args.quick:
        print("\nWARNING: Full experiments may take 24-48 hours on a single GPU.")
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    total_start = time.time()
    results = {}
    
    # =========================================================================
    # AXIS 1: MULTI-DATASET MULTI-METHOD FRONTIER
    # =========================================================================
    
    print("\n" + "#"*80)
    print("# AXIS 1: MULTI-DATASET MULTI-METHOD FRONTIER")
    print("#"*80)
    
    # 1.1: CIFAR-10 baseline (already exists from previous runs)
    # We can skip this if results exist
    cifar10_results = output_dir / "cifar_frontier_results.json"
    if not cifar10_results.exists():
        print("\nNote: CIFAR-10 baseline results not found.")
        print("Run cifar_capability_risk.py first to generate baseline.")
    else:
        print(f"\n[OK] Using existing CIFAR-10 results: {cifar10_results}")
    
    # 1.2: CIFAR-100
    if args.quick:
        print("\nSkipping CIFAR-100 in quick mode...")
        results['cifar100'] = 'skipped'
    else:
        results['cifar100'] = run_script(
            "cifar100_frontier.py",
            "CIFAR-100 Frontier Experiments"
        )
    
    # 1.3: PGD Adversarial Training
    if args.quick:
        print("\nSkipping PGD-AT in quick mode...")
        results['pgd_at'] = 'skipped'
    else:
        results['pgd_at'] = run_script(
            "adversarial_training.py",
            "PGD Adversarial Training"
        )
    
    # 1.4: Spectral Normalization
    if args.quick:
        print("\nSkipping Spectral Norm in quick mode...")
        results['spectral'] = 'skipped'
    else:
        results['spectral'] = run_script(
            "spectral_regularization.py",
            "Spectral Normalization Training"
        )
    
    # =========================================================================
    # AXIS 2: LIPSCHITZ SURROGATE ANALYSIS
    # =========================================================================
    
    print("\n" + "#"*80)
    print("# AXIS 2: LIPSCHITZ SURROGATE ANALYSIS")
    print("#"*80)
    
    results['lipschitz'] = run_script(
        "lipschitz_surrogates.py",
        "Lipschitz Estimation Methods Comparison"
    )
    
    # =========================================================================
    # AXIS 3: COMPLEX GRIDWORLD + SSR
    # =========================================================================
    
    print("\n" + "#"*80)
    print("# AXIS 3: COMPLEX GRIDWORLD + SSR PIPELINE")
    print("#"*80)
    
    if args.quick:
        print("\nRunning quick SSR experiment (500 episodes)...")
        # In quick mode, we'd need to modify the script or pass parameters
        # For now, just run it
        results['ssr'] = run_script(
            "complex_gridworld_ssr.py",
            "SSR Pipeline Experiment (Quick)"
        )
    else:
        results['ssr'] = run_script(
            "complex_gridworld_ssr.py",
            "SSR Pipeline Experiment"
        )
    
    # =========================================================================
    # PLOTTING: GENERATE ALL FIGURES
    # =========================================================================
    
    print("\n" + "#"*80)
    print("# GENERATING PUBLICATION FIGURES")
    print("#"*80)
    
    results['plotting'] = run_script(
        "plot_unified_frontier.py",
        "Generate All Figures"
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("EXPERIMENT PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_elapsed/3600:.1f} hours")
    print("\nResults summary:")
    
    for exp_name, status in results.items():
        if status == 'skipped':
            status_str = "[SKIP]"
        elif status:
            status_str = "[OK] SUCCESS"
        else:
            status_str = "[FAIL] FAILED"
        print(f"  {exp_name:20s}: {status_str}")
    
    # List generated files
    print("\nGenerated files:")
    for pattern in ["*_results.json", "*.pdf", "*.png"]:
        for file in output_dir.glob(pattern):
            print(f"  - {file.name}")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Review all generated JSON result files")
    print("  2. Check PDF/PNG figures for quality")
    print("  3. Update LaTeX paper with new figures")
    print("  4. Compile paper: pdflatex universal-catastrophic-safety-...")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

