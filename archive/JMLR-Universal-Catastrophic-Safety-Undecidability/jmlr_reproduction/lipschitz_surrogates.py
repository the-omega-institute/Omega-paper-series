"""
Lipschitz Surrogate Analysis for Data-Dependent Bounds
Implements and compares three Lipschitz estimation methods:
1. Global spectral norm (product of layer spectral norms)
2. Gradient-based local average
3. Finite-difference sampling

This validates Theorem 5.4 (Data-Dependent Capability-Risk Bound).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import sys


sys.path.append(str(Path(__file__).parent))
from cifar_capability_risk import ResNet18Width, power_iteration


@dataclass
class LipschitzConfig:
    """Configuration for Lipschitz surrogate analysis."""
    dataset: str = "cifar10"
    model_path: str = None  # Path to trained model checkpoint
    
    # Estimation parameters
    n_samples_grad: int = 1000      # Number of samples for gradient-based estimate
    n_samples_fd: int = 500         # Number of samples for finite-difference
    epsilon_fd: float = 0.01        # Perturbation size for finite-difference
    n_pairs_fd: int = 10            # Number of random directions per sample
    
    # Global spectral norm
    power_iter_steps: int = 20
    
    batch_size: int = 256
    device: str = "cuda"


def load_model_and_data(config: LipschitzConfig) -> Tuple[nn.Module, DataLoader]:
    """Load trained model and test dataset."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    if config.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        num_classes = 10
        dataset_class = datasets.CIFAR10
    elif config.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = dataset_class("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=4)
    
    # Load model (or create a fresh one for demonstration)
    model = ResNet18Width(width_factor=1.0, num_classes=num_classes).to(device)
    
    if config.model_path and Path(config.model_path).exists():
        model.load_state_dict(torch.load(config.model_path, map_location=device))
        print(f"Loaded model from {config.model_path}")
    else:
        print("Using randomly initialized model (for demonstration)")
    
    model.eval()
    return model, test_loader


def estimate_global_spectral(model: nn.Module, n_iter: int = 20) -> float:
    """
    Method 1: Global Lipschitz via product of layer spectral norms.
    This is the loosest but fastest estimate.
    """
    L = 1.0
    layer_norms = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight'):
                sigma = power_iteration(module.weight.data, n_iter)
                L *= sigma
                layer_norms.append((name, sigma))
    
    return L, layer_norms


def estimate_gradient_based(model: nn.Module, loader: DataLoader, 
                            n_samples: int, device: torch.device) -> Tuple[float, List[float]]:
    """
    Method 2: Gradient-based local Lipschitz estimate.
    
    L_local(x) ≈ ||∇_x ℓ(f(x), y)||_2
    L_avg = E_{(x,y)~D}[L_local(x)]
    
    This estimates the average local Lipschitz constant over the data distribution.
    """
    model.eval()
    local_lipschitz_values = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    count = 0
    
    for images, labels in loader:
        if count >= n_samples:
            break
        
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        
        # Forward pass
        outputs = model(images)
        losses = criterion(outputs, labels)
        
        # Compute gradients
        for i in range(min(images.size(0), n_samples - count)):
            if images.grad is not None:
                images.grad.zero_()
            
            losses[i].backward(retain_graph=(i < images.size(0) - 1))
            
            # L_local = ||∇_x ℓ||
            grad_norm = images.grad[i].norm().item()
            local_lipschitz_values.append(grad_norm)
            count += 1
            
            if count >= n_samples:
                break
    
    L_avg = np.mean(local_lipschitz_values)
    return L_avg, local_lipschitz_values


def estimate_finite_difference(model: nn.Module, loader: DataLoader,
                               n_samples: int, epsilon: float, n_pairs: int,
                               device: torch.device) -> Tuple[float, List[float]]:
    """
    Method 3: Finite-difference sampling.
    
    For each sample x, generate random perturbations δ with ||δ|| = ε,
    compute L_local(x) ≈ max_δ |ℓ(f(x+δ), y) - ℓ(f(x), y)| / ε
    
    L_avg = E_x[L_local(x)]
    """
    model.eval()
    local_lipschitz_values = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    count = 0
    
    with torch.no_grad():
        for images, labels in loader:
            if count >= n_samples:
                break
            
            images, labels = images.to(device), labels.to(device)
            
            # Original loss
            outputs_orig = model(images)
            loss_orig = criterion(outputs_orig, labels)
            
            for i in range(min(images.size(0), n_samples - count)):
                max_diff = 0.0
                
                # Try multiple random directions
                for _ in range(n_pairs):
                    # Random perturbation with L2 norm = epsilon
                    delta = torch.randn_like(images[i:i+1])
                    delta = delta / (delta.norm() + 1e-12) * epsilon
                    
                    # Perturbed loss
                    perturbed = torch.clamp(images[i:i+1] + delta, 0, 1)
                    outputs_pert = model(perturbed)
                    loss_pert = criterion(outputs_pert, labels[i:i+1])
                    
                    # Lipschitz estimate
                    diff = torch.abs(loss_pert - loss_orig[i]).item()
                    max_diff = max(max_diff, diff / epsilon)
                
                local_lipschitz_values.append(max_diff)
                count += 1
                
                if count >= n_samples:
                    break
    
    L_avg = np.mean(local_lipschitz_values)
    return L_avg, local_lipschitz_values


def analyze_lipschitz_surrogates(config: LipschitzConfig) -> Dict[str, Any]:
    """Run all three Lipschitz estimation methods and compare."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Lipschitz Surrogate Analysis: {config.dataset}")
    print(f"{'='*70}\n")
    
    # Load model and data
    model, test_loader = load_model_and_data(config)
    
    # Method 1: Global spectral norm
    print("Method 1: Global Spectral Norm Product...")
    L_global, layer_norms = estimate_global_spectral(model, config.power_iter_steps)
    print(f"  L_global = {L_global:.4e}")
    print(f"  Number of layers: {len(layer_norms)}")
    print(f"  Layer norms range: [{min(s for _, s in layer_norms):.2f}, "
          f"{max(s for _, s in layer_norms):.2f}]")
    
    # Method 2: Gradient-based
    print(f"\nMethod 2: Gradient-Based Local Average (n={config.n_samples_grad})...")
    L_grad, grad_values = estimate_gradient_based(
        model, test_loader, config.n_samples_grad, device
    )
    print(f"  L_avg_grad = {L_grad:.4e}")
    print(f"  Std: {np.std(grad_values):.4e}")
    print(f"  Min/Max: [{np.min(grad_values):.4e}, {np.max(grad_values):.4e}]")
    
    # Method 3: Finite-difference
    print(f"\nMethod 3: Finite-Difference Sampling (n={config.n_samples_fd}, "
          f"epsilon={config.epsilon_fd})...")
    L_fd, fd_values = estimate_finite_difference(
        model, test_loader, config.n_samples_fd, config.epsilon_fd, 
        config.n_pairs_fd, device
    )
    print(f"  L_avg_fd = {L_fd:.4e}")
    print(f"  Std: {np.std(fd_values):.4e}")
    print(f"  Min/Max: [{np.min(fd_values):.4e}, {np.max(fd_values):.4e}]")
    
    # Comparison
    print(f"\n{'='*70}")
    print("Comparison:")
    print(f"  L_global / L_grad = {L_global / L_grad:.2e}")
    print(f"  L_global / L_fd   = {L_global / L_fd:.2e}")
    print(f"  L_grad / L_fd     = {L_grad / L_fd:.2f}")
    print(f"{'='*70}")
    
    results = {
        "dataset": config.dataset,
        "global_spectral": {
            "L_estimate": float(L_global),
            "n_layers": len(layer_norms),
            "layer_norms": [(name, float(s)) for name, s in layer_norms]
        },
        "gradient_based": {
            "L_estimate": float(L_grad),
            "mean": float(np.mean(grad_values)),
            "std": float(np.std(grad_values)),
            "min": float(np.min(grad_values)),
            "max": float(np.max(grad_values)),
            "n_samples": len(grad_values)
        },
        "finite_difference": {
            "L_estimate": float(L_fd),
            "mean": float(np.mean(fd_values)),
            "std": float(np.std(fd_values)),
            "min": float(np.min(fd_values)),
            "max": float(np.max(fd_values)),
            "n_samples": len(fd_values),
            "epsilon": config.epsilon_fd
        },
        "ratios": {
            "global_over_grad": float(L_global / L_grad) if L_grad > 0 else float('inf'),
            "global_over_fd": float(L_global / L_fd) if L_fd > 0 else float('inf'),
            "grad_over_fd": float(L_grad / L_fd) if L_fd > 0 else 1.0
        }
    }
    
    return results


def main():
    """Run Lipschitz surrogate analysis experiments."""
    output_dir = Path(__file__).parent
    
    # Configurations for different datasets
    configs = [
        LipschitzConfig(dataset="cifar10", n_samples_grad=1000, n_samples_fd=500),
        # Add more configurations as needed
    ]
    
    all_results = []
    for i, config in enumerate(configs):
        print(f"\n\n{'#'*70}")
        print(f"# Experiment {i+1}/{len(configs)}")
        print(f"{'#'*70}")
        
        result = analyze_lipschitz_surrogates(config)
        all_results.append(result)
    
    # Save results
    output_file = output_dir / "lipschitz_surrogate_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print(f"Lipschitz surrogate analysis completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
    print("\nKey Findings:")
    for result in all_results:
        print(f"\n{result['dataset']}:")
        print(f"  Global spectral: {result['global_spectral']['L_estimate']:.2e}")
        print(f"  Gradient-based:  {result['gradient_based']['L_estimate']:.2e}")
        print(f"  Finite-diff:     {result['finite_difference']['L_estimate']:.2e}")
        print(f"  Tightness improvement: {result['ratios']['global_over_grad']:.1f}x")


if __name__ == "__main__":
    main()

