"""
PGD Adversarial Training for Capability-Risk Frontier Analysis
Implements PGD-AT as a robust training baseline for comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import sys


# Import ResNet from cifar_capability_risk
sys.path.append(str(Path(__file__).parent))
from cifar_capability_risk import ResNet18Width, estimate_lipschitz, flatten_params


@dataclass
class PGDATConfig:
    """Configuration for PGD Adversarial Training."""
    dataset: str = "cifar10"  # "cifar10" or "cifar100"
    width_factor: float = 1.0
    lr: float = 0.1
    weight_decay: float = 5e-4
    batch_size: int = 64  # Reduced from 128 to avoid OOM during PGD attack
    epochs: int = 200
    momentum: float = 0.9
    
    # PGD attack parameters
    epsilon: float = 8.0 / 255.0  # L-infinity perturbation budget
    alpha: float = 2.0 / 255.0     # PGD step size
    num_steps: int = 10            # PGD iterations
    
    # Evaluation
    noise_levels: List[float] = None
    power_iter_steps: int = 20
    pac_delta: float = 0.01
    prior_variance: float = 1.0
    device: str = "cuda"
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.25, 0.5]


def get_data_loaders(dataset: str, batch_size: int):
    """Get data loaders for CIFAR-10 or CIFAR-100."""
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        num_classes = 10
        dataset_class = datasets.CIFAR10
    elif dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = dataset_class("./data", train=True, download=True, transform=transform_train)
    test_dataset = dataset_class("./data", train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, num_classes


def pgd_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
              epsilon: float, alpha: float, num_steps: int,
              lower_bound: float = 0.0, upper_bound: float = 1.0) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.
    
    Args:
        model: Target model
        images: Clean images
        labels: True labels
        epsilon: L-infinity perturbation budget
        alpha: Step size
        num_steps: Number of PGD iterations
        lower_bound, upper_bound: Valid pixel range after normalization
    
    Returns:
        Adversarial examples
    """
    model.eval()
    
    # Start from random perturbation
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    
    for _ in range(num_steps):
        outputs = model(images + delta)
        loss = F.cross_entropy(outputs, labels)
        
        loss.backward()
        
        # Gradient ascent step
        grad_sign = delta.grad.data.sign()
        delta.data = delta.data + alpha * grad_sign
        
        # Project back to epsilon ball
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
        # Ensure valid pixel range
        delta.data = torch.clamp(images + delta.data, lower_bound, upper_bound) - images
        
        delta.grad.zero_()
    
    adv_images = images + delta.detach()
    return adv_images


def train_with_pgd(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                  criterion: nn.Module, device: torch.device, config: PGDATConfig) -> float:
    """Train one epoch with PGD adversarial training."""
    model.train()
    total_loss = 0.0
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = pgd_attack(model, images, labels, config.epsilon, 
                               config.alpha, config.num_steps)
        
        # Train on adversarial examples
        model.train()
        optimizer.zero_grad()
        
        outputs_adv = model(adv_images)
        loss = criterion(outputs_adv, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Track accuracy
        with torch.no_grad():
            outputs_clean = model(images)
            _, pred_clean = outputs_clean.max(1)
            _, pred_adv = outputs_adv.max(1)
            
            total += labels.size(0)
            correct_clean += pred_clean.eq(labels).sum().item()
            correct_adv += pred_adv.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    clean_acc = 100.0 * correct_clean / total
    adv_acc = 100.0 * correct_adv / total
    
    return avg_loss, clean_acc, adv_acc


def evaluate_clean(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate clean accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    error = 1.0 - (correct / total)
    return error


def evaluate_pgd_robust(model: nn.Module, loader: DataLoader, device: torch.device,
                       epsilon: float, alpha: float, num_steps: int) -> float:
    """Evaluate robust accuracy against PGD attack."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    error = 1.0 - (correct / total)
    return error


def add_gaussian_noise(images: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Add Gaussian noise for distribution shift evaluation."""
    noise = torch.randn_like(images) * noise_level
    noisy_images = torch.clamp(images + noise, 0.0, 1.0)
    return noisy_images


def evaluate_gaussian_robust(model: nn.Module, loader: DataLoader, 
                             device: torch.device, noise_level: float) -> float:
    """Evaluate robust accuracy under Gaussian noise."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            noisy_images = add_gaussian_noise(images, noise_level)
            outputs = model(noisy_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    error = 1.0 - (correct / total)
    return error


def pac_bayes_bound(empirical_error: float, model: nn.Module, 
                   n_samples: int, delta: float, prior_variance: float) -> float:
    """Compute PAC-Bayes generalization bound."""
    params = flatten_params(model)
    l2_norm_sq = float(params.pow(2).sum().item())
    kl_term = 0.5 * l2_norm_sq / prior_variance
    
    bound_term = np.sqrt((kl_term + np.log(2.0 / delta)) / (2.0 * n_samples))
    return empirical_error + bound_term


def train_pgd_at_model(config: PGDATConfig) -> Dict[str, Any]:
    """Train a model with PGD adversarial training."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print(f"PGD-AT Training: {config.dataset}, width={config.width_factor}, ε={config.epsilon:.3f}")
    print(f"{'='*70}")
    
    # Data
    train_loader, test_loader, num_classes = get_data_loaders(config.dataset, config.batch_size)
    
    # Model
    model = ResNet18Width(width_factor=config.width_factor, num_classes=num_classes).to(device)
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr,
                         momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Training
    for epoch in range(config.epochs):
        train_loss, clean_acc, adv_acc = train_with_pgd(
            model, train_loader, optimizer, criterion, device, config
        )
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] "
                  f"Loss: {train_loss:.4f}, "
                  f"Clean Acc: {clean_acc:.2f}%, "
                  f"Adv Acc: {adv_acc:.2f}%, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Evaluation
    print("\nEvaluating trained model...")
    
    # Clean error
    clean_error = evaluate_clean(model, test_loader, device)
    print(f"Clean test error: {clean_error:.4f}")
    
    # PGD robust error
    pgd_robust_error = evaluate_pgd_robust(model, test_loader, device,
                                           config.epsilon, config.alpha, config.num_steps)
    print(f"PGD robust error (ε={config.epsilon:.3f}): {pgd_robust_error:.4f}")
    
    # Gaussian noise robust errors
    gaussian_robust_errors = {}
    for rho in config.noise_levels:
        if rho > 0:
            rob_err = evaluate_gaussian_robust(model, test_loader, device, rho)
            gaussian_robust_errors[f"rho_{rho}"] = rob_err
            print(f"Gaussian robust error (ρ={rho}): {rob_err:.4f}")
    
    # Lipschitz constant
    L_est = estimate_lipschitz(model, config.power_iter_steps)
    print(f"Estimated Lipschitz constant: {L_est:.2e}")
    
    # PAC-Bayes bound
    n_train = 50000 if config.dataset == "cifar10" else 50000
    pac_bound = pac_bayes_bound(clean_error, model, n_train, 
                               config.pac_delta, config.prior_variance)
    print(f"PAC-Bayes bound: {pac_bound:.4f}")
    
    # Capability-risk bounds
    bounds = {}
    for rho in config.noise_levels:
        if rho > 0:
            combined_bound = pac_bound + L_est * rho
            bounds[f"bound_rho_{rho}"] = min(combined_bound, 1.0)
            print(f"Capability-Risk bound (ρ={rho}): {bounds[f'bound_rho_{rho}']:.4f}")
    
    return {
        "method": "PGD-AT",
        "config": asdict(config),
        "clean_error": clean_error,
        "pgd_robust_error": pgd_robust_error,
        "gaussian_robust_errors": gaussian_robust_errors,
        "lipschitz_constant": L_est,
        "pac_bayes_bound": pac_bound,
        "capability_risk_bounds": bounds
    }


def main():
    """Run PGD adversarial training experiments."""
    output_dir = Path(__file__).parent
    
    # Experiment configurations
    configs = [
        # CIFAR-10 experiments
        PGDATConfig(dataset="cifar10", width_factor=1.0, epsilon=8.0/255.0),
        PGDATConfig(dataset="cifar10", width_factor=2.0, epsilon=8.0/255.0),
        
        # CIFAR-100 experiment
        PGDATConfig(dataset="cifar100", width_factor=1.0, epsilon=8.0/255.0, epochs=200),
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n\n{'#'*70}")
        print(f"# Experiment {i+1}/{len(configs)}")
        print(f"{'#'*70}")
        
        result = train_pgd_at_model(config)
        results.append(result)
    
    # Save results
    output_file = output_dir / "pgd_at_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print(f"PGD-AT experiments completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

