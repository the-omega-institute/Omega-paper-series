"""
CIFAR-100 Capability-Risk Frontier Experiment
Extends CIFAR-10 analysis to 100-class setting for cross-dataset validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass
class CIFAR100Config:
    """Configuration for CIFAR-100 experiments."""
    width_factor: float = 1.0
    lr: float = 0.1
    weight_decay: float = 5e-4
    batch_size: int = 128
    epochs: int = 200
    momentum: float = 0.9
    device: str = "cuda"
    noise_levels: List[float] = None
    power_iter_steps: int = 20
    pac_delta: float = 0.01
    prior_variance: float = 1.0
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.25, 0.5]


class ResNet18CIFAR100(nn.Module):
    """Width-scaled ResNet-18 for CIFAR-100 (100 classes)."""
    
    def __init__(self, width_factor: float = 1.0):
        super().__init__()
        self.width_factor = width_factor
        base_width = int(64 * width_factor)
        
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        
        # 4 residual blocks
        self.layer1 = self._make_layer(base_width, base_width, 2, stride=1)
        self.layer2 = self._make_layer(base_width, base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_width * 2, base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_width * 4, base_width * 8, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8, 100)  # 100 classes for CIFAR-100
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(self._residual_block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_cifar100_loaders(batch_size: int, data_dir: str = "./data"):
    """Get CIFAR-100 train and test loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    return train_loader, test_loader


def power_iteration(weight: torch.Tensor, n_iter: int = 20) -> float:
    """Estimate spectral norm via power iteration."""
    if len(weight.shape) == 4:  # Conv layer
        weight = weight.view(weight.size(0), -1)
    elif len(weight.shape) != 2:
        return 1.0
    
    device = weight.device
    u = torch.randn(weight.size(0), device=device)
    u = u / (u.norm() + 1e-12)
    
    for _ in range(n_iter):
        v = torch.mv(weight.t(), u)
        v = v / (v.norm() + 1e-12)
        u = torch.mv(weight, v)
        u = u / (u.norm() + 1e-12)
    
    sigma = torch.dot(u, torch.mv(weight, v))
    return float(sigma.abs().item())


def estimate_lipschitz(model: nn.Module, n_iter: int = 20) -> float:
    """Estimate network Lipschitz constant as product of layer spectral norms."""
    L = 1.0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight'):
                sigma = power_iteration(module.weight.data, n_iter)
                L *= sigma
    return L


def compute_empirical_error(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification error on a dataset."""
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


def add_gaussian_noise(images: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Add Gaussian noise to simulate distribution shift."""
    noise = torch.randn_like(images) * noise_level
    noisy_images = torch.clamp(images + noise, 0.0, 1.0)
    return noisy_images


def compute_robust_error(model: nn.Module, loader: DataLoader, 
                        noise_level: float, device: torch.device) -> float:
    """Compute robust error under Gaussian noise corruption."""
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


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.view(-1) for p in model.parameters()])


def pac_bayes_bound(empirical_error: float, model: nn.Module, 
                   n_samples: int, delta: float, prior_variance: float) -> float:
    """Compute PAC-Bayes generalization bound."""
    params = flatten_params(model)
    l2_norm_sq = float(params.pow(2).sum().item())
    kl_term = 0.5 * l2_norm_sq / prior_variance
    
    bound_term = np.sqrt((kl_term + np.log(2.0 / delta)) / (2.0 * n_samples))
    return empirical_error + bound_term


def train_one_model(config: CIFAR100Config) -> Dict[str, Any]:
    """Train a single model and compute all metrics."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training CIFAR-100 model: width={config.width_factor}, wd={config.weight_decay}")
    print(f"{'='*60}")
    
    # Data
    train_loader, test_loader = get_cifar100_loaders(config.batch_size)
    
    # Model
    model = ResNet18CIFAR100(config.width_factor).to(device)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, 
                         momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            avg_loss = train_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Evaluation
    print("\nEvaluating model...")
    clean_error = compute_empirical_error(model, test_loader, device)
    print(f"Clean test error: {clean_error:.4f}")
    
    # Robust errors for different noise levels
    robust_errors = {}
    for rho in config.noise_levels:
        if rho > 0:
            rob_err = compute_robust_error(model, test_loader, rho, device)
            robust_errors[f"rho_{rho}"] = rob_err
            print(f"Robust error (ρ={rho}): {rob_err:.4f}")
    
    # Lipschitz constant
    L_est = estimate_lipschitz(model, config.power_iter_steps)
    print(f"Estimated Lipschitz constant: {L_est:.2e}")
    
    # PAC-Bayes bound
    n_train = 50000  # CIFAR-100 training set size
    pac_bound = pac_bayes_bound(clean_error, model, n_train, config.pac_delta, config.prior_variance)
    print(f"PAC-Bayes bound: {pac_bound:.4f}")
    
    # Combined capability-risk bounds
    bounds = {}
    for rho in config.noise_levels:
        if rho > 0:
            combined_bound = pac_bound + L_est * rho
            bounds[f"bound_rho_{rho}"] = min(combined_bound, 1.0)  # Cap at 1.0
            print(f"Capability-Risk bound (ρ={rho}): {bounds[f'bound_rho_{rho}']:.4f}")
    
    return {
        "config": asdict(config),
        "clean_error": clean_error,
        "robust_errors": robust_errors,
        "lipschitz_constant": L_est,
        "pac_bayes_bound": pac_bound,
        "capability_risk_bounds": bounds
    }


def main():
    """Run CIFAR-100 frontier experiments."""
    output_dir = Path(__file__).parent
    
    # Experiment configurations
    configs = [
        CIFAR100Config(width_factor=0.5, weight_decay=5e-4, epochs=200),
        CIFAR100Config(width_factor=1.0, weight_decay=5e-4, epochs=200),
        CIFAR100Config(width_factor=2.0, weight_decay=5e-4, epochs=200),
        # Additional config with stronger regularization
        CIFAR100Config(width_factor=1.0, weight_decay=1e-3, epochs=200),
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n\n{'#'*60}")
        print(f"# Experiment {i+1}/{len(configs)}")
        print(f"{'#'*60}")
        
        result = train_one_model(config)
        results.append(result)
    
    # Save results
    output_file = output_dir / "cifar100_frontier_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

