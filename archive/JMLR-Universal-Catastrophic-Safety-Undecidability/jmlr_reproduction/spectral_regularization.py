"""
Spectral Norm Regularization for Lipschitz-Constrained Training
Implements explicit spectral normalization as a robust training method.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import sys


sys.path.append(str(Path(__file__).parent))
from cifar_capability_risk import estimate_lipschitz, flatten_params


@dataclass
class SpectralRegConfig:
    """Configuration for spectral norm regularization experiments."""
    dataset: str = "cifar10"
    width_factor: float = 1.0
    lr: float = 0.1
    weight_decay: float = 5e-4
    batch_size: int = 128
    epochs: int = 200
    momentum: float = 0.9
    
    # Spectral norm parameters
    use_spectral_norm: bool = True
    n_power_iterations: int = 1  # For spectral_norm layer
    
    # Evaluation
    noise_levels: List[float] = None
    power_iter_steps: int = 20
    pac_delta: float = 0.01
    prior_variance: float = 1.0
    device: str = "cuda"
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.25, 0.5]


class SpectralResNet18(nn.Module):
    """ResNet-18 with spectral normalization on all linear/conv layers."""
    
    def __init__(self, width_factor: float = 1.0, num_classes: int = 10,
                 use_spectral_norm: bool = True, n_power_iterations: int = 1):
        super().__init__()
        self.width_factor = width_factor
        self.use_sn = use_spectral_norm
        base_width = int(64 * width_factor)
        
        # Apply spectral norm wrapper
        def maybe_spectral_norm(layer):
            if self.use_sn and isinstance(layer, (nn.Conv2d, nn.Linear)):
                return spectral_norm(layer, n_power_iterations=n_power_iterations)
            return layer
        
        self.conv1 = maybe_spectral_norm(
            nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        
        # Build layers
        self.layer1 = self._make_layer(base_width, base_width, 2, stride=1, maybe_sn=maybe_spectral_norm)
        self.layer2 = self._make_layer(base_width, base_width * 2, 2, stride=2, maybe_sn=maybe_spectral_norm)
        self.layer3 = self._make_layer(base_width * 2, base_width * 4, 2, stride=2, maybe_sn=maybe_spectral_norm)
        self.layer4 = self._make_layer(base_width * 4, base_width * 8, 2, stride=2, maybe_sn=maybe_spectral_norm)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = maybe_spectral_norm(nn.Linear(base_width * 8, num_classes))
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, maybe_sn):
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride, maybe_sn))
        for _ in range(1, num_blocks):
            layers.append(self._residual_block(out_channels, out_channels, 1, maybe_sn))
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels, out_channels, stride, maybe_sn):
        class BasicBlock(nn.Module):
            def __init__(self, in_ch, out_ch, s, maybe_sn_fn):
                super().__init__()
                self.conv1 = maybe_sn_fn(
                    nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=1, bias=False)
                )
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = maybe_sn_fn(
                    nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
                )
                self.bn2 = nn.BatchNorm2d(out_ch)
                
                # Shortcut connection
                self.shortcut = nn.Sequential()
                if s != 1 or in_ch != out_ch:
                    self.shortcut = nn.Sequential(
                        maybe_sn_fn(nn.Conv2d(in_ch, out_ch, 1, stride=s, bias=False)),
                        nn.BatchNorm2d(out_ch)
                    )
            
            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = self.relu(out)
                return out
        
        return BasicBlock(in_channels, out_channels, stride, maybe_sn)
    
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
    
    # Set num_workers=0 for Windows compatibility (avoids shared memory errors)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    
    return train_loader, test_loader, num_classes


def train_one_epoch(model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    return avg_loss, train_acc


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


def train_spectral_model(config: SpectralRegConfig) -> Dict[str, Any]:
    """Train a model with spectral normalization."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Spectral Norm Training: {config.dataset}, width={config.width_factor}, "
          f"SN={'ON' if config.use_spectral_norm else 'OFF'}")
    print(f"{'='*70}")
    
    # Data
    train_loader, test_loader, num_classes = get_data_loaders(config.dataset, config.batch_size)
    
    # Model with spectral normalization
    model = SpectralResNet18(
        width_factor=config.width_factor,
        num_classes=num_classes,
        use_spectral_norm=config.use_spectral_norm,
        n_power_iterations=config.n_power_iterations
    ).to(device)
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr,
                         momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Training
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] "
                  f"Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Evaluation
    print("\nEvaluating trained model...")
    
    # Clean error
    clean_error = evaluate_clean(model, test_loader, device)
    print(f"Clean test error: {clean_error:.4f}")
    
    # Gaussian noise robust errors
    robust_errors = {}
    for rho in config.noise_levels:
        if rho > 0:
            rob_err = evaluate_gaussian_robust(model, test_loader, device, rho)
            robust_errors[f"rho_{rho}"] = rob_err
            print(f"Robust error (ρ={rho}): {rob_err:.4f}")
    
    # Lipschitz constant estimation
    L_est = estimate_lipschitz(model, config.power_iter_steps)
    print(f"Estimated Lipschitz constant: {L_est:.2e}")
    
    # PAC-Bayes bound
    n_train = 50000
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
        "method": "Spectral-Norm" if config.use_spectral_norm else "Baseline",
        "config": asdict(config),
        "clean_error": clean_error,
        "robust_errors": robust_errors,
        "lipschitz_constant": L_est,
        "pac_bayes_bound": pac_bound,
        "capability_risk_bounds": bounds
    }


def main():
    """Run spectral normalization experiments."""
    output_dir = Path(__file__).parent
    
    # Experiment configurations
    configs = [
        # Baseline (no spectral norm)
        SpectralRegConfig(dataset="cifar10", width_factor=1.0, use_spectral_norm=False),
        
        # With spectral norm
        SpectralRegConfig(dataset="cifar10", width_factor=1.0, use_spectral_norm=True),
        SpectralRegConfig(dataset="cifar10", width_factor=2.0, use_spectral_norm=True),
        
        # CIFAR-100
        SpectralRegConfig(dataset="cifar100", width_factor=1.0, use_spectral_norm=True),
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n\n{'#'*70}")
        print(f"# Experiment {i+1}/{len(configs)}")
        print(f"{'#'*70}")
        
        result = train_spectral_model(config)
        results.append(result)
    
    # Save results
    output_file = output_dir / "spectral_norm_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print(f"Spectral norm experiments completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

