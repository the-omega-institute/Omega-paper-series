from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import os
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Adjust import path for module execution or script execution
try:
    from src.bounds.pacbayes_bound import pac_bayes_bound, PacBayesConfig
    from src.bounds.lipschitz_estimate import estimate_network_lipschitz
except ImportError:
    import sys
    # Add jmlr_reproduction root to sys.path, prioritizing it over project root src
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.bounds.pacbayes_bound import pac_bayes_bound, PacBayesConfig
    from src.bounds.lipschitz_estimate import estimate_network_lipschitz

@dataclass
class FrontierConfig:
    model_name: str = "resnet18"
    width_factor: float = 1.0
    lr: float = 0.1
    weight_decay: float = 5e-4
    batch_size: int = 128
    epochs: int = 1 # Reduced for demo purposes, originally 50
    device: str = "cuda"
    corruption_severity: int = 3
    pac_delta: float = 1e-3
    prior_variance: float = 1.0

def get_cifar10_loaders(batch_size: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download to a local data folder
    data_dir = os.path.join(os.path.dirname(__file__), "../../data")
    os.makedirs(data_dir, exist_ok=True)

    train_ds = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=transform_train)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True,
                               transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0) # num_workers=0 for windows compatibility
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader

def build_model(cfg: FrontierConfig) -> nn.Module:
    """
    这里用 torchvision 的 resnet18 做演示.
    width_factor 可以通过修改 channel 宽度来实现.
    真实实验里你可以写一个自定义 small resnet.
    """
    if cfg.model_name == "resnet18":
        # CIFAR-10 has 10 classes. 
        # We modify the first conv layer to handle 32x32 images better (kernel size 3, stride 1, padding 1)
        # and remove maxpool, as is common for CIFAR ResNets.
        model = models.resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")

    return model

def train_model(cfg: FrontierConfig) -> nn.Module:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    train_loader, _ = get_cifar10_loaders(cfg.batch_size)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.lr,
                          momentum=0.9,
                          weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"  Step {i}, Loss: {loss.item():.4f}")

    return model

def eval_clean_error(model: nn.Module, batch_size: int = 256) -> float:
    device = next(model.parameters()).device
    _, test_loader = get_cifar10_loaders(batch_size)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 1.0 - correct / total

def eval_corruption_error(
    model: nn.Module,
    corruption: str = "gaussian_noise",
    severity: int = 3,
    batch_size: int = 256,
) -> float:
    """
    这里先做一个占位版本: 可以用 CIFAR10 原数据并加上简单噪声.
    真正 ImageNet-C / CIFAR10-C 可以用已有数据集替换.
    """
    device = next(model.parameters()).device

    base_transform = transforms.ToTensor()
    data_dir = os.path.join(os.path.dirname(__file__), "../../data")
    clean_test = datasets.CIFAR10(data_dir, train=False, download=True,
                                  transform=base_transform)
    # 简单包装: 在线加噪
    def corrupt(x):
        if corruption == "gaussian_noise":
            noise = torch.randn_like(x) * 0.05 * severity # Adjusted scale
            return torch.clamp(x + noise, 0.0, 1.0)
        else:
            return x

    loader = DataLoader(clean_test, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = corrupt(x)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 1.0 - correct / total

def run_one_config(cfg: FrontierConfig) -> Dict[str, Any]:
    print(f"Running config: {cfg}")
    model = train_model(cfg)
    clean_err = eval_clean_error(model, batch_size=256)
    rob_err = eval_corruption_error(
        model,
        corruption="gaussian_noise",
        severity=cfg.corruption_severity,
        batch_size=256,
    )

    L_est = estimate_network_lipschitz(model)
    pac_cfg = PacBayesConfig(delta=cfg.pac_delta,
                             prior_variance=cfg.prior_variance)
    n_test = 10000  # CIFAR-10 测试集大小
    bound = pac_bayes_bound(clean_err, model, n_test, pac_cfg)
    
    result = {
        "config": asdict(cfg),
        "clean_error": clean_err,
        "robust_error": rob_err,
        "L_est": L_est,
        "pacbayes_bound": bound,
        "bound_plus_Lrho": bound + L_est * (0.05 * cfg.corruption_severity), 
    }
    print(f"Result: {result}")
    return result

def run_frontier_experiments() -> List[Dict[str, Any]]:
    # Reduced epochs and configs for demo speed
    cfgs = [
        FrontierConfig(width_factor=0.5, weight_decay=5e-3, epochs=1), 
        FrontierConfig(width_factor=1.0, weight_decay=1e-3, epochs=1),
        FrontierConfig(width_factor=2.0, weight_decay=1e-4, epochs=1),
    ]
    results = [run_one_config(cfg) for cfg in cfgs]
    
    output_path = os.path.join(os.path.dirname(__file__), "../../cifar_frontier_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return results

if __name__ == "__main__":
    run_frontier_experiments()

