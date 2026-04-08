# cifar_capability_risk.py
import math
import argparse
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
import json

# --------------------------
# Model: width-scaled ResNet-18
# --------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet18Width(nn.Module):
    def __init__(self, num_classes=10, width_factor=1.0):
        super().__init__()
        base_channels = int(64 * width_factor)
        self.inplanes = base_channels
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)  # CIFAR uses 3x3 conv
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()  # no initial maxpool for CIFAR
        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes,
                                 stride=stride, downsample=downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --------------------------
# Lipschitz estimation by spectral norm
# --------------------------
def power_iteration(W, num_iters=20, eps=1e-6, device=None):
    """Approximate spectral norm of matrix W by power iteration."""
    if device is None:
        device = W.device
    
    # Handle 4D convolutional weights (out_ch, in_ch, kh, kw)
    if W.ndim == 4:
        out_channels, in_channels, kh, kw = W.shape
        W = W.reshape(out_channels, in_channels * kh * kw)
    elif W.ndim != 2:
        raise ValueError(f"Expected 2D or 4D tensor, got {W.ndim}D")
    
    out_dim, in_dim = W.shape
    u = torch.randn(out_dim, device=device)
    u = u / (u.norm() + eps)
    for _ in range(num_iters):
        v = W.t().mv(u)
        v = v / (v.norm() + eps)
        u = W.mv(v)
        u = u / (u.norm() + eps)
    sigma = u.dot(W.mv(v))
    return sigma.abs().item()

def estimate_lipschitz(model, num_iters=20):
    """Very crude upper bound: product of per-layer spectral norms."""
    model.eval()
    L = 1.0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            W = m.weight.detach()
            out_c, in_c, kh, kw = W.shape
            W_flat = W.view(out_c, -1)
            sigma = power_iteration(W_flat, num_iters=num_iters,
                                    device=W.device)
            L *= sigma
        elif isinstance(m, nn.Linear):
            W = m.weight.detach()
            sigma = power_iteration(W, num_iters=num_iters,
                                    device=W.device)
            L *= sigma
    return L

# --------------------------
# PAC-Bayes-like bound
# --------------------------
def compute_empirical_error(model, loader, device):
    model.eval()
    total = 0
    wrong = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            wrong += (preds != y).sum().item()
            total += y.numel()
    return wrong / total

def flatten_params(model, device):
    params = []
    for p in model.parameters():
        params.append(p.detach().view(-1).to(device))
    if not params:
        return torch.zeros(1, device=device)
    return torch.cat(params, dim=0)

def pac_bayes_bound(train_error, model, n_samples,
                    delta=0.01, sigma_p=0.1, device="cpu"):
    """Simple Gaussian prior/posterior approximation."""
    w = flatten_params(model, device)
    kl = (w.pow(2).sum() / (2 * sigma_p ** 2)).item()
    bound = train_error + math.sqrt((kl + math.log(1.0 / delta)) /
                                    (2.0 * n_samples))
    return bound, kl

# --------------------------
# Training and evaluation
# --------------------------
@dataclass
class RunResult:
    width_factor: float
    rho: float
    L: float
    train_error: float
    test_error: float
    robust_error: float
    pac_bayes_bound: float
    cap_risk_bound: float
    kl_value: float

def add_gaussian_noise(x, sigma):
    noise = torch.randn_like(x) * sigma
    x_noisy = torch.clamp(x + noise, 0.0, 1.0)
    return x_noisy

def compute_robust_error(model, loader, device, sigma):
    model.eval()
    total = 0
    wrong = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_noisy = add_gaussian_noise(x, sigma)
            logits = model(x_noisy)
            preds = logits.argmax(dim=1)
            wrong += (preds != y).sum().item()
            total += y.numel()
    return wrong / total

def train_one_model(args, width_factor, device):
    # data
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
    ])
    train_set = torchvision.datasets.CIFAR10(root=args.data_dir,
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=args.data_dir,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    model = ResNet18Width(num_classes=10, width_factor=width_factor)
    model.to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    n_train = len(train_set)
    best_test_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        test_err = compute_empirical_error(model, test_loader, device)
        test_acc = 1.0 - test_err
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
            }
        print(f"Epoch {epoch}: test_acc={test_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    
    results = []
    train_clean_set = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=transform_test,
    )
    train_clean_loader = DataLoader(train_clean_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True)
    train_error = compute_empirical_error(model,
                                          train_clean_loader,
                                          device)
    test_error = compute_empirical_error(model,
                                         test_loader,
                                         device)
    L_est = estimate_lipschitz(model, num_iters=20)
    for rho, sigma in zip(args.rhos, args.noise_sigmas):
        robust_error = compute_robust_error(model,
                                            test_loader,
                                            device,
                                            sigma=sigma)
        pac_bound, kl_val = pac_bayes_bound(train_error,
                                            model,
                                            n_samples=n_train,
                                            delta=args.delta,
                                            sigma_p=args.sigma_p,
                                            device=device)
        cap_risk_bound = pac_bound + L_est * rho
        res = RunResult(
            width_factor=width_factor,
            rho=rho,
            L=L_est,
            train_error=train_error,
            test_error=test_error,
            robust_error=robust_error,
            pac_bayes_bound=pac_bound,
            cap_risk_bound=cap_risk_bound,
            kl_value=kl_val,
        )
        results.append(res)
        print(f"[w={width_factor}, rho={rho}] "
              f"test_err={test_error:.4f}, "
              f"robust_err={robust_error:.4f}, "
              f"bound={cap_risk_bound:.4f}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="cifar_results.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--width-factors", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--noise-sigmas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--sigma-p", type=float, default=0.1)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    all_results = []
    for w in args.width_factors:
        results = train_one_model(args, width_factor=w, device=device)
        all_results.extend(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

if __name__ == "__main__":
    main()

