from __future__ import annotations
from torch import nn
import torch

def spectral_norm_layer(layer: nn.Module, n_power_iter: int = 10) -> float:
    """
    用 power iteration 估计一个线性层的谱范数.
    只处理 nn.Linear / nn.Conv2d 这类, 其他直接返回 1.0.
    """
    if isinstance(layer, nn.Linear):
        weight = layer.weight
    elif isinstance(layer, nn.Conv2d):
        # Conv 看成矩阵
        weight = layer.weight
        out_channels, in_channels, kh, kw = weight.shape
        weight = weight.view(out_channels, in_channels * kh * kw)
    else:
        return 1.0

    w = weight.detach()
    device = w.device
    u = torch.randn(w.size(0), device=device)
    u = u / u.norm()

    for _ in range(n_power_iter):
        v = torch.mv(w.t(), u)
        v = v / (v.norm() + 1e-12)
        u = torch.mv(w, v)
        u = u / (u.norm() + 1e-12)

    sigma = u @ (w @ v)
    return float(sigma.abs().item())

def estimate_network_lipschitz(model: nn.Module) -> float:
    """
    粗略估计整个网络的 Lipschitz 常数.
    用各线性层谱范数的乘积作为上界.
    """
    L = 1.0
    for module in model.modules():
        s = spectral_norm_layer(module)
        L *= s
    return L

