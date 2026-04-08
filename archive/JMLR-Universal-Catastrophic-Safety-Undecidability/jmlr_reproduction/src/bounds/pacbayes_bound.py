from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn

@dataclass
class PacBayesConfig:
    delta: float = 1e-3
    prior_variance: float = 1.0

def pac_bayes_bound(
    empirical_error: float,
    model: nn.Module,
    n_samples: int,
    cfg: PacBayesConfig,
) -> float:
    """
    一个非常简单的 PAC-Bayes 占位实现.
    实际上你可以用更严谨的 KL 估计替换这里.
    """
    # 粗糙 proxy: 用参数 L2 范数近似 KL 与 prior 的关系
    params = torch.cat([p.view(-1) for p in model.parameters()])
    l2_norm_sq = float(params.pow(2).sum().item())
    kl_proxy = 0.5 * l2_norm_sq / cfg.prior_variance

    term = math.sqrt((kl_proxy + math.log(1.0 / cfg.delta)) / (2.0 * n_samples))
    return empirical_error + term

