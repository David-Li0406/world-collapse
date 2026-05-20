from __future__ import annotations

import torch
import torch.nn as nn


class RewardHead(nn.Module):
    """Small MLP from WM latent to scalar reward."""

    def __init__(self, latent_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        dims = [latent_dim] + [hidden] * layers + [1]
        net: list[nn.Module] = []
        for i in range(len(dims) - 2):
            net += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        net += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*net)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)
