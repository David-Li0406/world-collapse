from __future__ import annotations

import torch
import torch.nn as nn


class SemanticHead(nn.Module):
    """Predict the 11-d semantic state from a WM latent.

    Dual-purpose: (a) supplies the analytic, interpretable signal we compare
    against during fixed-probe semantic rollout error (proposal §4.5.2);
    (b) maps imagined latents back into the same low-dim space used by the
    coverage / visitation-density metric.
    """

    def __init__(self, latent_dim: int, semantic_dim: int = 11, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.semantic_dim = semantic_dim
        dims = [latent_dim] + [hidden] * layers + [semantic_dim]
        net: list[nn.Module] = []
        for i in range(len(dims) - 2):
            net += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        net += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*net)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)
