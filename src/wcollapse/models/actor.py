"""Tanh-squashed Gaussian actor over the 4-d Metaworld action space.

Designed for the Dreamer-V3 recipe: trained on reparameterized samples inside
imagination, evaluated deterministically (via the tanh-of-mean) at env step
time, with a small exploration noise injected during data collection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionSample:
    action: torch.Tensor       # (B, A) in [-1, 1]
    log_prob: torch.Tensor     # (B,)
    entropy: torch.Tensor      # (B,)


class Actor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 4,
        hidden: int = 256,
        layers: int = 2,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        dims = [latent_dim] + [hidden] * layers
        body: list[nn.Module] = []
        for i in range(len(dims) - 1):
            body += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        self.body = nn.Sequential(*body)
        self.mu_head = nn.Linear(dims[-1], action_dim)
        self.log_std_head = nn.Linear(dims[-1], action_dim)

    def _dist_params(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(latent)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, latent: torch.Tensor) -> ActionSample:
        mu, log_std = self._dist_params(latent)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)
        # log-prob with tanh correction (Haarnoja 2018).
        log_prob_u = -0.5 * (eps**2 + 2 * log_std + torch.log(torch.tensor(2 * torch.pi)))
        log_prob = (log_prob_u - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        entropy = (0.5 * (1.0 + torch.log(torch.tensor(2 * torch.pi))) + log_std).sum(-1)
        return ActionSample(action=a, log_prob=log_prob, entropy=entropy)

    @torch.no_grad()
    def act(self, latent: torch.Tensor, deterministic: bool = False, noise_std: float = 0.0) -> torch.Tensor:
        mu, log_std = self._dist_params(latent)
        if deterministic:
            a = torch.tanh(mu)
        else:
            std = log_std.exp()
            a = torch.tanh(mu + std * torch.randn_like(mu))
        if noise_std > 0.0:
            a = (a + noise_std * torch.randn_like(a)).clamp(-1.0, 1.0)
        return a
