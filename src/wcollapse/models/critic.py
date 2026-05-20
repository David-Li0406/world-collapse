from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, layers: int, out_dim: int) -> nn.Sequential:
    dims = [in_dim] + [hidden] * layers + [out_dim]
    seq: list[nn.Module] = []
    for i in range(len(dims) - 2):
        seq += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
    seq += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*seq)


class Critic(nn.Module):
    """Twin-Q action-value critic for SAC.

    Two heads Q1, Q2 over the (latent, action) concatenation. The SAC update
    bootstraps from ``min(Q1_target, Q2_target)`` so we keep two parallel nets
    plus EMA-tracked targets in the training loop.

    Backward-compatibility: if called with one tensor (``Critic(latent)``),
    behaves like the legacy V-only critic and returns the average of Q1, Q2 at
    a zero action — used only by old code that called ``Critic(latent)``. New
    code should call ``Critic(latent, action) -> (q1, q2)``.
    """

    def __init__(self, latent_dim: int, action_dim: int = 4, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.q1 = _mlp(latent_dim + action_dim, hidden, layers, 1)
        self.q2 = _mlp(latent_dim + action_dim, hidden, layers, 1)

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if action is None:
            # Legacy V-network call path: return mean Q under a zero action so
            # any callers from older code still get a scalar value back.
            zero_a = torch.zeros(*latent.shape[:-1], self.action_dim, device=latent.device, dtype=latent.dtype)
            q1 = self.q1(torch.cat([latent, zero_a], -1)).squeeze(-1)
            q2 = self.q2(torch.cat([latent, zero_a], -1)).squeeze(-1)
            return 0.5 * (q1 + q2)
        x = torch.cat([latent, action], -1)
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)
        return q1, q2
