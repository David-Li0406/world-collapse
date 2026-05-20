"""Dreamer-V3 style actor-critic update on imagined latent rollouts.

We sample real frames from the replay buffer, encode them to start latents,
then unroll the actor inside the latent dynamics for `H_imag` steps. Imagined
rewards come from the reward head; imagined values from the critic. λ-returns
serve as critic targets; the actor is updated by REINFORCE with an entropy
bonus (a numerically stabler variant than full reparameterized PG when going
through a residual MLP dynamics that may not be smooth in actions).

Per the plan, this is the ONLY place the actor and critic see gradients —
no real-env actor-critic updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from wcollapse.data.buffer import ReplayBuffer
from wcollapse.models.actor import Actor
from wcollapse.models.critic import Critic
from wcollapse.models.ivideogpt_wrapper import MiniWorldModel
from wcollapse.models.reward_head import RewardHead


@dataclass
class ImaginationOptimizers:
    actor: torch.optim.Optimizer
    critic: torch.optim.Optimizer
    critic_target: Critic
    ema_tau: float = 0.005

    def update_target(self, critic: Critic) -> None:
        with torch.no_grad():
            for p, p_t in zip(critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1 - self.ema_tau).add_(self.ema_tau * p.data)


def build_imagination_optimizers(
    actor: Actor,
    critic: Critic,
    cfg: DictConfig,
) -> ImaginationOptimizers:
    import copy

    critic_target = copy.deepcopy(critic)
    for p in critic_target.parameters():
        p.requires_grad_(False)
    return ImaginationOptimizers(
        actor=torch.optim.AdamW(actor.parameters(), lr=float(cfg.lr_actor)),
        critic=torch.optim.AdamW(critic.parameters(), lr=float(cfg.lr_critic)),
        critic_target=critic_target,
        ema_tau=float(cfg.get("ema_tau", 0.005)),
    )


def imagination_step(
    world_model: MiniWorldModel,
    reward_head: RewardHead,
    actor: Actor,
    critic: Critic,
    opts: ImaginationOptimizers,
    buffer: ReplayBuffer,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """One imagination update. Returns scalar metrics for logging."""
    world_model.eval()  # frozen during this update
    reward_head.eval()
    actor.train()
    critic.train()

    H = int(cfg.imag_horizon)
    gamma = float(cfg.gamma)
    lam = float(cfg.lambda_)
    batch_size = int(cfg.batch_size)
    entropy_coef = float(cfg.entropy_coef)

    batch = buffer.sample_sequences(batch_size=batch_size, seq_len=1)
    # Take only the first frame; we just need a real start state per imagined trajectory.
    rgb0 = torch.from_numpy(batch["rgb"][:, 0]).to(device)  # (B, H, W, 3) uint8
    x0 = (rgb0.float() / 127.5 - 1.0).permute(0, 3, 1, 2)  # (B, 3, H, W)
    with torch.no_grad():
        z = world_model.encoder(x0)  # (B, D)

    latents: list[torch.Tensor] = [z]
    rewards: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for h in range(H):
        sample = actor.sample(latents[-1])
        action = sample.action
        log_probs.append(sample.log_prob)
        entropies.append(sample.entropy)
        # WM step without grad (we treat WM as a fixed environment in this update).
        with torch.no_grad():
            z_next = world_model.step_latent(latents[-1], action)
        latents.append(z_next)
        with torch.no_grad():
            r = reward_head(z_next)
        rewards.append(r)

    rewards_t = torch.stack(rewards, dim=1)        # (B, H)
    log_probs_t = torch.stack(log_probs, dim=1)    # (B, H)
    entropies_t = torch.stack(entropies, dim=1)    # (B, H)
    latents_t = torch.stack(latents, dim=1)        # (B, H+1, D)

    with torch.no_grad():
        # Bootstrap from target critic.
        target_v = opts.critic_target(latents_t.reshape(-1, latents_t.shape[-1])).view(
            latents_t.shape[0], latents_t.shape[1]
        )
    # λ-return computed right-to-left over the imagined horizon.
    returns = torch.zeros_like(rewards_t)
    next_return = target_v[:, -1]
    for t in reversed(range(H)):
        bootstrap = (1 - lam) * target_v[:, t + 1] + lam * next_return
        next_return = rewards_t[:, t] + gamma * bootstrap
        returns[:, t] = next_return

    # Critic update.
    pred_v = critic(latents_t[:, :-1].reshape(-1, latents_t.shape[-1])).view(
        rewards_t.shape
    )
    critic_loss = F.mse_loss(pred_v, returns.detach())
    opts.critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    opts.critic.step()

    # Actor update — REINFORCE on advantages with entropy bonus.
    with torch.no_grad():
        baseline = critic(latents_t[:, :-1].reshape(-1, latents_t.shape[-1])).view(
            rewards_t.shape
        )
        advantages = returns - baseline
        # Normalize advantages (per-batch).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    actor_loss = -(log_probs_t * advantages.detach()).mean()
    entropy_bonus = entropies_t.mean()
    total_actor_loss = actor_loss - entropy_coef * entropy_bonus
    opts.actor.zero_grad(set_to_none=True)
    total_actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opts.actor.step()

    opts.update_target(critic)

    return {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy_bonus.item()),
        "imagined_return": float(returns.mean().item()),
        "imagined_reward": float(rewards_t.mean().item()),
    }
