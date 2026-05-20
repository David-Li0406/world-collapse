"""SAC + MBPO update over the iVideoGPT latent.

Replaces the prior REINFORCE-imagination loop. The agent trains off-policy
on real (s, a, r, s') transitions sampled from the active buffer, mixed with
short-horizon transitions imagined under the current world model — the
MBPO recipe used in iVideoGPT/mbrl/train_metaworld_mbpo.py.

Differences from the iVideoGPT recipe:
  * We don't use the iVideoGPT autoregressive transformer for the WM rollout;
    we use the same learned residual MLP dynamics that the rest of the
    pipeline uses (mini or ivideogpt backbone — both expose `encoder` and
    `step_latent`).
  * We don't unfreeze the tokenizer here. The actor and critic train, the
    reward/semantic heads train, the WM dynamics MLP trains; the encoder
    is whatever the WM wrapper exposes.

Notes:
  * actions stored in the buffer are the actions the data-collecting actor
    (or scripted policy, for D_pre) executed. They are tanh-bounded to
    [-1, 1] already because the env's action space is that box.
  * SAC alpha (entropy temperature) is learnable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from wcollapse.data.buffer import ReplayBuffer
from wcollapse.models.actor import Actor
from wcollapse.models.critic import Critic
from wcollapse.models.ivideogpt_wrapper import MiniWorldModel
from wcollapse.models.reward_head import RewardHead


@dataclass
class SacOptimizers:
    actor: torch.optim.Optimizer
    critic: torch.optim.Optimizer
    log_alpha: torch.Tensor
    alpha_opt: torch.optim.Optimizer
    target_critic: Critic
    target_entropy: float
    ema_tau: float = 0.005
    gamma: float = 0.99

    def soft_update(self, critic: Critic) -> None:
        with torch.no_grad():
            for p, p_t in zip(critic.parameters(), self.target_critic.parameters()):
                p_t.data.mul_(1.0 - self.ema_tau).add_(self.ema_tau * p.data)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()


def build_sac_optimizers(
    actor: Actor,
    critic: Critic,
    cfg: DictConfig,
    device: torch.device,
) -> SacOptimizers:
    target_critic = copy.deepcopy(critic).to(device)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    init_alpha = float(cfg.get("init_alpha", 0.2))
    log_alpha = torch.tensor(np.log(init_alpha), device=device, dtype=torch.float32, requires_grad=True)

    # Target entropy = -|A| is the standard SAC default for continuous control.
    action_dim = int(cfg.get("action_dim", 4))
    target_entropy = float(cfg.get("target_entropy", -action_dim))

    return SacOptimizers(
        actor=torch.optim.AdamW(actor.parameters(), lr=float(cfg.lr_actor)),
        critic=torch.optim.AdamW(critic.parameters(), lr=float(cfg.lr_critic)),
        log_alpha=log_alpha,
        alpha_opt=torch.optim.AdamW([log_alpha], lr=float(cfg.get("lr_alpha", 1.0e-4))),
        target_critic=target_critic,
        target_entropy=target_entropy,
        ema_tau=float(cfg.get("ema_tau", 0.005)),
        gamma=float(cfg.get("gamma", 0.99)),
    )


def _encode_no_grad(world_model: MiniWorldModel, rgb_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Encode a uint8 RGB tensor (B, ..., H, W, 3) -> (B, ..., D) latents.

    Accepts shapes (B, H, W, 3) or (B, T, H, W, 3).
    """
    x = torch.from_numpy(np.ascontiguousarray(rgb_np)).to(device).float() / 127.5 - 1.0
    if x.dim() == 4:  # B, H, W, 3
        x = x.permute(0, 3, 1, 2)
        with torch.no_grad():
            return world_model.encoder(x)  # (B, D)
    elif x.dim() == 5:  # B, T, H, W, 3
        B, T = x.shape[:2]
        x = x.permute(0, 1, 4, 2, 3)
        with torch.no_grad():
            z = world_model.encoder(x.reshape(B * T, *x.shape[2:]))
        return z.view(B, T, -1)
    raise ValueError(f"Unexpected rgb shape: {x.shape}")


@torch.no_grad()
def _imagine_transitions(
    world_model: MiniWorldModel,
    reward_head: RewardHead,
    actor: Actor,
    init_latents: torch.Tensor,
    horizon: int,
) -> dict[str, torch.Tensor]:
    """Roll out the WM under the current policy for `horizon` steps from each
    real start latent. Returns flat tensors (B*horizon, ...) of imagined
    transitions: (s, a, r, s_next, done=0).
    """
    B = init_latents.shape[0]
    s = init_latents
    flat_s, flat_a, flat_r, flat_s_next = [], [], [], []
    for _ in range(horizon):
        sample = actor.sample(s)
        a = sample.action  # (B, A) in [-1, 1]
        s_next = world_model.step_latent(s, a)
        r = reward_head(s_next)
        flat_s.append(s)
        flat_a.append(a)
        flat_r.append(r)
        flat_s_next.append(s_next)
        s = s_next
    return {
        "s": torch.cat(flat_s, dim=0),
        "a": torch.cat(flat_a, dim=0),
        "r": torch.cat(flat_r, dim=0),
        "s_next": torch.cat(flat_s_next, dim=0),
        "done": torch.zeros(B * horizon, device=init_latents.device, dtype=torch.float32),
    }


def _sample_real_transitions(
    buffer: ReplayBuffer, world_model: MiniWorldModel, batch_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Sample 1-step real transitions from the buffer and encode through WM."""
    batch = buffer.sample_sequences(batch_size=batch_size, seq_len=1)
    z = _encode_no_grad(world_model, batch["rgb"], device)  # (B, 2, D)
    s = z[:, 0]
    s_next = z[:, 1]
    a = torch.from_numpy(batch["actions"][:, 0]).to(device)  # (B, A)
    r = torch.from_numpy(batch["rewards"][:, 0]).to(device)  # (B,)
    d = torch.from_numpy(batch["dones"][:, 0].astype(np.float32)).to(device)  # (B,)
    return {"s": s, "a": a, "r": r, "s_next": s_next, "done": d}


def sac_step(
    world_model: MiniWorldModel,
    reward_head: RewardHead,
    actor: Actor,
    critic: Critic,
    opts: SacOptimizers,
    buffer: ReplayBuffer,
    cfg: DictConfig,
    device: torch.device,
    use_imagination: bool = True,
) -> dict[str, float]:
    """One SAC + MBPO update.

    Composition: real transitions are sampled from `buffer` (the active
    replay-buffer for this condition). Imagined transitions are short-horizon
    rollouts in the WM latent from real start states. The two are concatenated
    50/50 by default (`real_ratio`).
    """
    actor.train()
    critic.train()
    world_model.eval()
    reward_head.eval()

    batch_size = int(cfg.batch_size)
    real_ratio = float(cfg.get("real_ratio", 0.5))
    imag_horizon = int(cfg.get("mbpo_imag_horizon", 1))

    n_real = max(1, int(round(batch_size * real_ratio)))
    n_imag = max(0, batch_size - n_real)
    real = _sample_real_transitions(buffer, world_model, n_real, device)

    if use_imagination and n_imag > 0:
        # Need n_imag transitions total; with horizon H each real start gives H,
        # so we need ceil(n_imag / H) start states. Round up and crop.
        n_start = max(1, (n_imag + imag_horizon - 1) // imag_horizon)
        start_real = _sample_real_transitions(buffer, world_model, n_start, device)
        imag = _imagine_transitions(
            world_model, reward_head, actor, start_real["s"], imag_horizon
        )
        # Crop to exactly n_imag transitions.
        imag = {k: v[:n_imag] for k, v in imag.items()}
        merged = {k: torch.cat([real[k], imag[k]], dim=0) for k in real}
    else:
        merged = real

    s = merged["s"]
    a = merged["a"]
    r = merged["r"]
    s_next = merged["s_next"]
    d = merged["done"]

    # --- Critic update ---
    with torch.no_grad():
        a_next = actor.sample(s_next)
        q1_t, q2_t = opts.target_critic(s_next, a_next.action)
        q_next = torch.min(q1_t, q2_t) - opts.alpha.detach() * a_next.log_prob
        target = r + opts.gamma * (1.0 - d) * q_next
    q1, q2 = critic(s, a)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    opts.critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    opts.critic.step()

    # --- Actor update (reparameterized) ---
    a_pi = actor.sample(s)
    q1_pi, q2_pi = critic(s, a_pi.action)
    q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = (opts.alpha.detach() * a_pi.log_prob - q_pi).mean()
    opts.actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opts.actor.step()

    # --- Alpha (entropy temperature) update ---
    alpha_loss = -(opts.log_alpha * (a_pi.log_prob.detach() + opts.target_entropy)).mean()
    opts.alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    opts.alpha_opt.step()

    opts.soft_update(critic)

    return {
        "sac_critic_loss": float(critic_loss.item()),
        "sac_actor_loss": float(actor_loss.item()),
        "sac_alpha": float(opts.alpha.item()),
        "sac_alpha_loss": float(alpha_loss.item()),
        "sac_target_mean": float(target.mean().item()),
        "sac_q_mean": float(q_pi.mean().item()),
        "sac_entropy": float(-a_pi.log_prob.mean().item()),
        "sac_reward_mean": float(r.mean().item()),
    }
