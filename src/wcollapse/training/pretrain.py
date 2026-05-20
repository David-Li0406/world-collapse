"""Phase A: pretrain the world model + reward head + semantic head on D_pre.

`D_pre` is constructed by `collect_seed_dataset` (mixed scripted+random over the
full goal range). The pretrained checkpoint is then reused as the starting
point for all three Phase-B conditions (collapse-prone, balanced replay,
frozen WM). It is ALSO the baseline checkpoint M_0 used by the forgetting score
in §4.5.2 — we snapshot it to disk and never overwrite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from wcollapse.data.buffer import ReplayBuffer
from wcollapse.data.trajectory import Trajectory
from wcollapse.models.ivideogpt_wrapper import MiniWorldModel
from wcollapse.models.reward_head import RewardHead
from wcollapse.models.semantic_head import SemanticHead


def pretrain(
    world_model: MiniWorldModel,
    reward_head: RewardHead,
    semantic_head: SemanticHead,
    buffer: ReplayBuffer,
    cfg: DictConfig,
    device: torch.device,
    log_fn=lambda step, metrics: None,
) -> None:
    """Run Phase A. Mutates the modules in place."""
    world_model.train()
    reward_head.train()
    semantic_head.train()

    wm_opt = torch.optim.AdamW(world_model.parameters(), lr=cfg.lr_wm)
    rew_opt = torch.optim.AdamW(reward_head.parameters(), lr=cfg.lr_head)
    sem_opt = torch.optim.AdamW(semantic_head.parameters(), lr=cfg.lr_head)

    for step in range(int(cfg.steps)):
        batch = buffer.sample_sequences(batch_size=int(cfg.batch_size))
        # World-model update.
        wm_opt.zero_grad(set_to_none=True)
        wm_loss = world_model.compute_loss(batch)
        wm_loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
        wm_opt.step()

        # Reward + semantic heads on encoded latents.
        with torch.no_grad():
            rgb = torch.from_numpy(batch["rgb"]).to(device)
            # encode every frame; use the post-action frames for supervision.
            x = (rgb.float() / 127.5 - 1.0).permute(0, 1, 4, 2, 3)
            B, Lp1 = x.shape[:2]
            z = world_model.encoder(x.reshape(B * Lp1, *x.shape[2:])).view(B, Lp1, -1)
        target_rewards = torch.from_numpy(batch["rewards"]).to(device)            # (B, L)
        target_semantic = torch.from_numpy(batch["semantic"]).to(device)[:, 1:]   # (B, L, 11)
        z_post = z[:, 1:]  # latents at t+1

        rew_opt.zero_grad(set_to_none=True)
        rew_pred = reward_head(z_post.reshape(-1, z_post.shape[-1]))
        rew_loss = F.mse_loss(rew_pred, target_rewards.reshape(-1))
        rew_loss.backward()
        rew_opt.step()

        sem_opt.zero_grad(set_to_none=True)
        sem_pred = semantic_head(z_post.reshape(-1, z_post.shape[-1]))
        sem_loss = F.mse_loss(sem_pred, target_semantic.reshape(-1, target_semantic.shape[-1]))
        sem_loss.backward()
        sem_opt.step()

        if step % max(1, int(cfg.get("log_every", 50))) == 0:
            log_fn(
                step,
                {
                    "phase": "pretrain",
                    "wm_total": float(wm_loss["total"].item()),
                    "wm_recon": float(wm_loss["recon"].item()),
                    "wm_dyn": float(wm_loss["dynamics"].item()),
                    "reward_mse": float(rew_loss.item()),
                    "semantic_mse": float(sem_loss.item()),
                },
            )


def build_pretrain_buffer(
    trajectories: list[Trajectory],
    *,
    image_size: int,
    seq_len: int,
    capacity: int = 1_000_000,
) -> ReplayBuffer:
    buf = ReplayBuffer(
        capacity=capacity,
        window_size=capacity,
        seq_len=seq_len,
        image_size=image_size,
        mode="uniform",
    )
    for tr in trajectories:
        buf.add_trajectory(tr)
    return buf
