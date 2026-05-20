"""World-model evaluation on the fixed probe bank (proposal §4.5.2).

For each probe we:
  1. Encode the probe's start RGB to a latent via the current WM.
  2. Roll out the latent dynamics under the probe's fixed action sequence.
  3. Decode predicted latents through the semantic head -> semantic predictions.
  4. Compare against the stored ground-truth semantic rollout.

We report semantic L2 error at horizons {1, 5, 10, 15}, separately for the
visited and under-visited probe subsets (the partition is supplied by the
coverage module). The forgetting score is the difference between current and
pretrained-WM errors on the same probe bank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from wcollapse.data.probe_bank import ProbeBank
from wcollapse.models.ivideogpt_wrapper import MiniWorldModel, build_world_model
from wcollapse.models.semantic_head import SemanticHead
from wcollapse.utils.checkpoint import load_checkpoint


def _predict_semantic_rollout(
    world_model: MiniWorldModel,
    semantic_head: SemanticHead,
    bank: ProbeBank,
    device: torch.device,
) -> np.ndarray:
    """Returns (N_probes, H+1, semantic_dim) predicted semantic states."""
    world_model.eval()
    semantic_head.eval()
    N = len(bank)
    H = bank.horizon
    if N == 0:
        return np.zeros((0, H + 1, semantic_head.semantic_dim), dtype=np.float32)
    # Start frames: shape (N, H, W, 3).
    rgb0 = np.stack([p.gt_rgb[0] for p in bank.probes], axis=0)
    actions = np.stack([p.actions for p in bank.probes], axis=0)  # (N, H, A)
    with torch.no_grad():
        x0 = torch.from_numpy(rgb0).to(device).float() / 127.5 - 1.0
        x0 = x0.permute(0, 3, 1, 2)  # (N, 3, H, W)
        z = world_model.encoder(x0)  # (N, D)
        a_t = torch.from_numpy(actions).to(device).float()  # (N, H, A)
        latents = [z]
        for h in range(H):
            z = world_model.dynamics(z, a_t[:, h])
            latents.append(z)
        zs = torch.stack(latents, dim=1)  # (N, H+1, D)
        pred = semantic_head(zs.reshape(-1, zs.shape[-1])).view(N, H + 1, -1)
    return pred.cpu().numpy()


def _semantic_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-probe, per-horizon L2 error across the semantic vector. Returns (N, H+1)."""
    return np.linalg.norm(pred - gt, axis=-1)


def probe_eval(
    world_model: MiniWorldModel,
    semantic_head: SemanticHead,
    probe_bank: ProbeBank,
    wm_baseline_path: Path,
    device: torch.device,
    cfg: DictConfig,
    visited_mask: np.ndarray,
) -> dict[str, float]:
    """Compute semantic rollout error + forgetting + off-support gap."""
    H = probe_bank.horizon
    horizons = [h for h in (1, 5, 10, 15) if h <= H]

    gt = np.stack([p.gt_semantic for p in probe_bank.probes], axis=0) if len(probe_bank) else np.zeros((0, H + 1, 11), dtype=np.float32)
    pred = _predict_semantic_rollout(world_model, semantic_head, probe_bank, device)
    err = _semantic_errors(pred, gt)  # (N, H+1)

    # Forgetting score: same metric under the post-pretrain M_0 WM.
    # We share the current semantic_head — we want to isolate WM drift, not
    # head drift. (If the head also drifts, that's part of the system's
    # forgetting; it's fine to attribute jointly.) Route through build_world_model
    # so the M_0 instance matches whichever backbone the active config uses.
    M0 = build_world_model(world_model.cfg).to(device)
    state = load_checkpoint(wm_baseline_path, map_location=str(device))
    M0.load_state_dict(state["world_model"])
    pred_0 = _predict_semantic_rollout(M0, semantic_head, probe_bank, device)
    err_0 = _semantic_errors(pred_0, gt)
    forgetting = err - err_0  # positive = current WM is worse than M_0

    out: dict[str, float] = {}
    if len(probe_bank) == 0:
        out["probe_count"] = 0
        for h in horizons:
            out[f"sem_err_h{h}_overall"] = 0.0
            out[f"sem_err_h{h}_visited"] = 0.0
            out[f"sem_err_h{h}_underv"] = 0.0
            out[f"forget_h{h}_overall"] = 0.0
            out[f"forget_h{h}_underv"] = 0.0
            out[f"gap_h{h}"] = 0.0
        return out

    # Ensure mask shape matches probe count.
    if visited_mask.size != len(probe_bank):
        visited_mask = np.ones(len(probe_bank), dtype=bool)

    for h in horizons:
        err_h = err[:, h]
        forget_h = forgetting[:, h]
        out[f"sem_err_h{h}_overall"] = float(err_h.mean())
        out[f"sem_err_h{h}_visited"] = float(err_h[visited_mask].mean()) if visited_mask.any() else 0.0
        out[f"sem_err_h{h}_underv"] = float(err_h[~visited_mask].mean()) if (~visited_mask).any() else 0.0
        out[f"forget_h{h}_overall"] = float(forget_h.mean())
        out[f"forget_h{h}_underv"] = float(forget_h[~visited_mask].mean()) if (~visited_mask).any() else 0.0
        out[f"gap_h{h}"] = out[f"sem_err_h{h}_underv"] - out[f"sem_err_h{h}_visited"]

    out["probe_count"] = float(len(probe_bank))
    return out
