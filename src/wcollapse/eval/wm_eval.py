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


def _aggregate_under_mask(err_h: np.ndarray, mask: np.ndarray) -> float:
    if mask.size == 0 or not mask.any():
        return float("nan")
    return float(err_h[mask].mean())


def probe_eval(
    world_model: MiniWorldModel,
    semantic_head: SemanticHead,
    probe_bank: ProbeBank,
    wm_baseline_path: Path,
    device: torch.device,
    cfg: DictConfig,
    visited_mask: np.ndarray,
    static_visited_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute semantic rollout error + forgetting + off-support gap under
    BOTH the dynamic (policy-density) partition and the static (goal-region)
    partition. Logging keys are prefixed accordingly so downstream analysis
    can compare definitions side-by-side.
    """
    H = probe_bank.horizon
    horizons = [h for h in (1, 5, 10, 15) if h <= H]

    gt = (
        np.stack([p.gt_semantic for p in probe_bank.probes], axis=0)
        if len(probe_bank)
        else np.zeros((0, H + 1, 11), dtype=np.float32)
    )
    pred = _predict_semantic_rollout(world_model, semantic_head, probe_bank, device)
    err = _semantic_errors(pred, gt)  # (N, H+1)

    # Forgetting score: same metric under the post-pretrain M_0 WM.
    # We share the current semantic_head — we want to isolate WM drift, not
    # head drift. Route through build_world_model so the M_0 instance matches
    # whichever backbone the active config uses.
    M0 = build_world_model(world_model.cfg).to(device)
    state = load_checkpoint(wm_baseline_path, map_location=str(device))
    M0.load_state_dict(state["world_model"])
    pred_0 = _predict_semantic_rollout(M0, semantic_head, probe_bank, device)
    err_0 = _semantic_errors(pred_0, gt)
    forgetting = err - err_0  # positive = current WM is worse than M_0

    out: dict[str, float] = {"probe_count": float(len(probe_bank))}
    if len(probe_bank) == 0:
        return out

    if visited_mask.size != len(probe_bank):
        visited_mask = np.ones(len(probe_bank), dtype=bool)
    if static_visited_mask is None or static_visited_mask.size != len(probe_bank):
        static_visited_mask = np.ones(len(probe_bank), dtype=bool)

    for h in horizons:
        err_h = err[:, h]
        forget_h = forgetting[:, h]
        out[f"sem_err_h{h}_overall"] = float(err_h.mean())
        out[f"forget_h{h}_overall"] = float(forget_h.mean())

        # Dynamic policy-density partition (original behavior).
        out[f"sem_err_h{h}_visited"] = _aggregate_under_mask(err_h, visited_mask)
        out[f"sem_err_h{h}_underv"] = _aggregate_under_mask(err_h, ~visited_mask)
        out[f"forget_h{h}_underv"] = _aggregate_under_mask(forget_h, ~visited_mask)
        v = out[f"sem_err_h{h}_visited"]
        u = out[f"sem_err_h{h}_underv"]
        out[f"gap_h{h}"] = (u - v) if (v == v and u == u) else 0.0

        # Static goal-region partition (independent of actor learning).
        out[f"sem_err_h{h}_static_visited"] = _aggregate_under_mask(err_h, static_visited_mask)
        out[f"sem_err_h{h}_static_underv"] = _aggregate_under_mask(err_h, ~static_visited_mask)
        out[f"forget_h{h}_static_underv"] = _aggregate_under_mask(forget_h, ~static_visited_mask)
        sv = out[f"sem_err_h{h}_static_visited"]
        su = out[f"sem_err_h{h}_static_underv"]
        out[f"static_gap_h{h}"] = (su - sv) if (sv == sv and su == su) else 0.0

    # Also save raw per-probe semantic-L2-error vectors at the four horizons,
    # plus per-probe forgetting, so any future re-partitioning is offline.
    for h in horizons:
        out[f"per_probe_err_h{h}"] = err[:, h].astype(float).tolist()
        out[f"per_probe_forget_h{h}"] = forgetting[:, h].astype(float).tolist()
    out["per_probe_goal_x"] = [float(p.gt_semantic[0, 6]) for p in probe_bank.probes]

    return out
