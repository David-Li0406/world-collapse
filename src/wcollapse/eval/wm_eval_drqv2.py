"""Probe-bank evaluation against a `VideoPredictor` world model.

Replaces the semantic-head pipeline used in the SAC path. Each probe stores
(qpos, qvel, goal) + a fixed action sequence + the ground-truth RGB rollout
under that sequence. Here we:

  1. Build a frame-stacked initial observation from the probe's first frame
     (repeat 3x to populate the stack — same as FrameStackWrapper.reset).
  2. Roll the VideoPredictor forward under a teacher-forced policy that
     returns the probe's fixed actions.
  3. Compare predicted frames to ground-truth frames (pixel MSE on the
     newly-emitted frame at each horizon).
  4. Partition probes into visited vs under-visited (dynamic, policy-density)
     and trained-subregion vs held-out (static, goal-x split), and report
     mean errors per partition + forgetting score vs M_0.

Reported keys mirror the old `wm_eval.py` so the aggregator stays compatible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from wcollapse.data.probe_bank import ProbeBank


def _build_init_stack(rgb_hwc: np.ndarray, frame_stack: int) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 3*frame_stack, H, W) float on CPU. Values in [0, 255]."""
    chw = rgb_hwc.transpose(2, 0, 1)  # (3, H, W)
    stack = np.concatenate([chw] * frame_stack, axis=0)  # (3*fs, H, W)
    return torch.from_numpy(stack).unsqueeze(0).float()


def _predict_rollout_pixels(
    video_predictor,
    bank: ProbeBank,
    device: torch.device,
    frame_stack: int,
    horizon: int,
    chunk_size: int = 8,
) -> np.ndarray:
    """Returns (N, horizon+1, H, W, 3) uint8 predicted RGB frames per probe.

    Index 0 is the probe's start frame (matches gt_rgb[0]).
    Index t (t>=1) is the predicted frame after action a_{t-1}.

    Uses VideoPredictor.rollout with a teacher-forced policy closure.
    """
    video_predictor.eval()
    N = len(bank)
    if N == 0:
        return np.zeros((0, horizon + 1, 64, 64, 3), dtype=np.uint8)

    out_frames: list[np.ndarray] = []

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        probes = bank.probes[start:end]
        B = len(probes)
        obs_stacks = torch.cat(
            [_build_init_stack(p.gt_rgb[0], frame_stack) for p in probes], dim=0
        ).to(device)
        actions_per_step = torch.from_numpy(
            np.stack([p.actions for p in probes], axis=0)
        ).to(device).float()  # (B, H, 4)

        def teacher_forced_policy(_obs, t):
            return actions_per_step[:, t]

        with torch.no_grad():
            obss, _acts, _rews = video_predictor.rollout(
                obs_stacks, teacher_forced_policy, horizon
            )
        # obss: (B, H+1, 3*frame_stack, H, W) in [0, 1] float (rollout divides by 255)
        # Extract the newest frame at each step (last 3 channels).
        latest = obss[:, :, -3:, :, :]  # (B, H+1, 3, H, W)
        latest = latest.clamp(0, 1).permute(0, 1, 3, 4, 2)  # (B, H+1, H, W, 3)
        out_frames.append((latest.cpu().float().numpy() * 255).astype(np.uint8))

    return np.concatenate(out_frames, axis=0)


def _pixel_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-probe, per-horizon pixel MSE on [0,1] scale. Returns (N, H+1)."""
    a = pred.astype(np.float32) / 255.0
    b = gt.astype(np.float32) / 255.0
    return ((a - b) ** 2).mean(axis=(2, 3, 4))


def _aggregate_under_mask(err_h: np.ndarray, mask: np.ndarray) -> float:
    if mask.size == 0 or not mask.any():
        return float("nan")
    return float(err_h[mask].mean())


def probe_eval(
    video_predictor,
    probe_bank: ProbeBank,
    wm_baseline,
    device: torch.device,
    cfg: DictConfig,
    visited_mask: np.ndarray,
    static_visited_mask: np.ndarray | None = None,
    frame_stack: int = 3,
) -> dict:
    """Pixel-MSE probe eval + visited/under-visited split + forgetting score.

    `wm_baseline` is a second VideoPredictor instance holding M_0 weights
    (loaded by the caller). Passing None skips the forgetting computation.
    """
    H = probe_bank.horizon
    horizons = [h for h in (1, 5, 10, 15) if h <= H]

    if len(probe_bank) == 0:
        return {"probe_count": 0.0}

    pred = _predict_rollout_pixels(video_predictor, probe_bank, device, frame_stack, H)
    gt = np.stack([p.gt_rgb for p in probe_bank.probes], axis=0)
    err = _pixel_mse(pred, gt)  # (N, H+1)

    if wm_baseline is not None:
        pred0 = _predict_rollout_pixels(wm_baseline, probe_bank, device, frame_stack, H)
        err0 = _pixel_mse(pred0, gt)
        forgetting = err - err0
    else:
        forgetting = np.zeros_like(err)

    out = {"probe_count": float(len(probe_bank))}
    if visited_mask.size != len(probe_bank):
        visited_mask = np.ones(len(probe_bank), dtype=bool)
    if static_visited_mask is None or static_visited_mask.size != len(probe_bank):
        static_visited_mask = np.ones(len(probe_bank), dtype=bool)

    for h in horizons:
        err_h = err[:, h]
        forget_h = forgetting[:, h]
        out[f"sem_err_h{h}_overall"] = float(err_h.mean())
        out[f"forget_h{h}_overall"] = float(forget_h.mean())

        out[f"sem_err_h{h}_visited"] = _aggregate_under_mask(err_h, visited_mask)
        out[f"sem_err_h{h}_underv"] = _aggregate_under_mask(err_h, ~visited_mask)
        out[f"forget_h{h}_underv"] = _aggregate_under_mask(forget_h, ~visited_mask)
        v = out[f"sem_err_h{h}_visited"]
        u = out[f"sem_err_h{h}_underv"]
        out[f"gap_h{h}"] = (u - v) if (v == v and u == u) else 0.0

        out[f"sem_err_h{h}_static_visited"] = _aggregate_under_mask(err_h, static_visited_mask)
        out[f"sem_err_h{h}_static_underv"] = _aggregate_under_mask(err_h, ~static_visited_mask)
        out[f"forget_h{h}_static_underv"] = _aggregate_under_mask(forget_h, ~static_visited_mask)
        sv = out[f"sem_err_h{h}_static_visited"]
        su = out[f"sem_err_h{h}_static_underv"]
        out[f"static_gap_h{h}"] = (su - sv) if (sv == sv and su == su) else 0.0

    for h in horizons:
        out[f"per_probe_err_h{h}"] = err[:, h].astype(float).tolist()
        out[f"per_probe_forget_h{h}"] = forgetting[:, h].astype(float).tolist()
    out["per_probe_goal_x"] = [float(p.gt_semantic[0, 6]) for p in probe_bank.probes]

    return out
