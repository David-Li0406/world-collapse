"""Coverage / visitation metrics (proposal §4.5.1).

We work in the 3-D semantic subspace (obj_x, obj_y, goal_x): the puck position
and the part of the goal we restrict during collapse-induction. Histograms over
this subspace give us:
  * visitation entropy of the recent online window (Figure 1),
  * support gap |S_ref \\ S_t| / |S_ref| against the broad D_pre support,
  * per-probe density used to label probes visited vs under-visited (Figure 3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig

from wcollapse.data.buffer import ReplayBuffer
from wcollapse.data.probe_bank import ProbeBank


_BINS_DEFAULT = 20
_DENSITY_FLOOR_DEFAULT = 1e-4
_VISITED_PERCENTILE_DEFAULT = 50.0


def _collect_semantic_points(buffer: ReplayBuffer, recent_only: bool) -> np.ndarray:
    """Pull semantic states from the buffer's stored episodes.

    Returns an (N, 3) array of (obj_x, obj_y, goal_x).
    """
    if recent_only:
        ep_ids = list(buffer._recent_episode_ids)
    else:
        ep_ids = list(range(len(buffer._episodes)))
    rows: list[np.ndarray] = []
    for ei in ep_ids:
        ep = buffer._episodes[ei]
        sem = ep["semantic"]  # (T+1, 11)
        rows.append(sem[:, [3, 4, 6]])  # obj_x, obj_y, goal_x
    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32)


def _histogram_bounds(env_goal_low: np.ndarray, env_goal_high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounds for the 3-D histogram. Object x/y use the obj range; goal x uses goal range."""
    lo = np.array([-0.12, 0.58, env_goal_low[0] - 0.01], dtype=np.float32)
    hi = np.array([0.12, 0.72, env_goal_high[0] + 0.01], dtype=np.float32)
    return lo, hi


def _bin_points(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, bins: int) -> np.ndarray:
    """Return an empirical density tensor of shape (bins, bins, bins). Sums to 1."""
    if points.size == 0:
        return np.zeros((bins, bins, bins), dtype=np.float32)
    H, edges = np.histogramdd(
        points,
        bins=bins,
        range=list(zip(lo.tolist(), hi.tolist())),
    )
    total = H.sum()
    return (H / total).astype(np.float32) if total > 0 else H.astype(np.float32)


def _entropy(density: np.ndarray) -> float:
    p = density.reshape(-1)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _density_at(points: np.ndarray, density: np.ndarray, lo: np.ndarray, hi: np.ndarray, bins: int) -> np.ndarray:
    """Look up the density at each query point. Returns (N,)."""
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    idx = ((points - lo) / (hi - lo) * bins).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)
    return density[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)


def coverage_metrics(
    pretrain_buffer: ReplayBuffer,
    active_buffer: ReplayBuffer,
    probe_bank: ProbeBank,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Compute scalar metrics + per-probe visited mask + density grids for plotting."""
    bins = int(cfg.get("coverage_bins", _BINS_DEFAULT))
    floor = float(cfg.get("density_floor", _DENSITY_FLOOR_DEFAULT))
    visited_pct = float(cfg.get("visited_percentile", _VISITED_PERCENTILE_DEFAULT))

    # The reference support is everything D_pre saw (and any visitation since).
    ref_points = _collect_semantic_points(pretrain_buffer, recent_only=False)
    # The current support is just the recent window.
    cur_points = _collect_semantic_points(active_buffer, recent_only=True)
    if cur_points.size == 0:
        # Fallback when the recent buffer hasn't filled yet — use the full
        # active buffer so the metric is defined.
        cur_points = _collect_semantic_points(active_buffer, recent_only=False)

    if ref_points.size == 0:
        # No data yet — return zeros so the loop can keep going.
        return {
            "scalar": {
                "visitation_entropy": 0.0,
                "support_gap": 0.0,
                "n_ref_points": 0,
                "n_cur_points": int(cur_points.shape[0]),
            },
            "visited_mask": np.ones(len(probe_bank), dtype=bool),
            "density_ref": np.zeros((bins, bins, bins), dtype=np.float32),
            "density_cur": np.zeros((bins, bins, bins), dtype=np.float32),
        }

    # Histogram bounds — derive from D_pre's spread, padded slightly.
    lo = np.minimum(ref_points.min(0), cur_points.min(0) if cur_points.size else ref_points.min(0)) - 0.02
    hi = np.maximum(ref_points.max(0), cur_points.max(0) if cur_points.size else ref_points.max(0)) + 0.02

    p_ref = _bin_points(ref_points, lo, hi, bins)
    p_cur = _bin_points(cur_points, lo, hi, bins)

    visitation_entropy = _entropy(p_cur)
    # Support gap: bins above floor in p_ref but not in p_cur.
    s_ref = p_ref > floor
    s_cur = p_cur > floor
    if s_ref.sum() == 0:
        support_gap = 0.0
    else:
        forgotten = (s_ref & ~s_cur).sum()
        support_gap = float(forgotten) / float(s_ref.sum())

    # Per-probe density under the current policy density.
    probe_pts = np.stack(
        [[float(p.start.qpos[9 + 0]) if p.start.qpos.shape[0] >= 12 else 0.0,
          float(p.start.qpos[9 + 1]) if p.start.qpos.shape[0] >= 12 else 0.0,
          float(p.start.target_pos[0])] for p in probe_bank.probes],
        dtype=np.float32,
    ).reshape(-1, 3) if len(probe_bank) > 0 else np.zeros((0, 3), dtype=np.float32)
    # NOTE: object qpos indices for push-v3 are at qpos[9:12] (see _set_obj_xyz);
    # falling back to gt_semantic is safer.
    probe_pts = np.stack(
        [p.gt_semantic[0, [3, 4, 6]] for p in probe_bank.probes],
        dtype=np.float32,
    ) if len(probe_bank) > 0 else np.zeros((0, 3), dtype=np.float32)

    densities = _density_at(probe_pts, p_cur, lo, hi, bins)
    if densities.size > 0:
        thresh = np.percentile(densities, visited_pct)
        visited_mask = densities > thresh
    else:
        visited_mask = np.zeros((0,), dtype=bool)

    return {
        "scalar": {
            "visitation_entropy": visitation_entropy,
            "support_gap": support_gap,
            "n_ref_points": int(ref_points.shape[0]),
            "n_cur_points": int(cur_points.shape[0]),
            "n_visited_probes": int(visited_mask.sum()),
            "n_underv_probes": int((~visited_mask).sum()),
        },
        "visited_mask": visited_mask,
        "density_ref": p_ref,
        "density_cur": p_cur,
    }
