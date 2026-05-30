"""Coverage / visitation metrics for the DrQ-v2 + VideoPredictor path.

Reads from wmcollapse's `ReplayBufferStorage` NPZ episode files. Each
episode stores a `state` field (the raw 39-d MetaWorld obs that our
ExtendedTimeStepWrapper populates), from which we derive (obj_x, obj_y,
goal_x) via the same indices used by `semantic_from_obs`.

Reproduces every key emitted by the SAC-era `coverage.py` so the downstream
aggregator stays identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from omegaconf import DictConfig

from wcollapse.data.probe_bank import ProbeBank


_BINS_DEFAULT = 20
_DENSITY_FLOOR_DEFAULT = 1e-4
_VISITED_RADIUS_DEFAULT = 0.05
_STATIC_GOAL_SPLIT_DEFAULT = 0.0

_OBS_OBJ_X = 4
_OBS_OBJ_Y = 5
_OBS_GOAL_X = 36


def _count_within_radius(probe_pts: np.ndarray, actor_pts: np.ndarray, radius: float) -> np.ndarray:
    if probe_pts.size == 0 or actor_pts.size == 0:
        return np.zeros(len(probe_pts), dtype=np.int64)
    r2 = radius * radius
    out = np.zeros(probe_pts.shape[0], dtype=np.int64)
    chunk = max(1, 1_000_000 // max(1, actor_pts.shape[0]))
    for i in range(0, probe_pts.shape[0], chunk):
        diff = probe_pts[i : i + chunk, None, :] - actor_pts[None, :, :]
        sq = (diff * diff).sum(-1)
        out[i : i + chunk] = (sq < r2).sum(-1)
    return out


def _list_npz(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(d.glob("*.npz"))


def _semantic_from_npz_files(files: Iterable[Path]) -> np.ndarray:
    """Pull (obj_x, obj_y, goal_x) from a list of NPZ episode files."""
    rows: list[np.ndarray] = []
    for fn in files:
        try:
            with np.load(fn) as ep:
                if "state" not in ep:
                    continue
                state = ep["state"]  # (T+1, 39)
                rows.append(state[:, [_OBS_OBJ_X, _OBS_OBJ_Y, _OBS_GOAL_X]])
        except Exception:
            continue
    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32)


def _recent_npz_files(d: Path, n_recent: int) -> list[Path]:
    """ReplayBufferStorage filenames are `<ts>_<idx>_<eps_len>.npz`. Sort by idx, take the last n."""
    files = _list_npz(d)
    if n_recent <= 0:
        return files
    keyed = []
    for f in files:
        try:
            idx = int(f.stem.split("_")[-2])
            keyed.append((idx, f))
        except Exception:
            keyed.append((0, f))
    keyed.sort()
    return [f for _, f in keyed[-n_recent:]]


def _bin_points(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, bins: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((bins, bins, bins), dtype=np.float32)
    H, _ = np.histogramdd(
        points, bins=bins, range=list(zip(lo.tolist(), hi.tolist())),
    )
    total = H.sum()
    return (H / total).astype(np.float32) if total > 0 else H.astype(np.float32)


def _entropy(d: np.ndarray) -> float:
    p = d.reshape(-1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def coverage_metrics(
    pretrain_buffer_dir: Path,
    active_buffer_dir: Path,
    recent_n_episodes: int,
    probe_bank: ProbeBank,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Compute scalar metrics + per-probe visited mask + density grids.

    `pretrain_buffer_dir` is the round-0 buffer (D_real_R0 in the proposal).
    `active_buffer_dir` is this Phase B run's collected real env transitions.
    `recent_n_episodes` is the per-probe density window for the dynamic
    partition (e.g. cfg.recent_window // 200).
    """
    bins = int(cfg.get("coverage_bins", _BINS_DEFAULT))
    floor = float(cfg.get("density_floor", _DENSITY_FLOOR_DEFAULT))

    ref_points = _semantic_from_npz_files(_list_npz(pretrain_buffer_dir))
    cur_points = _semantic_from_npz_files(_recent_npz_files(active_buffer_dir, recent_n_episodes))
    if cur_points.size == 0:
        cur_points = _semantic_from_npz_files(_list_npz(active_buffer_dir))

    goal_split = float(cfg.get("static_goal_split", _STATIC_GOAL_SPLIT_DEFAULT))
    if len(probe_bank) > 0:
        goal_xs = np.array([p.gt_semantic[0, 6] for p in probe_bank.probes], dtype=np.float32)
        static_visited_mask = goal_xs < goal_split
    else:
        static_visited_mask = np.zeros((0,), dtype=bool)

    if ref_points.size == 0:
        empty_mask = np.zeros(len(probe_bank), dtype=bool)
        return {
            "scalar": {
                "visitation_entropy": 0.0,
                "support_gap": 0.0,
                "n_ref_points": 0,
                "n_cur_points": int(cur_points.shape[0]),
                "n_visited_probes": 0,
                "n_underv_probes": int(len(probe_bank)),
                "n_visited_static": int(static_visited_mask.sum()),
                "n_underv_static": int((~static_visited_mask).sum()),
            },
            "visited_mask": empty_mask,
            "static_visited_mask": static_visited_mask,
            "density_ref": np.zeros((bins, bins, bins), dtype=np.float32),
            "density_cur": np.zeros((bins, bins, bins), dtype=np.float32),
        }

    lo = np.minimum(ref_points.min(0), cur_points.min(0) if cur_points.size else ref_points.min(0)) - 0.02
    hi = np.maximum(ref_points.max(0), cur_points.max(0) if cur_points.size else ref_points.max(0)) + 0.02
    p_ref = _bin_points(ref_points, lo, hi, bins)
    p_cur = _bin_points(cur_points, lo, hi, bins)
    visitation_entropy = _entropy(p_cur)
    s_ref = p_ref > floor
    s_cur = p_cur > floor
    if s_ref.sum() == 0:
        support_gap = 0.0
    else:
        forgotten = (s_ref & ~s_cur).sum()
        support_gap = float(forgotten) / float(s_ref.sum())

    probe_pts = (
        np.stack([p.gt_semantic[0, [3, 4, 6]] for p in probe_bank.probes], axis=0).astype(np.float32)
        if len(probe_bank) > 0
        else np.zeros((0, 3), dtype=np.float32)
    )
    radius = float(cfg.get("visited_radius", _VISITED_RADIUS_DEFAULT))
    counts = _count_within_radius(probe_pts, cur_points, radius)
    visited_mask = counts > 0

    return {
        "scalar": {
            "visitation_entropy": visitation_entropy,
            "support_gap": support_gap,
            "n_ref_points": int(ref_points.shape[0]),
            "n_cur_points": int(cur_points.shape[0]),
            "n_visited_probes": int(visited_mask.sum()),
            "n_underv_probes": int((~visited_mask).sum()),
            "n_visited_static": int(static_visited_mask.sum()),
            "n_underv_static": int((~static_visited_mask).sum()),
            "visited_radius": radius,
        },
        "visited_mask": visited_mask,
        "static_visited_mask": static_visited_mask,
        "density_ref": p_ref,
        "density_cur": p_cur,
    }
