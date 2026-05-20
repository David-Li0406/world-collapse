"""Trajectory dataclass and HDF5 I/O.

A `Trajectory` is one episode: stacked per-step arrays plus episode-level
metadata. We store both RGB frames (uint8, the image-space data the world model
trains on) and the 11-d semantic state (the readout used for coverage metrics).

HDF5 is one file per shard, with each episode as a top-level group. This lets
us stream episodes lazily without loading the whole D_pre into memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


@dataclass
class Trajectory:
    rgb: np.ndarray            # (T+1, H, W, 3) uint8 — includes the initial frame
    obs: np.ndarray            # (T+1, 39) float32
    semantic: np.ndarray       # (T+1, 11) float32
    actions: np.ndarray        # (T, 4) float32
    rewards: np.ndarray        # (T,) float32
    dones: np.ndarray          # (T,) bool — terminated|truncated
    successes: np.ndarray      # (T,) float32
    qpos: np.ndarray           # (T+1, dim_q) float64
    qvel: np.ndarray           # (T+1, dim_q) float64
    target_pos: np.ndarray     # (3,) float64
    rand_vec: np.ndarray       # (6,) float64
    obj_init_pos: np.ndarray   # (3,) float64
    policy_source: str = "unknown"   # "scripted", "scripted_noisy", "random", "actor"
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def success(self) -> float:
        return float(self.successes.max()) if self.successes.size else 0.0


def save_trajectories(path: str | Path, trajs: Iterable[Trajectory]) -> int:
    """Write trajectories to a single HDF5 file. Returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with h5py.File(path, "w") as f:
        for i, tr in enumerate(trajs):
            g = f.create_group(f"ep_{i:06d}")
            g.create_dataset("rgb", data=tr.rgb, compression="gzip", compression_opts=4)
            g.create_dataset("obs", data=tr.obs)
            g.create_dataset("semantic", data=tr.semantic)
            g.create_dataset("actions", data=tr.actions)
            g.create_dataset("rewards", data=tr.rewards)
            g.create_dataset("dones", data=tr.dones)
            g.create_dataset("successes", data=tr.successes)
            g.create_dataset("qpos", data=tr.qpos)
            g.create_dataset("qvel", data=tr.qvel)
            g.attrs["target_pos"] = tr.target_pos
            g.attrs["rand_vec"] = tr.rand_vec
            g.attrs["obj_init_pos"] = tr.obj_init_pos
            g.attrs["policy_source"] = tr.policy_source
            for k, v in tr.meta.items():
                g.attrs[f"meta_{k}"] = v
            n += 1
    return n


def load_trajectories(path: str | Path) -> list[Trajectory]:
    path = Path(path)
    out: list[Trajectory] = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            out.append(
                Trajectory(
                    rgb=g["rgb"][:],
                    obs=g["obs"][:],
                    semantic=g["semantic"][:],
                    actions=g["actions"][:],
                    rewards=g["rewards"][:],
                    dones=g["dones"][:],
                    successes=g["successes"][:],
                    qpos=g["qpos"][:],
                    qvel=g["qvel"][:],
                    target_pos=np.asarray(g.attrs["target_pos"]),
                    rand_vec=np.asarray(g.attrs["rand_vec"]),
                    obj_init_pos=np.asarray(g.attrs["obj_init_pos"]),
                    policy_source=str(g.attrs.get("policy_source", "unknown")),
                )
            )
    return out


def iter_trajectories(path: str | Path):
    """Lazy iterator — one Trajectory at a time, never holding the whole shard."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            yield Trajectory(
                rgb=g["rgb"][:],
                obs=g["obs"][:],
                semantic=g["semantic"][:],
                actions=g["actions"][:],
                rewards=g["rewards"][:],
                dones=g["dones"][:],
                successes=g["successes"][:],
                qpos=g["qpos"][:],
                qvel=g["qvel"][:],
                target_pos=np.asarray(g.attrs["target_pos"]),
                rand_vec=np.asarray(g.attrs["rand_vec"]),
                obj_init_pos=np.asarray(g.attrs["obj_init_pos"]),
                policy_source=str(g.attrs.get("policy_source", "unknown")),
            )
