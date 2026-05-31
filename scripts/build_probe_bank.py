#!/usr/bin/env python3
"""Build the shared probe bank for the DrQ-v2 pipeline.

Runs once per task. Collects a small batch of scripted+random trajectories
via the gym-style env (so we get qpos/qvel needed for deterministic replay),
then builds the probe bank with scripted action sequences.

Output: HDF5 at <shared_dir>/probe_bank.h5 — loaded by every Phase B run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

# Pre-load NVIDIA EGL ICD so glvnd dispatches to it (cdll.LoadLibrary primes
# the in-process ICD registry; otherwise glvnd silently falls back to Mesa
# which can't open /dev/dri on this runner).
import ctypes as _ctypes
import glob as _glob
for _libpath in sorted(_glob.glob("/opt/nvidia-*/lib64/libEGL_nvidia.so.0")):
    try:
        _ctypes.cdll.LoadLibrary(_libpath)
        print(f"[boot] preloaded NVIDIA EGL ICD: {_libpath}", flush=True)
        break
    except OSError as _exc:
        print(f"[boot] WARN preload {_libpath} failed: {_exc}", flush=True)

import numpy as np

from wcollapse.data.probe_bank import build_probe_bank, save_probe_bank
from wcollapse.envs.metaworld_env import MetaworldVisualEnv
from wcollapse.training.collection import collect_seed_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="push-v3")
    p.add_argument("--n_seed_episodes", type=int, default=100)
    p.add_argument("--n_probes", type=int, default=192)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="Output HDF5 path for the probe bank.")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = MetaworldVisualEnv(task_name=args.task, seed=args.seed)
    print(f"[probe_bank] collecting {args.n_seed_episodes} seed trajectories", flush=True)
    trajs = collect_seed_dataset(
        env=env,
        n_episodes=args.n_seed_episodes,
        scripted_fraction=0.7,
        scripted_noise=0.3,
        goal_subregion=None,
        seed=args.seed,
    )
    print(f"[probe_bank] building bank: n_probes={args.n_probes} horizon={args.horizon}", flush=True)
    bank = build_probe_bank(
        env=env,
        seed_trajectories=trajs,
        n_probes=args.n_probes,
        horizon=args.horizon,
        action_source="scripted",
        seed=args.seed,
    )
    save_probe_bank(out, bank)
    print(f"[probe_bank] wrote {len(bank)} probes to {out}", flush=True)


if __name__ == "__main__":
    main()
