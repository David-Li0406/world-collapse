#!/usr/bin/env python3
"""Per-step timing probe: physics vs render, for one or more Meta-world tasks.

Isolates whether a task (e.g. coffee-push-v3) is a true hang or just slow, and
which part (mujoco physics step vs EGL render) dominates. Completes in ~1 min
even if a task is slow (bounded step count), unlike a full probe-bank build.

Run: python scripts/env_timing_probe.py --tasks push-v3 coffee-push-v3 --n_steps 50
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

# Preload NVIDIA EGL ICD (same as build_probe_bank) so glvnd uses it, not Mesa.
import ctypes as _ctypes
import glob as _glob
for _libpath in sorted(_glob.glob("/opt/nvidia-*/lib64/libEGL_nvidia.so.0")):
    try:
        _ctypes.cdll.LoadLibrary(_libpath)
        print(f"[boot] preloaded NVIDIA EGL ICD: {_libpath}", flush=True)
        break
    except OSError as exc:
        print(f"[boot] WARN preload {_libpath} failed: {exc}", flush=True)

import numpy as np

from wcollapse.envs.metaworld_env import MetaworldVisualEnv


def scripted_policy(task: str):
    if task == "coffee-push-v3":
        from metaworld.policies.sawyer_coffee_push_v3_policy import SawyerCoffeePushV3Policy
        return SawyerCoffeePushV3Policy()
    from metaworld.policies.sawyer_push_v3_policy import SawyerPushV3Policy
    return SawyerPushV3Policy()


def probe(task: str, n_steps: int):
    print(f"\n=== {task} ===", flush=True)
    t0 = time.time()
    env = MetaworldVisualEnv(task_name=task, seed=0)
    print(f"  construct: {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    out = env.reset(seed=0)
    print(f"  reset+first-render: {time.time()-t0:.2f}s  rgb={out['rgb'].shape}", flush=True)

    pol = scripted_policy(task)
    inner = env._env
    phys_t, rend_t = [], []
    for i in range(n_steps):
        a = np.asarray(pol.get_action(out["obs"].astype(np.float64)), np.float32).reshape(4)
        a = np.clip(a, -1.0, 1.0)
        t = time.time(); inner.step(a);            phys_t.append(time.time()-t)
        t = time.time(); rgb = inner.render();     rend_t.append(time.time()-t)
        out = env._pack(inner._get_obs() if hasattr(inner, "_get_obs") else out["obs"],
                        {}, reward=0.0, terminated=False, truncated=False)
        if i in (0, 4, n_steps - 1):
            print(f"  step {i:3d}: phys={phys_t[-1]*1000:7.1f}ms  render={rend_t[-1]*1000:7.1f}ms", flush=True)
    phys = np.array(phys_t) * 1000; rend = np.array(rend_t) * 1000
    print(f"  MEAN over {n_steps}: phys={phys.mean():7.1f}ms  render={rend.mean():7.1f}ms  "
          f"total={(phys.mean()+rend.mean()):7.1f}ms/step", flush=True)
    print(f"  MAX:               phys={phys.max():7.1f}ms  render={rend.max():7.1f}ms", flush=True)


def probe_collection(task: str, n_episodes: int):
    """Exercise the REAL collect_seed_dataset path (per-episode reset +
    get_probe_state) that stalled in the setup build."""
    from wcollapse.training.collection import collect_seed_dataset
    print(f"\n=== {task}: collect_seed_dataset({n_episodes} eps) ===", flush=True)
    env = MetaworldVisualEnv(task_name=task, seed=0)
    t0 = time.time()
    trajs = collect_seed_dataset(env=env, n_episodes=n_episodes,
                                 scripted_fraction=0.7, scripted_noise=0.3, seed=0)
    print(f"  collected {len(trajs)} trajs in {time.time()-t0:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["push-v3", "coffee-push-v3"])
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--collect_episodes", type=int, default=0,
                   help="If >0, also time collect_seed_dataset for this many episodes.")
    args = p.parse_args()
    for task in args.tasks:
        try:
            probe(task, args.n_steps)
            if args.collect_episodes > 0:
                probe_collection(task, args.collect_episodes)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"{task}: PROBE FAILED {ex!r}", flush=True)


if __name__ == "__main__":
    main()
