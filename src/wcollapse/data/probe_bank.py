"""Fixed probe bank — the cornerstone of the world-collapse measurement.

A probe consists of:
  - a saved sim-state (qpos, qvel) and goal,
  - a fixed action sequence ``a_{1:H}``,
  - the realized real-env rollout (RGB, semantic, qpos) under that action
    sequence, recorded once at bank construction.

At evaluation time we restore the start state in the env, run the world model
under the same actions, and compare predicted vs. realized. The "visited vs.
under-visited" partition is computed per checkpoint by the coverage module
from the current policy density — it is NOT baked into the bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from wcollapse.envs.metaworld_env import MetaworldVisualEnv, ProbeState
from wcollapse.data.trajectory import Trajectory


@dataclass
class Probe:
    probe_id: int
    start: ProbeState
    actions: np.ndarray         # (H, 4) float32
    gt_rgb: np.ndarray          # (H+1, image, image, 3) uint8
    gt_semantic: np.ndarray     # (H+1, 11) float32
    gt_qpos: np.ndarray         # (H+1, dim_q) float64
    source_episode: int         # which trajectory in D_pre we sourced the start state from
    source_step: int            # step index within that trajectory
    episode_stage: str          # approach / contact / push / settle


@dataclass
class ProbeBank:
    probes: list[Probe]
    horizon: int
    image_size: int

    def __len__(self) -> int:
        return len(self.probes)

    def __iter__(self):
        return iter(self.probes)


def _stage_label(semantic: np.ndarray) -> str:
    """Coarse stage tag based on object-to-goal distance + gripper opening.

    semantic is the 11-d vector defined in metaworld_env.semantic_from_obs.
    Indices: 6:9 = goal, 3:6 = obj, 10 = obj_to_goal_dist, 9 = gripper.
    """
    dist = float(semantic[10])
    gripper = float(semantic[9])
    if dist > 0.20:
        return "approach"
    if dist > 0.10:
        return "contact" if gripper < 0.6 else "approach"
    if dist > 0.05:
        return "push"
    return "settle"


def build_probe_bank(
    env: MetaworldVisualEnv,
    seed_trajectories: Sequence[Trajectory],
    n_probes: int = 256,
    horizon: int = 15,
    action_source: str = "scripted",
    seed: int = 0,
) -> ProbeBank:
    """Construct the probe bank from a seed dataset (D_pre).

    Each probe samples a random (episode, step) from `seed_trajectories`,
    restores that sim state in `env`, then either re-uses the original action
    sequence from that point (action_source="trajectory") or rolls a new
    sequence with a scripted policy (action_source="scripted") or random
    actions (action_source="random"). The realized rollout is recorded as
    ground truth.

    Stratification: we attempt to balance probes across the four episode
    stages so coverage of approach/contact/push/settle is even.
    """
    rng = np.random.default_rng(seed)
    if action_source == "scripted":
        # Imported lazily; otherwise the import would force metaworld policies
        # to load even when callers don't need scripted probes.
        from metaworld.policies.sawyer_push_v3_policy import SawyerPushV3Policy
        scripted = SawyerPushV3Policy()
    else:
        scripted = None

    # Sample candidate (ep, step) pairs first, oversample, then stratify.
    candidates: list[tuple[int, int, str]] = []
    for _ in range(n_probes * 4):
        ei = int(rng.integers(0, len(seed_trajectories)))
        tr = seed_trajectories[ei]
        if tr.length < horizon + 4:
            continue
        si = int(rng.integers(2, tr.length - horizon - 1))
        stage = _stage_label(tr.semantic[si])
        candidates.append((ei, si, stage))

    # Round-robin pick from each stage to get balanced strata.
    by_stage: dict[str, list[tuple[int, int]]] = {"approach": [], "contact": [], "push": [], "settle": []}
    for ei, si, stage in candidates:
        by_stage.setdefault(stage, []).append((ei, si))
    selection: list[tuple[int, int, str]] = []
    stage_keys = [k for k, v in by_stage.items() if v]
    if not stage_keys:
        raise RuntimeError("No candidates produced when stratifying probes.")
    cursors = {k: 0 for k in stage_keys}
    while len(selection) < n_probes:
        progressed = False
        for k in stage_keys:
            if cursors[k] < len(by_stage[k]):
                ei, si = by_stage[k][cursors[k]]
                cursors[k] += 1
                selection.append((ei, si, k))
                progressed = True
                if len(selection) >= n_probes:
                    break
        if not progressed:
            break

    probes: list[Probe] = []
    for pid, (ei, si, stage) in enumerate(selection):
        tr = seed_trajectories[ei]
        # Reconstruct the ProbeState from the trajectory frame.
        start = ProbeState(
            qpos=tr.qpos[si].copy(),
            qvel=tr.qvel[si].copy(),
            target_pos=tr.target_pos.copy(),
            obj_init_pos=tr.obj_init_pos.copy(),
            rand_vec=tr.rand_vec.copy(),
        )
        env.reset(sim_state=start)
        # Pick the action sequence.
        if action_source == "trajectory":
            actions = tr.actions[si : si + horizon].astype(np.float32)
        elif action_source == "scripted":
            assert scripted is not None
            # Roll forward in a small inner loop to harvest scripted actions
            # in this state. We snapshot before each step so the inner rollout
            # doesn't disturb the env's state — actually it WILL advance state,
            # which is fine because we then replay them as the ground truth
            # below from the same start.
            actions = np.zeros((horizon, 4), dtype=np.float32)
            obs = tr.obs[si].astype(np.float64)
            env.reset(sim_state=start)
            for h in range(horizon):
                a = scripted.get_action(obs).astype(np.float32)
                actions[h] = a
                stepped = env.step(a)
                obs = stepped["obs"].astype(np.float64)
        elif action_source == "random":
            actions = rng.uniform(-1.0, 1.0, size=(horizon, 4)).astype(np.float32)
        else:
            raise ValueError(f"Unknown action_source={action_source}")

        # Now replay deterministically to record the ground-truth rollout.
        out = env.reset(sim_state=start)
        gt_rgb = [out["rgb"]]
        gt_semantic = [out["semantic"]]
        gt_qpos = [start.qpos.copy()]
        for h in range(horizon):
            stepped = env.step(actions[h])
            gt_rgb.append(stepped["rgb"])
            gt_semantic.append(stepped["semantic"])
            gt_qpos.append(stepped["probe_state"].qpos.copy())
        probes.append(
            Probe(
                probe_id=pid,
                start=start,
                actions=actions,
                gt_rgb=np.stack(gt_rgb).astype(np.uint8),
                gt_semantic=np.stack(gt_semantic).astype(np.float32),
                gt_qpos=np.stack(gt_qpos).astype(np.float64),
                source_episode=ei,
                source_step=si,
                episode_stage=stage,
            )
        )
    return ProbeBank(probes=probes, horizon=horizon, image_size=env.image_size)


def save_probe_bank(path: str | Path, bank: ProbeBank) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["horizon"] = bank.horizon
        f.attrs["image_size"] = bank.image_size
        for p in bank.probes:
            g = f.create_group(f"probe_{p.probe_id:06d}")
            g.create_dataset("actions", data=p.actions)
            g.create_dataset("gt_rgb", data=p.gt_rgb, compression="gzip", compression_opts=4)
            g.create_dataset("gt_semantic", data=p.gt_semantic)
            g.create_dataset("gt_qpos", data=p.gt_qpos)
            g.create_dataset("start_qpos", data=p.start.qpos)
            g.create_dataset("start_qvel", data=p.start.qvel)
            g.attrs["target_pos"] = p.start.target_pos
            g.attrs["obj_init_pos"] = p.start.obj_init_pos
            g.attrs["rand_vec"] = p.start.rand_vec
            g.attrs["source_episode"] = p.source_episode
            g.attrs["source_step"] = p.source_step
            g.attrs["episode_stage"] = p.episode_stage


def load_probe_bank(path: str | Path) -> ProbeBank:
    path = Path(path)
    probes: list[Probe] = []
    with h5py.File(path, "r") as f:
        horizon = int(f.attrs["horizon"])
        image_size = int(f.attrs["image_size"])
        for key in sorted(f.keys()):
            g = f[key]
            start = ProbeState(
                qpos=np.asarray(g["start_qpos"][:]),
                qvel=np.asarray(g["start_qvel"][:]),
                target_pos=np.asarray(g.attrs["target_pos"]),
                obj_init_pos=np.asarray(g.attrs["obj_init_pos"]),
                rand_vec=np.asarray(g.attrs["rand_vec"]),
            )
            probes.append(
                Probe(
                    probe_id=int(key.split("_")[-1]),
                    start=start,
                    actions=g["actions"][:],
                    gt_rgb=g["gt_rgb"][:],
                    gt_semantic=g["gt_semantic"][:],
                    gt_qpos=g["gt_qpos"][:],
                    source_episode=int(g.attrs["source_episode"]),
                    source_step=int(g.attrs["source_step"]),
                    episode_stage=str(g.attrs["episode_stage"]),
                )
            )
    return ProbeBank(probes=probes, horizon=horizon, image_size=image_size)
