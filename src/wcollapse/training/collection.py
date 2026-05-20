"""Trajectory collection helpers (used by both Phase A and Phase B).

`collect_seed_dataset` builds D_pre with a mix of scripted-with-noise and
random policies, sampled over the full goal range. `rollout_actor` runs the
Dreamer actor in the real env for an online iteration.

Both call the same `_run_episode` core that handles RGB + obs + semantic +
sim-state recording.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from wcollapse.data.trajectory import Trajectory
from wcollapse.envs.metaworld_env import MetaworldVisualEnv, semantic_from_obs


def _run_episode(
    env: MetaworldVisualEnv,
    action_fn: Callable[[dict, int], np.ndarray],
    *,
    goal_subregion: tuple[np.ndarray, np.ndarray] | None,
    policy_source: str,
    seed: int | None,
) -> Trajectory:
    """One episode. action_fn is called as action_fn(step_dict, step_idx) -> action."""
    out = env.reset(goal_subregion=goal_subregion, seed=seed)
    rgb = [out["rgb"]]
    obs = [out["obs"]]
    semantic = [out["semantic"]]
    qpos = [out["probe_state"].qpos.copy()]
    qvel = [out["probe_state"].qvel.copy()]

    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    successes: list[float] = []

    for t in range(env.max_episode_steps):
        a = np.asarray(action_fn(out, t), dtype=np.float32).reshape(4)
        a = np.clip(a, -1.0, 1.0)
        out = env.step(a)
        actions.append(a)
        rewards.append(out["reward"])
        success = float(out["info"].get("success", 0.0))
        successes.append(success)
        done = out["terminated"] or out["truncated"]
        dones.append(done)
        rgb.append(out["rgb"])
        obs.append(out["obs"])
        semantic.append(out["semantic"])
        qpos.append(out["probe_state"].qpos.copy())
        qvel.append(out["probe_state"].qvel.copy())
        if done:
            break

    start_probe = env.get_probe_state()  # reflects last state; we only need target_pos/rand_vec/obj_init_pos
    return Trajectory(
        rgb=np.stack(rgb).astype(np.uint8),
        obs=np.stack(obs).astype(np.float32),
        semantic=np.stack(semantic).astype(np.float32),
        actions=np.stack(actions).astype(np.float32) if actions else np.zeros((0, 4), np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        successes=np.asarray(successes, dtype=np.float32),
        qpos=np.stack(qpos).astype(np.float64),
        qvel=np.stack(qvel).astype(np.float64),
        target_pos=start_probe.target_pos,
        rand_vec=start_probe.rand_vec,
        obj_init_pos=start_probe.obj_init_pos,
        policy_source=policy_source,
    )


def collect_seed_dataset(
    env: MetaworldVisualEnv,
    n_episodes: int,
    scripted_fraction: float = 0.7,
    scripted_noise: float = 0.3,
    goal_subregion: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int = 0,
) -> list[Trajectory]:
    """Phase A data — mixed scripted+noise and random rollouts on the full goal range.

    The proposal's recommendation (§4.3): broad coverage via a mixture of
    expert demos and random exploration.
    """
    rng = np.random.default_rng(seed)
    from metaworld.policies.sawyer_push_v3_policy import SawyerPushV3Policy

    scripted = SawyerPushV3Policy()

    trajectories: list[Trajectory] = []
    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        use_scripted = rng.random() < scripted_fraction
        if use_scripted:
            def action_fn(step: dict, t: int, _rng=rng) -> np.ndarray:
                a = scripted.get_action(step["obs"].astype(np.float64))
                a = a + scripted_noise * _rng.standard_normal(4).astype(np.float32)
                return a
            source = "scripted_noisy"
        else:
            def action_fn(step: dict, t: int, _rng=rng) -> np.ndarray:
                return _rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
            source = "random"
        tr = _run_episode(
            env,
            action_fn,
            goal_subregion=goal_subregion,
            policy_source=source,
            seed=ep_seed,
        )
        trajectories.append(tr)
    return trajectories


def rollout_actor(
    env: MetaworldVisualEnv,
    world_model,                       # MiniWorldModel
    actor,                             # Actor
    n_episodes: int,
    *,
    goal_subregion: tuple[np.ndarray, np.ndarray] | None,
    exploration_noise: float = 0.1,
    deterministic: bool = False,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> list[Trajectory]:
    """Run the current actor in the real env. Used to collect D_online."""
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    trajectories: list[Trajectory] = []
    world_model.eval()
    actor.eval()
    for ep in range(n_episodes):
        def action_fn(step: dict, t: int) -> np.ndarray:
            rgb = step["rgb"]  # H, W, 3 uint8
            with torch.no_grad():
                latent = world_model.encode_frame(rgb[None])  # (1, D)
                a = actor.act(latent, deterministic=deterministic, noise_std=exploration_noise)
            return a.squeeze(0).cpu().numpy()
        ep_seed = int(rng.integers(0, 2**31 - 1))
        tr = _run_episode(
            env,
            action_fn,
            goal_subregion=goal_subregion,
            policy_source="actor",
            seed=ep_seed,
        )
        trajectories.append(tr)
    return trajectories
