"""Behavioral evaluation (proposal §4.5.3 — goal-shift adaptation).

We run the current actor in two goal subregions:
  * trained:  the sub-box used during online collection (matched distribution),
  * holdout:  the complementary sub-box reserved for evaluation (shifted goal).

The world-collapse signature requires the trained-region success to stay flat
while holdout success drops as the WM forgets the held-out region.

We compute success rate, mean final obj-to-goal distance, and minimum
obj-to-goal distance achieved during the episode.
"""

from __future__ import annotations

import numpy as np
import torch

from wcollapse.envs.metaworld_env import MetaworldVisualEnv
from wcollapse.models.actor import Actor
from wcollapse.models.ivideogpt_wrapper import MiniWorldModel


def _eval_region(
    env: MetaworldVisualEnv,
    world_model: MiniWorldModel,
    actor: Actor,
    goal_subregion: tuple[np.ndarray, np.ndarray],
    n_episodes: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    world_model.eval()
    actor.eval()
    successes: list[float] = []
    final_dists: list[float] = []
    min_dists: list[float] = []
    returns: list[float] = []
    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        out = env.reset(goal_subregion=goal_subregion, seed=ep_seed)
        success = 0.0
        cum_reward = 0.0
        episode_min = float("inf")
        for _ in range(env.max_episode_steps):
            rgb = out["rgb"]
            with torch.no_grad():
                z = world_model.encode_frame(rgb[None])
                action = actor.act(z, deterministic=True, noise_std=0.0)
            out = env.step(action.squeeze(0).cpu().numpy())
            cum_reward += out["reward"]
            success = max(success, float(out["info"].get("success", 0.0)))
            episode_min = min(episode_min, float(out["semantic"][10]))
            if out["terminated"] or out["truncated"]:
                break
        successes.append(success)
        final_dists.append(float(out["semantic"][10]))
        min_dists.append(episode_min if episode_min != float("inf") else 0.0)
        returns.append(cum_reward)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_final_dist": float(np.mean(final_dists)),
        "mean_min_dist": float(np.mean(min_dists)),
        "n_eval_episodes": float(len(successes)),
    }


def goal_shift_eval(
    env: MetaworldVisualEnv,
    world_model: MiniWorldModel,
    actor: Actor,
    trained_subregion: tuple[np.ndarray, np.ndarray],
    holdout_subregion: tuple[np.ndarray, np.ndarray],
    n_eval_episodes: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    trained = _eval_region(
        env, world_model, actor, trained_subregion, n_eval_episodes, device, seed
    )
    holdout = _eval_region(
        env, world_model, actor, holdout_subregion, n_eval_episodes, device, seed + 1
    )
    out: dict[str, float] = {}
    for k, v in trained.items():
        out[f"trained_{k}"] = v
    for k, v in holdout.items():
        out[f"holdout_{k}"] = v
    out["shift_success_drop"] = trained["success_rate"] - holdout["success_rate"]
    return out
