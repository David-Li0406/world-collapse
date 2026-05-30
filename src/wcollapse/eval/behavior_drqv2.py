"""Goal-shift behavioural evaluation for the DrQ-v2 + dm_env path.

Same semantics as the SAC-era `behavior.py`: run the current agent in two
sub-regions of the goal space — `trained` (matched to the bias_goal data
distribution) and `holdout` (the complement) — and report success-rate,
mean return, and obj-to-goal distance separately for each.

This version drives a dm_env-style `eval_env` whose innermost wrapper
exposes `set_goal_subregion(low, high)`, and acts with a DrQ-v2 agent on
stacked uint8 CHW observations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _drill_to_inner(env):
    cur = env
    while not hasattr(cur, "set_goal_subregion"):
        cur = cur._env
    return cur


def _eval_region(
    eval_env,
    agent,
    goal_subregion: tuple[np.ndarray, np.ndarray],
    n_episodes: int,
    global_step: int,
) -> dict[str, float]:
    inner = _drill_to_inner(eval_env)
    inner.set_goal_subregion(*goal_subregion)
    successes: list[float] = []
    final_dists: list[float] = []
    min_dists: list[float] = []
    returns: list[float] = []
    for _ in range(n_episodes):
        ts = eval_env.reset()
        cum_reward = 0.0
        ep_success = 0
        episode_min = float("inf")
        while not ts.last():
            with torch.no_grad():
                action = agent.act(ts.observation, global_step, eval_mode=True)
            ts = eval_env.step(action)
            cum_reward += ts.reward
            ep_success += ts.success
            state = getattr(ts, "state", None)
            if state is not None:
                obj = state[4:7]
                goal = state[36:39]
                episode_min = min(episode_min, float(np.linalg.norm(obj - goal)))
        successes.append(1.0 if ep_success >= 1.0 else 0.0)
        state = getattr(ts, "state", None)
        if state is not None:
            obj = state[4:7]
            goal = state[36:39]
            final_dists.append(float(np.linalg.norm(obj - goal)))
        else:
            final_dists.append(0.0)
        min_dists.append(episode_min if episode_min != float("inf") else 0.0)
        returns.append(cum_reward)
    inner.set_goal_subregion(None, None)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_final_dist": float(np.mean(final_dists)),
        "mean_min_dist": float(np.mean(min_dists)),
        "n_eval_episodes": float(len(successes)),
    }


def goal_shift_eval(
    eval_env,
    agent,
    trained_subregion: tuple[np.ndarray, np.ndarray],
    holdout_subregion: tuple[np.ndarray, np.ndarray],
    n_eval_episodes: int,
    global_step: int,
) -> dict[str, float]:
    trained = _eval_region(eval_env, agent, trained_subregion, n_eval_episodes, global_step)
    holdout = _eval_region(eval_env, agent, holdout_subregion, n_eval_episodes, global_step)
    out: dict[str, float] = {}
    for k, v in trained.items():
        out[f"trained_{k}"] = v
    for k, v in holdout.items():
        out[f"holdout_{k}"] = v
    out["shift_success_drop"] = trained["success_rate"] - holdout["success_rate"]
    return out
