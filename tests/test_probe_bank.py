"""Probe-bank determinism test (plan §Verification §2).

If `set_env_state(probe.start)` is not deterministic, the entire probe-bank
methodology is broken: we can't compare the WM's predicted rollout from a
saved state to a "ground truth" if the ground truth itself isn't reproducible.
This test fails loudly if Metaworld state restoration ever drifts.
"""

from __future__ import annotations

import numpy as np
import pytest

from wcollapse.envs.metaworld_env import make_env
from wcollapse.training.collection import collect_seed_dataset
from wcollapse.data.probe_bank import build_probe_bank


@pytest.mark.slow
def test_probe_bank_determinism():
    env = make_env(task_name="push-v3", seed=0, image_size=48, max_episode_steps=20)
    seed_trajs = collect_seed_dataset(env, n_episodes=2, seed=0)
    bank = build_probe_bank(
        env=env, seed_trajectories=seed_trajs, n_probes=2, horizon=5, action_source="random", seed=0
    )
    for probe in bank:
        # Replay twice from the saved start state under the same actions.
        out1 = env.reset(sim_state=probe.start)
        qpos1 = [out1["probe_state"].qpos.copy()]
        sem1 = [out1["semantic"].copy()]
        for h in range(probe.actions.shape[0]):
            stepped = env.step(probe.actions[h])
            qpos1.append(stepped["probe_state"].qpos.copy())
            sem1.append(stepped["semantic"].copy())
        out2 = env.reset(sim_state=probe.start)
        qpos2 = [out2["probe_state"].qpos.copy()]
        sem2 = [out2["semantic"].copy()]
        for h in range(probe.actions.shape[0]):
            stepped = env.step(probe.actions[h])
            qpos2.append(stepped["probe_state"].qpos.copy())
            sem2.append(stepped["semantic"].copy())
        for a, b in zip(qpos1, qpos2):
            assert np.allclose(a, b, atol=1e-6), "qpos diverged across identical replays"
        for a, b in zip(sem1, sem2):
            assert np.allclose(a, b, atol=1e-5), "semantic state diverged across identical replays"
