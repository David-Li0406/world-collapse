"""Single-task Metaworld wrapper for the world-collapse experiment.

Constructs the underlying `SawyerXYZEnv` directly (bypassing `make_mt_envs`'s
gymnasium wrapper stack) so we have unmediated access to `set_env_state`,
`_last_rand_vec`, and `_target_pos`. Returns RGB frames plus the documented
semantic state (hand_xyz, obj_xyz, goal_xyz, gripper, obj_to_goal_dist) on every
step.

Two features the rest of the pipeline relies on:

- ``reset(goal_subregion=...)`` restricts the sampled goal to a sub-Box of the
  task's goal range, so we can bias online data collection to a "trained
  subregion" and reserve the complement for goal-shift evaluation.
- ``reset(sim_state=...)`` deterministically restores ``(qpos, qvel)`` + the
  goal site, so the probe bank can replay a saved state under fixed actions
  and produce identical sim trajectories. (See plan §Module 3.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium.spaces import Box

import metaworld
# NOTE: we no longer hard-import a single env class. Any Meta-world v3 task whose
# random_reset_space is laid out as [obj_xyz(3), goal_xyz(3)] works (push-v3,
# coffee-push-v3, ...). The concrete class is resolved at runtime via MT1.

# Indices into the 39-element observation produced by SawyerXYZEnv._get_obs.
# See Metaworld/metaworld/sawyer_xyz_env.py:475-527.
_OBS_HAND = slice(0, 3)
_OBS_GRIPPER = 3
_OBS_OBJ = slice(4, 7)
_OBS_GOAL = slice(36, 39)


@dataclass
class ProbeState:
    """Everything needed to deterministically restore a probe start state."""

    qpos: np.ndarray
    qvel: np.ndarray
    target_pos: np.ndarray
    obj_init_pos: np.ndarray
    rand_vec: np.ndarray


def semantic_from_obs(obs: np.ndarray) -> np.ndarray:
    """Extract the task-relevant low-dimensional state from a 39-d obs.

    Returns a length-11 vector: hand(3) + obj(3) + goal(3) + gripper(1) + dist(1).
    """
    hand = obs[_OBS_HAND]
    obj = obs[_OBS_OBJ]
    goal = obs[_OBS_GOAL]
    gripper = np.array([obs[_OBS_GRIPPER]])
    dist = np.array([np.linalg.norm(obj - goal)])
    return np.concatenate([hand, obj, goal, gripper, dist]).astype(np.float32)


SEMANTIC_DIM = 11
SEMANTIC_NAMES = (
    "hand_x", "hand_y", "hand_z",
    "obj_x", "obj_y", "obj_z",
    "goal_x", "goal_y", "goal_z",
    "gripper",
    "obj_to_goal",
)


class MetaworldVisualEnv:
    """Push-v3 env returning RGB + low-dim obs + semantic + sim-state info.

    Not a strict gym.Env subclass — `step` returns a richer tuple that's more
    convenient for offline data writing. Use `as_gym()` if you need the
    standard 5-tuple.
    """

    # Tasks sharing push-v3's [obj_xyz, goal_xyz] reset layout. Verified that
    # coffee-push-v3 has the same 6-D rand_vec and a splittable goal-x span.
    TASK_NAMES = {"push-v3", "coffee-push-v3"}

    def __init__(
        self,
        task_name: str = "push-v3",
        seed: int | None = None,
        image_size: int = 64,
        camera_name: str = "corner",
        max_episode_steps: int = 200,
    ):
        if task_name not in self.TASK_NAMES:
            raise ValueError(
                f"Only {self.TASK_NAMES} are supported by the minimal setting; got {task_name}"
            )
        self.task_name = task_name
        self.image_size = image_size
        self.camera_name = camera_name
        self.max_episode_steps = max_episode_steps

        # MT1 gives us a benchmark with pre-sampled tasks and the env class.
        # We construct the env ourselves so we can address `unwrapped` semantics
        # directly (set_env_state, _last_rand_vec, _target_pos).
        bench = metaworld.MT1(task_name, seed=seed)
        env_cls = bench.train_classes[task_name]
        self._env = env_cls(
            render_mode="rgb_array",
            camera_name=camera_name,
            height=image_size,
            width=image_size,
        )
        # Set an initial task to satisfy assert_task_is_set; we'll override
        # _last_rand_vec on each reset as needed.
        initial_task = next(t for t in bench.train_tasks if t.env_name == task_name)
        self._env.set_task(initial_task)
        if seed is not None:
            self._env.seed(seed)
            self._env.action_space.seed(seed)

        self._steps = 0
        self._last_obs: np.ndarray | None = None

    # ----- gym-style spaces -----
    @property
    def action_space(self) -> Box:
        return self._env.action_space

    @property
    def observation_space(self) -> Box:
        return self._env.observation_space

    @property
    def goal_low(self) -> np.ndarray:
        # Slice the rand_vec range that corresponds to goal coords.
        return np.asarray(self._env._random_reset_space.low[3:], dtype=np.float64)

    @property
    def goal_high(self) -> np.ndarray:
        return np.asarray(self._env._random_reset_space.high[3:], dtype=np.float64)

    @property
    def obj_low(self) -> np.ndarray:
        return np.asarray(self._env._random_reset_space.low[:3], dtype=np.float64)

    @property
    def obj_high(self) -> np.ndarray:
        return np.asarray(self._env._random_reset_space.high[:3], dtype=np.float64)

    # ----- core API -----
    def reset(
        self,
        goal_subregion: tuple[np.ndarray, np.ndarray] | None = None,
        sim_state: ProbeState | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Reset the env. Returns a dict with 'rgb', 'obs', 'semantic', 'info'.

        Args:
            goal_subregion: optional (low_3, high_3) restricting goal sampling to
                this 3-D sub-box. obj initial position remains over the full obj
                range. Ignored if `sim_state` is provided.
            sim_state: optional ProbeState to restore (deterministic replay path).
            seed: optional seed for this reset.
        """
        if sim_state is not None:
            return self._reset_to_probe(sim_state)

        # Sample a rand_vec ourselves so we can restrict the goal sub-region.
        rng = np.random.default_rng(seed)
        obj_lo, obj_hi = self.obj_low, self.obj_high
        if goal_subregion is None:
            goal_lo, goal_hi = self.goal_low, self.goal_high
        else:
            goal_lo, goal_hi = np.asarray(goal_subregion[0]), np.asarray(goal_subregion[1])
        obj = rng.uniform(obj_lo, obj_hi)
        # Match push-v3's "object not too close to goal" rejection (see
        # sawyer_push_v3.py:145-147). 0.15 is the threshold there.
        for _ in range(10):
            goal = rng.uniform(goal_lo, goal_hi)
            if np.linalg.norm(obj[:2] - goal[:2]) >= 0.15:
                break
        rand_vec = np.concatenate([obj, goal])
        self._env._freeze_rand_vec = True
        self._env._last_rand_vec = rand_vec
        obs, info = self._env.reset(seed=seed)
        self._steps = 0
        self._last_obs = obs
        return self._pack(obs, info, reward=0.0, terminated=False, truncated=False)

    def _reset_to_probe(self, probe: ProbeState) -> dict[str, Any]:
        # Trigger a normal reset first so all internal state (e.g. init_tcp,
        # objHeight, maxPushDist) is recomputed for the probe's rand_vec.
        self._env._freeze_rand_vec = True
        self._env._last_rand_vec = probe.rand_vec
        self._env.reset()
        # Now overwrite qpos/qvel and the goal site to exactly match the probe.
        self._env.set_env_state((probe.qpos.copy(), probe.qvel.copy()))
        self._env._target_pos = probe.target_pos.copy()
        self._env.model.site("goal").pos = probe.target_pos.copy()
        # Mujoco re-step is harmless and forces consistent derived quantities.
        obs = self._env._get_obs()
        self._steps = 0
        self._last_obs = obs
        return self._pack(obs, {"probe_replay": True}, reward=0.0, terminated=False, truncated=False)

    def step(self, action: np.ndarray) -> dict[str, Any]:
        _dbg = os.environ.get("WCOLLAPSE_STEP_DEBUG")
        if _dbg:
            import time as _t
            _t0 = _t.time()
            obs, reward, terminated, truncated, info = self._env.step(action.astype(np.float32))
            _finite = bool(np.isfinite(obs).all())
            print(f"[stepdbg] t={self._steps} phys_dt={_t.time()-_t0:.3f}s finite={_finite}", flush=True)
        else:
            obs, reward, terminated, truncated, info = self._env.step(action.astype(np.float32))
        self._steps += 1
        if self._steps >= self.max_episode_steps:
            truncated = True
        # Guard: a degenerate physics state (non-finite obs) hangs the EGL
        # renderer (NaN geom positions). Terminate this episode before rendering.
        if not np.isfinite(obs).all():
            print(f"[stepdbg] NON-FINITE obs at t={self._steps}; terminating episode pre-render", flush=True)
            truncated = True
            self._last_obs = obs
            packed = self._pack_safe(obs, info, reward=float(reward),
                                     terminated=bool(terminated), truncated=True)
            return packed
        if _dbg:
            import time as _t
            _t1 = _t.time()
            packed = self._pack(obs, info, reward=float(reward),
                                terminated=bool(terminated), truncated=bool(truncated))
            print(f"[stepdbg] t={self._steps} render_dt={_t.time()-_t1:.3f}s", flush=True)
            self._last_obs = obs
            return packed
        self._last_obs = obs
        return self._pack(obs, info, reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))

    def _pack_safe(self, obs, info, reward, terminated, truncated) -> dict[str, Any]:
        """Like _pack but reuses the last good frame instead of rendering a
        degenerate (non-finite) state, which can hang the renderer."""
        rgb = getattr(self, "_last_rgb", None)
        if rgb is None:
            rgb = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        safe_obs = np.nan_to_num(np.asarray(obs, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "obs": safe_obs,
            "rgb": rgb,
            "semantic": semantic_from_obs(safe_obs),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "probe_state": self.get_probe_state(),
        }

    def render(self) -> np.ndarray:
        """Return a uint8 HxWx3 RGB frame from the configured camera.

        Mujoco renders top-down then flips with a negative-stride view; we
        force a contiguous copy so downstream torch/HDF5 paths don't trip.
        """
        rgb = self._env.render()
        rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        self._last_rgb = rgb
        return rgb

    def get_probe_state(self) -> ProbeState:
        """Snapshot enough state to replay this moment deterministically."""
        qpos, qvel = self._env.get_env_state()
        return ProbeState(
            qpos=np.array(qpos, copy=True),
            qvel=np.array(qvel, copy=True),
            target_pos=np.array(self._env._target_pos, copy=True),
            obj_init_pos=np.array(self._env.obj_init_pos, copy=True),
            rand_vec=np.array(self._env._last_rand_vec, copy=True),
        )

    # ----- internals -----
    def _pack(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        return {
            "obs": obs.astype(np.float32),
            "rgb": self.render(),
            "semantic": semantic_from_obs(obs),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "probe_state": self.get_probe_state(),
        }


def make_env(
    task_name: str = "push-v3",
    seed: int | None = None,
    image_size: int = 64,
    camera_name: str = "corner",
    max_episode_steps: int = 200,
) -> MetaworldVisualEnv:
    return MetaworldVisualEnv(
        task_name=task_name,
        seed=seed,
        image_size=image_size,
        camera_name=camera_name,
        max_episode_steps=max_episode_steps,
    )
