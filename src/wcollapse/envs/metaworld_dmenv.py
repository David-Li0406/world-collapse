"""dm_env-compatible adapter around our v3 `MetaworldVisualEnv`.

DrQ-v2, VideoPredictor, and the wmcollapse replay-buffer all expect the
dm_env API: `observation_spec()`, `action_spec()`, `reset()`/`step()`
returning an `ExtendedTimeStep`. This module provides exactly that on top of
our existing gym-style wrapper, so we can reuse the wmcollapse training stack
while keeping push-v3, `set_env_state`-based probe replay, and the
`goal_subregion` mechanism that drives the bias_goal condition.

The interface mirrors `wmcollapse/ivideogpt/mbrl/metaworld_env.py` exactly:
the same `make(name, frame_stack, action_repeat, seed, camera, duration,
succ_bonus)` factory, the same `ExtendedTimeStep` named tuple, the same
wrapper stack (`ActionDTypeWrapper -> ActionScaleWrapper -> FrameStackWrapper
-> ExtendedTimeStepWrapper`).
"""

from __future__ import annotations

from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_env import StepType, specs

from wcollapse.envs.metaworld_env import (
    MetaworldVisualEnv,
    ProbeState,
    semantic_from_obs,
)


# --- ExtendedTimeStep (identical to wmcollapse's; lets the replay buffer index
# by spec name) ----------------------------------------------------------------


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    success: Any
    state: Any = None

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)


# --- Inner v3 env -------------------------------------------------------------


class MetaWorldV3(dm_env.Environment):
    """dm_env wrapper around our v3 `MetaworldVisualEnv`.

    Returns HWC uint8 frames (FrameStackWrapper will transpose + stack).
    Implements `action_repeat` internally so the upstream pipeline keeps the
    same semantics as wmcollapse's `MetaWorld` class.
    """

    def __init__(
        self,
        name: str,
        seed: int | None = None,
        action_repeat: int = 1,
        size: tuple[int, int] = (64, 64),
        camera: str = "corner",
        duration: int = 100,
        succ_bonus: float = 0.0,
    ):
        self._inner = MetaworldVisualEnv(
            task_name=name,
            seed=seed,
            image_size=size[0],
            camera_name=camera,
            max_episode_steps=duration,
        )
        self._size = size
        self._action_repeat = action_repeat
        self._duration = duration
        self._succ_bonus = succ_bonus
        self._steps = None
        self._last_state = None
        self._goal_subregion: tuple[np.ndarray, np.ndarray] | None = None

    # ---- goal-subregion bias (used by the bias_goal mechanism) ----
    def set_goal_subregion(self, low: np.ndarray | None, high: np.ndarray | None) -> None:
        if low is None or high is None:
            self._goal_subregion = None
        else:
            self._goal_subregion = (np.asarray(low, np.float64), np.asarray(high, np.float64))

    @property
    def goal_low(self) -> np.ndarray:
        return self._inner.goal_low

    @property
    def goal_high(self) -> np.ndarray:
        return self._inner.goal_high

    # ---- dm_env API ----
    def observation_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=self._size + (3,),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def action_spec(self) -> specs.BoundedArray:
        space = self._inner.action_space
        return specs.BoundedArray(
            shape=space.shape,
            dtype=np.float32,
            minimum=float(space.low.min()),
            maximum=float(space.high.max()),
            name="action",
        )

    def reset(self) -> dm_env.TimeStep:
        out = self._inner.reset(goal_subregion=self._goal_subregion)
        self._steps = 0
        self._last_state = out["obs"].copy()
        return _StepWithExtras(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=out["rgb"],
            success=0.0,
            state=self._last_state,
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        assert self._steps is not None, "Must reset env first"
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        last_done = False
        for _ in range(self._action_repeat):
            out = self._inner.step(action)
            reward += float(out["reward"])
            info = out.get("info") or {}
            success += float(info.get("success", 0.0))
            last_done = bool(out["terminated"] or out["truncated"])
            self._last_state = out["obs"].copy()
            if last_done:
                break
        success = 1.0 if success >= 1.0 else 0.0
        if success == 1.0:
            reward += self._succ_bonus
        self._steps += 1
        terminal = last_done or (self._steps >= self._duration)
        if terminal:
            self._steps = None
        return _StepWithExtras(
            step_type=StepType.LAST if terminal else StepType.MID,
            reward=reward,
            discount=1.0,
            observation=out["rgb"],
            success=success,
            state=self._last_state,
        )

    # ---- probe / state extras (used by probe bank + coverage) ----
    def reset_to_probe(self, probe_state: ProbeState) -> dm_env.TimeStep:
        out = self._inner.reset(sim_state=probe_state)
        self._steps = 0
        self._last_state = out["obs"].copy()
        return _StepWithExtras(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=out["rgb"],
            success=0.0,
            state=self._last_state,
        )

    def get_probe_state(self) -> ProbeState:
        return self._inner.get_probe_state()

    def last_raw_obs(self) -> np.ndarray:
        return self._last_state


class _StepWithExtras(dm_env.TimeStep):
    """dm_env TimeStep extended with success + state attributes.

    dm_env.TimeStep is a NamedTuple — `_replace` creates a fresh instance that
    drops Python-level attributes. We override `_replace` so wrappers like
    FrameStackWrapper can swap the observation without losing extras.
    """

    def __new__(cls, step_type, reward, discount, observation, success, state):
        ts = dm_env.TimeStep.__new__(cls, step_type, reward, discount, observation)
        ts._success = success
        ts._state = state
        return ts

    def _replace(self, **kwargs):
        return _StepWithExtras(
            step_type=kwargs.get("step_type", self.step_type),
            reward=kwargs.get("reward", self.reward),
            discount=kwargs.get("discount", self.discount),
            observation=kwargs.get("observation", self.observation),
            success=kwargs.get("success", self._success),
            state=kwargs.get("state", self._state),
        )

    @property
    def success(self):
        return self._success

    @property
    def state(self):
        return self._state


# --- Wrapper stack (mirrors wmcollapse) ---------------------------------------


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        a = env.action_spec()
        self._action_spec = specs.BoundedArray(a.shape, dtype, a.minimum, a.maximum, "action")

    def step(self, action):
        return self._env.step(action.astype(self._env.action_spec().dtype))

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionScaleWrapper(dm_env.Environment):
    def __init__(self, env, minimum: float, maximum: float):
        self._env = env
        a = env.action_spec()
        orig_lo = np.asarray(a.minimum, np.float64)
        orig_hi = np.asarray(a.maximum, np.float64)
        scale = (orig_hi - orig_lo) / (maximum - minimum)
        self._transform = lambda x: (orig_lo + scale * (x - minimum)).astype(a.dtype, copy=False)
        self._action_spec = a.replace(minimum=minimum, maximum=maximum, dtype=np.float32)

    def step(self, action):
        return self._env.step(self._transform(action))

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    """Stack the last N frames along channel dim. Returns CHW uint8."""

    def __init__(self, env, num_frames: int):
        self._env = env
        self._n = num_frames
        self._frames: deque = deque([], maxlen=num_frames)
        ws = env.observation_spec().shape
        if len(ws) == 4:
            ws = ws[1:]
        self._obs_spec = specs.BoundedArray(
            shape=(ws[2] * num_frames, ws[0], ws[1]),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _extract(self, ts):
        rgb = ts.observation
        if rgb.ndim == 4:
            rgb = rgb[0]
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    def _wrap(self, ts):
        assert len(self._frames) == self._n
        stacked = np.concatenate(list(self._frames), axis=0)
        return ts._replace(observation=stacked)

    def reset(self):
        ts = self._env.reset()
        chw = self._extract(ts)
        for _ in range(self._n):
            self._frames.append(chw)
        return self._wrap(ts)

    def step(self, action):
        ts = self._env.step(action)
        self._frames.append(self._extract(ts))
        return self._wrap(ts)

    def reset_to_probe(self, probe_state: ProbeState):
        ts = self._env.reset_to_probe(probe_state)
        chw = self._extract(ts)
        for _ in range(self._n):
            self._frames.append(chw)
        return self._wrap(ts)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    """Normalize a dm_env TimeStep to an ExtendedTimeStep with action+success."""

    def __init__(self, env):
        self._env = env

    def reset(self):
        return self._aug(self._env.reset(), action=None)

    def step(self, action):
        return self._aug(self._env.step(action), action=action)

    def reset_to_probe(self, probe_state: ProbeState):
        return self._aug(self._env.reset_to_probe(probe_state), action=None)

    def _aug(self, ts, action):
        if action is None:
            action = np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype)
        return ExtendedTimeStep(
            step_type=ts.step_type,
            reward=getattr(ts, "reward", 0.0) or 0.0,
            discount=getattr(ts, "discount", 1.0) or 1.0,
            observation=ts.observation,
            action=action,
            success=getattr(ts, "success", 0.0),
            state=getattr(ts, "state", None),
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


# --- Factory (signature matches wmcollapse's metaworld_env.make) --------------


def make(
    name: str,
    frame_stack: int,
    action_repeat: int,
    seed: int,
    camera: str,
    duration: int,
    succ_bonus: float,
):
    """Returns a wrapped dm_env-compatible push-v3 env.

    Name handling: wmcollapse passes task names dashed (e.g. `button-press-topdown-wall`).
    We accept either `push-v3` or `push_v3` and normalize to push-v3.
    """
    norm = name.replace("_", "-")
    if norm in {"push", "push-v3"}:
        norm = "push-v3"
    elif norm in {"coffee-push", "coffee-push-v3"}:
        norm = "coffee-push-v3"
    env = MetaWorldV3(
        name=norm,
        action_repeat=action_repeat,
        seed=seed,
        size=(64, 64),
        camera=camera,
        duration=duration,
        succ_bonus=succ_bonus,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionScaleWrapper(env, minimum=-1.0, maximum=+1.0)
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env
