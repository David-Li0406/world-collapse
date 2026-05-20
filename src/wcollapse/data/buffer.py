"""Replay buffer with two sampling modes that differentiate experimental conditions.

The buffer holds (rgb, action, reward, semantic, done) transitions in memory.
Two condition-specific samplers:

- ``recent``  : FIFO window of the last ``window_size`` transitions. This is the
  collapse-prone setting from proposal §4.3 — recency-biased updates.
- ``uniform`` : uniform over the entire stored history (including the seeded
  D_pre). This is the balanced-replay baseline.

Both share the same write path; only the index sampler differs.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np

from wcollapse.data.trajectory import Trajectory

SamplingMode = Literal["recent", "uniform"]


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 1_000_000,
        window_size: int = 5_000,
        seq_len: int = 16,
        image_size: int = 64,
        mode: SamplingMode = "uniform",
    ):
        self.capacity = capacity
        self.window_size = window_size
        self.seq_len = seq_len
        self.image_size = image_size
        self.mode = mode

        # Episode-keyed storage: list of arrays, one per episode. Sampling
        # picks a (episode, start) pair such that episode has >= seq_len+1 frames.
        self._episodes: list[dict] = []
        self._total_transitions = 0
        self._recent_episode_ids: deque[int] = deque()  # episode indices in recency order
        self._recent_total = 0
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return self._total_transitions

    def add_trajectory(self, tr: Trajectory) -> None:
        T = tr.length
        if T < self.seq_len + 1:
            return  # too short to sample even one (context+target) window
        ep = {
            "rgb": tr.rgb,
            "obs": tr.obs,
            "semantic": tr.semantic,
            "actions": tr.actions,
            "rewards": tr.rewards,
            "dones": tr.dones,
            "length": T,
        }
        self._episodes.append(ep)
        idx = len(self._episodes) - 1
        self._total_transitions += T
        self._recent_episode_ids.append(idx)
        self._recent_total += T
        # Trim the recent window to roughly window_size transitions.
        while self._recent_total > self.window_size and len(self._recent_episode_ids) > 1:
            old = self._recent_episode_ids.popleft()
            self._recent_total -= self._episodes[old]["length"]
        # Trim total capacity (rare in practice with our sizes).
        while self._total_transitions > self.capacity and len(self._episodes) > 1:
            old = self._episodes.pop(0)
            self._total_transitions -= old["length"]
            # Shift recent indices since we popped from the front.
            self._recent_episode_ids = deque(i - 1 for i in self._recent_episode_ids if i > 0)

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample a batch of sequences of length ``seq_len`` (default ``self.seq_len``).

        Returns dict with:
            rgb       (B, L+1, H, W, 3) uint8 — context frame + L target frames
            actions   (B, L, 4)         float32
            rewards   (B, L)            float32
            semantic  (B, L+1, 11)      float32
            dones     (B, L)            bool
        """
        L = seq_len or self.seq_len
        candidate_ids = (
            list(self._recent_episode_ids) if self.mode == "recent" else list(range(len(self._episodes)))
        )
        if not candidate_ids:
            raise RuntimeError("ReplayBuffer is empty; cannot sample.")
        # Filter to episodes long enough.
        candidate_ids = [i for i in candidate_ids if self._episodes[i]["length"] >= L + 1]
        if not candidate_ids:
            raise RuntimeError(f"No episode in the active window is at least {L + 1} steps long.")
        ep_indices = self._rng.choice(candidate_ids, size=batch_size, replace=True)
        rgb_out = np.empty((batch_size, L + 1, self.image_size, self.image_size, 3), dtype=np.uint8)
        act_out = np.empty((batch_size, L, 4), dtype=np.float32)
        rew_out = np.empty((batch_size, L), dtype=np.float32)
        sem_out = np.empty((batch_size, L + 1, 11), dtype=np.float32)
        done_out = np.empty((batch_size, L), dtype=bool)
        for b, ei in enumerate(ep_indices):
            ep = self._episodes[ei]
            start = int(self._rng.integers(0, ep["length"] - L))  # inclusive
            rgb_out[b] = ep["rgb"][start : start + L + 1]
            act_out[b] = ep["actions"][start : start + L]
            rew_out[b] = ep["rewards"][start : start + L]
            sem_out[b] = ep["semantic"][start : start + L + 1]
            done_out[b] = ep["dones"][start : start + L]
        return {
            "rgb": rgb_out,
            "actions": act_out,
            "rewards": rew_out,
            "semantic": sem_out,
            "dones": done_out,
        }

    @property
    def total_transitions(self) -> int:
        return self._total_transitions

    @property
    def recent_transitions(self) -> int:
        return self._recent_total
