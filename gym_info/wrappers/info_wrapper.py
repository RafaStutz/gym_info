from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


ObsArray = NDArray[np.float32]
Action = int | float | NDArray[np.float32]


@dataclass
class TrajectoryBuffers:
    """In-memory buffers for a single run of an instrumented environment."""

    observations: list[ObsArray] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    episode_start_indices: list[int] = field(default_factory=list)
    episode_end_indices: list[int] = field(default_factory=list)
    num_steps: int = 0
    num_episodes: int = 0


class InfoMeasureWrapper(gym.Wrapper):
    """Gymnasium wrapper that collects trajectories for information measures."""

    def __init__(
        self,
        env: gym.Env,
        *,
        env_id: str | None = None,
        run_id: str | None = None,
        preset: str | None = None,
    ) -> None:
        super().__init__(env)
        self.env_id: str | None = env_id
        self.run_id: str | None = run_id
        self.preset: str | None = preset

        self._buffers = TrajectoryBuffers()

    def reset(self, **kwargs: object) -> tuple[ObsArray, dict[str, object]]:
        obs, info = super().reset(**kwargs)

        obs_array = np.asarray(obs, dtype=np.float32)
        self._buffers.episode_start_indices.append(self._buffers.num_steps)

        return obs_array, dict(info)

    def step(
        self,
        action: Action,
    ) -> tuple[ObsArray, float, bool, bool, dict[str, object]]:
        self._buffers.actions.append(action)
        self._buffers.num_steps += 1

        obs, reward, terminated, truncated, info = super().step(action)
        obs_array = np.asarray(obs, dtype=np.float32)
        self._buffers.observations.append(obs_array)

        if terminated or truncated:
            self._buffers.episode_end_indices.append(self._buffers.num_steps)
            self._buffers.num_episodes += 1

        return obs_array, float(reward), bool(terminated), bool(truncated), dict(info)

    @property
    def buffers(self) -> TrajectoryBuffers:
        return self._buffers
