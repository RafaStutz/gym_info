from __future__ import annotations

from collections.abc import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_info.discretization.envs import make_env_discretization


class DummyBoxEnv(gym.Env):
    """
    Minimal environment with a Box observation space and Discrete action space.

    This class is used to test make_env_discretization in isolation,
    without depending on a full Gymnasium environment implementation.
    """

    metadata = {}

    def __init__(self, low: Iterable[float], high: Iterable[float]) -> None:
        super().__init__()
        low_array = np.asarray(low, dtype=np.float32)
        high_array = np.asarray(high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_array,
            high=high_array,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.spec = None

    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[override]
        raise NotImplementedError

    def step(self, action):  # type: ignore[override]
        raise NotImplementedError


def test_make_env_discretization_respects_scalar_obs_bins() -> None:
    """
    When obs_bins is an integer, the same bin count must be used in all
    observation dimensions.
    """
    env = DummyBoxEnv(low=[-1.0, -2.0, -3.0], high=[1.0, 2.0, 3.0])

    discretization = make_env_discretization(
        env,
        obs_bins=10,
    )

    assert discretization.obs_config is not None
    assert discretization.obs_config.ndim == 3
    assert discretization.obs_config.n_bins == (10, 10, 10)

    assert discretization.action_config is None


def test_make_env_discretization_respects_vector_obs_bins() -> None:
    """
    When obs_bins is a sequence, each dimension must receive its own bin
    count, in the same order as the flattened observation space.
    """
    env = DummyBoxEnv(low=[-1.0, -2.0, -3.0], high=[1.0, 2.0, 3.0])

    discretization = make_env_discretization(
        env,
        obs_bins=[4, 8, 16],
    )

    assert discretization.obs_config is not None
    assert discretization.obs_config.ndim == 3
    assert discretization.obs_config.n_bins == (4, 8, 16)

    assert discretization.action_config is None
