from __future__ import annotations

from typing import Iterable

import gymnasium as gym
from gymnasium import spaces

from .core import EnvDiscretization, make_box_binning_from_space

# Default number of bins per dimension for continuous spaces
_DEFAULT_BINS_PER_DIMENSION = 20


def _resolve_env_id(env: gym.Env, explicit_env_id: str | None) -> str | None:
    """
    Resolve the environment identifier.

    Preference order:
    1. Explicit env_id passed by the user.
    2. env.spec.id if available.
    3. None otherwise.
    """
    if explicit_env_id is not None:
        return explicit_env_id

    spec = getattr(env, "spec", None)
    if spec is not None:
        env_id = getattr(spec, "id", None)
        if isinstance(env_id, str):
            return env_id

    return None


def _default_bins_for_box(
    space: spaces.Box,
    *,
    preset: str | None,
    env_id: str | None,
) -> int:
    """
    Choose a default number of bins per dimension for a Box space.

    For now this is a simple policy that returns a single global default.
    The preset and env_id parameters are accepted to keep the interface
    open for future environment-specific or preset-specific tuning.
    """
    _ = space
    _ = preset
    _ = env_id
    return _DEFAULT_BINS_PER_DIMENSION


def make_env_discretization(
    env: gym.Env,
    *,
    env_id: str | None = None,
    preset: str | None = None,
    obs_bins: int | Iterable[int] | None = None,
    action_bins: int | Iterable[int] | None = None,
) -> EnvDiscretization:
    """
    Build an EnvDiscretization configuration from a Gymnasium environment.

    The function inspects the observation and action spaces to decide how
    to discretize them:

    - Box spaces are discretized using uniform bins along each dimension.
    - Discrete spaces are treated as already discrete and left without a
      BoxBinningConfig (the corresponding config is set to None).

    Parameters
    ----------
    env:
        Gymnasium environment whose spaces will be discretized.
    env_id:
        Optional explicit environment identifier. If None, the function
        will attempt to use env.spec.id when available.
    preset:
        Optional preset name that can be used to choose sensible defaults
        for the number of bins. Currently only a simple global default is
        applied, but the parameter is kept to allow future extensions.
    obs_bins:
        Optional override for the number of bins in the observation space.
        If an integer, the same number of bins is used for all dimensions.
        If an iterable, it must have length equal to the dimensionality of
        the Box space. If None, a preset-based default is used.
    action_bins:
        Optional override for the number of bins in the action space, with
        the same semantics as obs_bins.

    Returns
    -------
    EnvDiscretization
        A configuration object describing how to discretize observations
        and actions for this environment.
    """
    resolved_env_id = _resolve_env_id(env, env_id)

    obs_space = env.observation_space
    if isinstance(obs_space, spaces.Box):
        effective_obs_bins: int | Iterable[int]
        if obs_bins is not None:
            effective_obs_bins = obs_bins
        else:
            effective_obs_bins = _default_bins_for_box(
                obs_space,
                preset=preset,
                env_id=resolved_env_id,
            )
        obs_config = make_box_binning_from_space(
            obs_space,
            n_bins=effective_obs_bins,
            clip=True,
        )
    else:
        obs_config = None

    act_space = env.action_space
    if isinstance(act_space, spaces.Box):
        effective_action_bins: int | Iterable[int]
        if action_bins is not None:
            effective_action_bins = action_bins
        else:
            effective_action_bins = _default_bins_for_box(
                act_space,
                preset=preset,
                env_id=resolved_env_id,
            )
        action_config = make_box_binning_from_space(
            act_space,
            n_bins=effective_action_bins,
            clip=True,
        )
    else:
        action_config = None

    return EnvDiscretization(
        env_id=resolved_env_id,
        obs_config=obs_config,
        action_config=action_config,
    )
