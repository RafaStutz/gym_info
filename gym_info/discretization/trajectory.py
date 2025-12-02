from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from ..wrappers.info_wrapper import TrajectoryBuffers
from .core import (
    EnvDiscretization,
    IndexArray,
    ObsArray,
    BoxBinningConfig,
    discretize_box,
)

_EMPTY_DIMENSION = 0
_SINGLE_ACTION_DIMENSION = 1


@dataclass(frozen=True)
class DiscreteTrajectory:
    """
    Discrete state/action sequences derived from a collected trajectory.

    Shapes
    ------
    states:
        Integer bin indices of shape (T, d_s), where T is the number of
        steps and d_s is the dimensionality of the (flattened) state.
    actions:
        Integer bin indices of shape (T, d_a), where d_a is the action
        dimensionality. For Discrete action spaces, d_a is 1.
    episode_start_indices:
        Tuple of indices into the time dimension (0 <= index <= T) marking
        the start of each episode.
    episode_end_indices:
        Tuple of indices into the time dimension (0 <= index <= T) marking
        the end (exclusive) of each episode.
    num_steps:
        Total number of steps recorded in the underlying buffers.
    num_episodes:
        Total number of episodes recorded in the underlying buffers.
    """

    states: IndexArray
    actions: IndexArray
    episode_start_indices: tuple[int, ...]
    episode_end_indices: tuple[int, ...]
    num_steps: int
    num_episodes: int


def _empty_state_array(config: EnvDiscretization) -> IndexArray:
    if config.obs_config is None:
        return np.empty((0, _EMPTY_DIMENSION), dtype=np.int32)
    return np.empty((0, config.obs_config.ndim), dtype=np.int32)


def _empty_action_array(
    action_space: gym.Space,
    config: EnvDiscretization,
) -> IndexArray:
    if isinstance(action_space, spaces.Discrete):
        return np.empty((0, _SINGLE_ACTION_DIMENSION), dtype=np.int32)

    if isinstance(action_space, spaces.Box) and config.action_config is not None:
        return np.empty((0, config.action_config.ndim), dtype=np.int32)

    return np.empty((0, _EMPTY_DIMENSION), dtype=np.int32)


def _discretize_states(
    buffers: TrajectoryBuffers,
    obs_config: BoxBinningConfig | None,
) -> IndexArray:
    if buffers.num_steps == 0:
        if obs_config is None:
            return np.empty((0, _EMPTY_DIMENSION), dtype=np.int32)
        return np.empty((0, obs_config.ndim), dtype=np.int32)

    if obs_config is None:
        msg = "Observation discretization is not configured for this environment."
        raise ValueError(msg)

    obs_list = buffers.observations
    obs_array: ObsArray = np.stack(obs_list, axis=0).astype(np.float32, copy=False)
    return discretize_box(obs_array, obs_config)


def _discretize_actions(
    buffers: TrajectoryBuffers,
    action_space: gym.Space,
    action_config: BoxBinningConfig | None,
) -> IndexArray:
    if buffers.num_steps == 0:
        if isinstance(action_space, spaces.Discrete):
            return np.empty((0, _SINGLE_ACTION_DIMENSION), dtype=np.int32)
        if isinstance(action_space, spaces.Box) and action_config is not None:
            return np.empty((0, action_config.ndim), dtype=np.int32)
        return np.empty((0, _EMPTY_DIMENSION), dtype=np.int32)

    actions_list = buffers.actions

    if isinstance(action_space, spaces.Discrete):
        action_indices = np.array([int(a) for a in actions_list], dtype=np.int32)
        return action_indices.reshape(-1, _SINGLE_ACTION_DIMENSION)

    if isinstance(action_space, spaces.Box):
        if action_config is None:
            msg = "Box action space requires an action discretization config."
            raise ValueError(msg)

        actions_array = np.asarray(actions_list, dtype=np.float32)
        if actions_array.ndim == 1:
            actions_array = actions_array.reshape(-1, 1)
        elif actions_array.ndim > 2:
            actions_array = actions_array.reshape(actions_array.shape[0], -1)

        return discretize_box(actions_array, action_config)

    msg = f"Unsupported action space type: {type(action_space)}"
    raise TypeError(msg)


def discretize_trajectory(
    buffers: TrajectoryBuffers,
    config: EnvDiscretization,
    action_space: gym.Space,
) -> DiscreteTrajectory:
    """
    Discretize a collected trajectory using an EnvDiscretization.

    Parameters
    ----------
    buffers:
        Raw trajectory buffers collected by the InfoMeasureWrapper.
    config:
        Discretization configuration for observations and actions.
    action_space:
        The action space of the environment. Used to decide how to treat
        actions (Discrete vs Box) when discretizing.

    Returns
    -------
    DiscreteTrajectory
        A discrete representation of the trajectory, suitable for building
        histograms and computing entropy-based metrics.
    """
    states = _discretize_states(buffers, config.obs_config)
    actions = _discretize_actions(buffers, action_space, config.action_config)

    episode_start_indices = tuple(buffers.episode_start_indices)
    episode_end_indices = tuple(buffers.episode_end_indices)

    return DiscreteTrajectory(
        states=states,
        actions=actions,
        episode_start_indices=episode_start_indices,
        episode_end_indices=episode_end_indices,
        num_steps=buffers.num_steps,
        num_episodes=buffers.num_episodes,
    )


def slice_trajectory(
    trajectory: DiscreteTrajectory,
    start: int,
    end: int,
) -> DiscreteTrajectory:
    """
    Build a DiscreteTrajectory corresponding to a contiguous slice [start, end).

    The resulting trajectory has a single episode covering [0, end - start).
    """
    if start < 0 or end < start or end > trajectory.num_steps:
        msg = (
            f"Invalid slice [{start}, {end}) for trajectory of length "
            f"{trajectory.num_steps}"
        )
        raise ValueError(msg)

    # Determine state/action dimensionality (may be zero in degenerate cases).
    if trajectory.states.size == 0:
        state_dim = 0
    else:
        if trajectory.states.ndim != 2:
            msg = f"states must have shape (T, d_s), got shape {trajectory.states.shape}"
            raise ValueError(msg)
        state_dim = trajectory.states.shape[1]

    if trajectory.actions.size == 0:
        action_dim = 0
    else:
        if trajectory.actions.ndim != 2:
            msg = f"actions must have shape (T, d_a), got shape {trajectory.actions.shape}"
            raise ValueError(msg)
        action_dim = trajectory.actions.shape[1]

    length = end - start

    if length == 0:
        empty_states = np.empty((0, state_dim), dtype=np.int32)
        empty_actions = np.empty((0, action_dim), dtype=np.int32)
        return DiscreteTrajectory(
            states=empty_states,
            actions=empty_actions,
            episode_start_indices=(0,),
            episode_end_indices=(0,),
            num_steps=0,
            num_episodes=1,
        )

    if state_dim == 0:
        states_slice = np.empty((length, 0), dtype=np.int32)
    else:
        states_slice = trajectory.states[start:end, :]

    if action_dim == 0:
        actions_slice = np.empty((length, 0), dtype=np.int32)
    else:
        actions_slice = trajectory.actions[start:end, :]

    return DiscreteTrajectory(
        states=states_slice,
        actions=actions_slice,
        episode_start_indices=(0,),
        episode_end_indices=(length,),
        num_steps=length,
        num_episodes=1,
    )

