from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..wrappers.info_wrapper import TrajectoryBuffers
from .core import EnvDiscretization, IndexArray, discretize_box

_EMPTY_DIMENSION = 1
_SINGLE_ACTION_DIMENSION = 1


@dataclass(frozen=True)
class DiscreteTrajectory:
    """
    Discrete representation of a collected trajectory.

    """

    states: IndexArray
    actions: IndexArray
    episode_start_indices: tuple[int, ...]
    episode_end_indices: tuple[int, ...]
    num_steps: int
    num_episodes: int


def _empty_state_array(config: EnvDiscretization) -> IndexArray:
    """
    Return an empty state array with the correct second dimension.
    """
    if config.obs_config is None:
        return np.empty((0, _EMPTY_DIMENSION), dtype=np.int32)
    return np.empty((0, config.obs_config.ndim), dtype=np.int32)


def _empty_action_array(
    action_space: gym.Space, config: EnvDiscretization
) -> IndexArray:
    """
    Return an empty action array with the correct second dimension.
    """
    if isinstance(action_space, spaces.Discrete):
        return np.empty((0, _SINGLE_ACTION_DIMENSION), dtype=np.int32)

    if isinstance(action_space, spaces.Box) and config.action_config is not None:
        return np.empty((0, config.action_config.ndim), dtype=np.int32)

    if isinstance(action_space, spaces.Box) and config.action_config is None:
        return np.empty((0, _EMPTY_DIMENSION), dtype=np.int32)

    msg = f"Unsupported action space type: {type(action_space)}"
    raise TypeError(msg)


def _discretize_states_for_steps(
    buffers: TrajectoryBuffers,
    config: EnvDiscretization,
) -> IndexArray:
    """
    Build a (T, d_s) integer array of discretized states aligned with steps.

    The InfoMeasureWrapper stores observations at every reset and after each
    environment step. This means there are generally more observations than
    steps. Here we align observations and steps episode-by-episode so that
    each step t is paired with the observation at the beginning of that step.
    """
    T = buffers.num_steps
    if T == 0:
        return _empty_state_array(config)

    if config.obs_config is None:
        return np.empty((T, _EMPTY_DIMENSION), dtype=np.int32)

    if not buffers.observations:
        return np.empty((T, config.obs_config.ndim), dtype=np.int32)

    obs_array = np.stack(buffers.observations, axis=0)
    disc_obs = discretize_box(obs_array, config.obs_config)

    state_indices = np.empty((T,), dtype=np.int64)

    starts = list(buffers.episode_start_indices)
    ends = list(buffers.episode_end_indices)
    num_episodes = buffers.num_episodes

    obs_offset = 0

    for k in range(num_episodes):
        start = starts[k]
        end = ends[k]
        length = end - start
        if length <= 0:
            continue

        step_indices = np.arange(start, end, dtype=np.int64)
        local_indices = step_indices - start
        obs_indices = obs_offset + local_indices
        state_indices[start:end] = obs_indices

        obs_offset += length + 1

    if len(starts) > num_episodes:
        start = starts[-1]
        end = T
        length = end - start
        if length > 0:
            step_indices = np.arange(start, end, dtype=np.int64)
            local_indices = step_indices - start
            obs_indices = obs_offset + local_indices
            state_indices[start:end] = obs_indices

    states = disc_obs[state_indices, :]
    return states.astype(np.int32)


def _discretize_actions_for_steps(
    buffers: TrajectoryBuffers,
    config: EnvDiscretization,
    action_space: gym.Space,
) -> IndexArray:
    """
    Build a (T, d_a) integer array of discretized actions aligned with steps.
    """
    T = buffers.num_steps
    if T == 0:
        return _empty_action_array(action_space, config)

    actions = np.asarray(buffers.actions)

    if isinstance(action_space, spaces.Discrete):
        actions = actions.reshape(T, 1)
        return actions.astype(np.int32)

    if isinstance(action_space, spaces.Box):
        actions = np.asarray(actions, dtype=np.float32).reshape(T, -1)
        if config.action_config is None:
            return np.empty((T, _EMPTY_DIMENSION), dtype=np.int32)
        disc = discretize_box(actions, config.action_config)
        return disc.astype(np.int32)

    msg = f"Unsupported action space type: {type(action_space)}"
    raise TypeError(msg)


def discretize_trajectory(
    buffers: TrajectoryBuffers,
    config: EnvDiscretization,
    action_space: gym.Space,
) -> DiscreteTrajectory:
    """
    Discretize a collected trajectory using an EnvDiscretization.
    """
    states = _discretize_states_for_steps(buffers, config)
    actions = _discretize_actions_for_steps(buffers, config, action_space)

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
    Return a view of a single-episode slice [start, end) of a trajectory.
    """
    if start < 0 or end < start or end > trajectory.num_steps:
        msg = (
            f"Invalid slice [{start}, {end}) for trajectory of length "
            f"{trajectory.num_steps}"
        )
        raise ValueError(msg)

    length = end - start

    if trajectory.states.size == 0:
        state_dim = 0
    else:
        if trajectory.states.ndim != 2:
            msg = (
                f"states must have shape (T, d_s), got shape {trajectory.states.shape}"
            )
            raise ValueError(msg)
        state_dim = trajectory.states.shape[1]

    if trajectory.actions.size == 0:
        action_dim = 0
    else:
        if trajectory.actions.ndim != 2:
            msg = f"actions must have shape (T, d_a), got shape {trajectory.actions.shape}"
            raise ValueError(msg)
        action_dim = trajectory.actions.shape[1]

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
