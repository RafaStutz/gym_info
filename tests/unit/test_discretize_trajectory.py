from __future__ import annotations

import gymnasium as gym
import numpy as np

import gym_info
from gym_info.discretization.envs import make_env_discretization
from gym_info.discretization.trajectory import DiscreteTrajectory, discretize_trajectory
from gym_info.wrappers.info_wrapper import InfoMeasureWrapper


def test_discretize_trajectory_cartpole_shapes_and_types() -> None:
    env = gym.make("CartPole-v1")
    env = gym_info.attach(env, preset="classic_control", run_id="test-run")

    # Collect a small trajectory
    obs, info = env.reset(seed=0)
    steps = 0
    max_steps = 50

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            obs, info = env.reset()

    # Ensure we are working with the wrapper and buffers are populated
    assert isinstance(env, InfoMeasureWrapper)
    assert env.buffers.num_steps == steps

    # Build discretization config from the environment
    config = make_env_discretization(
        env,
        preset="classic_control",
    )

    # Discretize the collected trajectory
    discrete = discretize_trajectory(
        buffers=env.buffers,
        config=config,
        action_space=env.action_space,
    )

    # Basic type and shape checks
    assert isinstance(discrete, DiscreteTrajectory)
    assert discrete.num_steps == env.buffers.num_steps
    assert discrete.num_episodes == env.buffers.num_episodes

    T = env.buffers.num_steps

    # States
    assert discrete.states.shape[0] == T
    assert discrete.states.dtype == np.int32

    # Actions
    assert discrete.actions.shape[0] == T
    assert discrete.actions.dtype == np.int32

    # Episode indices must be tuples and consistent with num_steps
    assert isinstance(discrete.episode_start_indices, tuple)
    assert isinstance(discrete.episode_end_indices, tuple)
    assert all(0 <= idx <= T for idx in discrete.episode_start_indices)
    assert all(0 <= idx <= T for idx in discrete.episode_end_indices)
