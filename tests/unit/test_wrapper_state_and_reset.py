import gymnasium as gym
import pytest

import gym_info


@pytest.fixture
def cartpole_env() -> gym.Env:
    """
    Instrumented CartPole environment for wrapper tests.

    The fixture returns an environment already wrapped by gym_info.attach,
    so tests can focus on the wrapper behaviour rather than on setup.
    """
    env = gym.make("CartPole-v1")
    wrapped = gym_info.attach(
        env,
        env_id="CartPole-v1",
        run_id="test-run",
        obs_bins=10,
        action_bins=None,
    )
    return wrapped


def test_reset_run_clears_buffers(cartpole_env: gym.Env) -> None:
    """
    After collecting some data, reset_run must clear all buffers and counters.

    This ensures that a user can start a fresh run on the same environment
    instance without residual trajectory data.
    """
    obs, info = cartpole_env.reset(seed=0)
    for _ in range(10):
        action = cartpole_env.action_space.sample()
        obs, reward, terminated, truncated, info = cartpole_env.step(action)
        if terminated or truncated:
            obs, info = cartpole_env.reset()

    buffers = cartpole_env.buffers
    assert buffers.num_steps > 0
    assert buffers.num_episodes >= 0
    assert buffers.observations
    assert buffers.actions

    cartpole_env.reset_run()

    buffers_after = cartpole_env.buffers
    assert buffers_after.num_steps == 0
    assert buffers_after.num_episodes == 0
    assert buffers_after.observations == []
    assert buffers_after.actions == []
    assert buffers_after.episode_start_indices == []
    assert buffers_after.episode_end_indices == []


def test_get_and_set_state_roundtrip(cartpole_env: gym.Env) -> None:
    """
    get_state and set_state must support a lossless roundtrip for logging data.

    We collect a non-trivial trajectory on one wrapped environment, snapshot
    its state, then restore that state into a fresh wrapped environment and
    compare identifiers and buffer statistics.
    """
    obs, info = cartpole_env.reset(seed=0)
    for _ in range(20):
        action = cartpole_env.action_space.sample()
        obs, reward, terminated, truncated, info = cartpole_env.step(action)
        if terminated or truncated:
            obs, info = cartpole_env.reset()

    original_buffers = cartpole_env.buffers
    original_state = cartpole_env.get_state()

    raw_env_2 = gym.make("CartPole-v1")
    env_2 = gym_info.attach(raw_env_2)
    env_2.set_state(original_state)

    restored_buffers = env_2.buffers

    assert env_2.env_id == cartpole_env.env_id
    assert env_2.run_id == cartpole_env.run_id

    assert restored_buffers.num_steps == original_buffers.num_steps
    assert restored_buffers.num_episodes == original_buffers.num_episodes

    assert len(restored_buffers.observations) == len(original_buffers.observations)
    assert len(restored_buffers.actions) == len(original_buffers.actions)

    assert (
        restored_buffers.episode_start_indices == original_buffers.episode_start_indices
    )
    assert restored_buffers.episode_end_indices == original_buffers.episode_end_indices
