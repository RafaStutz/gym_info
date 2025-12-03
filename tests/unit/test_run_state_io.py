import gymnasium as gym
import pytest

import gym_info
from gym_info.api import save_run_state, load_run_state


@pytest.fixture
def cartpole_env() -> gym.Env:
    """
    Instrumented CartPole environment for run-state I/O tests.
    """
    env = gym.make("CartPole-v1")
    wrapped = gym_info.attach(
        env,
        env_id="CartPole-v1",
        run_id="io-test-run",
        obs_bins=10,
        action_bins=None,
    )
    return wrapped


def test_save_and_load_run_state_roundtrip(cartpole_env: gym.Env, tmp_path) -> None:
    """
    save_run_state and load_run_state must perform a lossless roundtrip.


    """
    obs, info = cartpole_env.reset(seed=0)
    for _ in range(30):
        action = cartpole_env.action_space.sample()
        obs, reward, terminated, truncated, info = cartpole_env.step(action)
        if terminated or truncated:
            obs, info = cartpole_env.reset()

    original_buffers = cartpole_env.buffers

    path = tmp_path / "run_state.pkl"
    save_run_state(cartpole_env, path)

    raw_env_2 = gym.make("CartPole-v1")
    env_2 = gym_info.attach(raw_env_2)
    load_run_state(env_2, path)

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


def test_load_run_state_missing_file_raises(cartpole_env: gym.Env, tmp_path) -> None:
    """
    load_run_state must raise FileNotFoundError when the file is missing.

    """
    missing_path = tmp_path / "nonexistent.pkl"

    with pytest.raises(FileNotFoundError):
        load_run_state(cartpole_env, missing_path)
