import pytest
import gymnasium as gym
import gym_info


@pytest.fixture
def cartpole_env():
    env = gym.make("CartPole-v1")
    return gym_info.attach(
        env,
        env_id="CartPole-v1",
        run_id="test-run",
        obs_bins=10,
        action_bins=None,
    )


def test_attach_stores_discretization_config(cartpole_env):
    assert cartpole_env.discretization.obs_bins == 10
    assert cartpole_env.discretization.action_bins is None
    assert cartpole_env.env_id == "CartPole-v1"
    assert cartpole_env.run_id == "test-run"
