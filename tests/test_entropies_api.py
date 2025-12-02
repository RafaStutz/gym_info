from __future__ import annotations

import math

import gymnasium as gym
import pytest

import gym_info


@pytest.mark.parametrize("env_id", ["CartPole-v1", "MountainCar-v0"])
def test_entropies_basic_properties_for_classic_control(env_id: str) -> None:
    env = gym.make(env_id)
    env = gym_info.attach(env, preset="classic_control", run_id="test-run")

    obs, info = env.reset(seed=0)

    # run a short random policy
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    metrics = gym_info.entropies(env)
    assert isinstance(metrics, gym_info.Entropies)

    for value in (metrics.H_S, metrics.H_A, metrics.H_A_given_S):
        assert isinstance(value, float)
        assert value >= 0.0
        assert math.isfinite(value)

    # information inequality: H(A | S) <= H(A)
    assert metrics.H_A_given_S <= metrics.H_A + 1e-6

    # for Discrete action spaces, H(A) is upper-bounded by log2(|A|)
    if hasattr(env.action_space, "n"):
        num_actions = env.action_space.n
        if num_actions > 0:
            max_entropy_bits = math.log2(num_actions)
            assert metrics.H_A <= max_entropy_bits + 1e-6

    # summary should reflect that some interaction happened
    summ = gym_info.summary(env)
    assert isinstance(summ, gym_info.Summary)
    assert summ.num_steps > 0
    assert summ.num_episodes >= 0
