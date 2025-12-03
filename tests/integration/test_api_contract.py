from __future__ import annotations

import math

import gymnasium as gym

import gym_info


def test_api_contract_cartpole() -> None:
    env = gym.make("CartPole-v1")
    env = gym_info.attach(env, preset="classic_control", run_id="test-run")

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    metrics = gym_info.entropies(env)
    assert isinstance(metrics, gym_info.Entropies)

    for value in (metrics.H_S, metrics.H_A, metrics.H_A_given_S):
        assert isinstance(value, float)
        assert value >= 0.0
        assert math.isfinite(value)

    assert metrics.H_A_given_S <= metrics.H_A + 1e-6

    summ = gym_info.summary(env)
    assert isinstance(summ, gym_info.Summary)
    assert summ.num_steps >= 1
    assert summ.num_episodes >= 0

    gym_info.print_table(summ)
    gym_info.plot_entropies(env)
