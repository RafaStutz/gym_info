from __future__ import annotations

import gymnasium as gym

import gym_info

from gym_info.wrappers.info_wrapper import InfoMeasureWrapper


def test_collection_counts_steps_and_episodes() -> None:
    env = gym.make("CartPole-v1")
    env = gym_info.attach(env, preset="classic_control", run_id="test-run")

    steps = 0
    episodes = 0
    obs, info = env.reset(seed=0)

    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()

    assert isinstance(env, InfoMeasureWrapper)
    assert env.buffers.num_steps == steps
    assert env.buffers.num_episodes == episodes
