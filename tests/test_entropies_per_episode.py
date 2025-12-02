from __future__ import annotations

import math

import gymnasium as gym

import gym_info


def _run_cartpole_episodes(
    num_episodes: int,
    *,
    max_steps_per_episode: int = 200,
) -> gym.Env:
    """
    Helper that runs a number of CartPole episodes with a random policy.

    Returns the instrumented environment so that its buffers can be
    inspected by gym_info.
    """
    env = gym.make("CartPole-v1")
    env = gym_info.attach(env, preset="classic_control", run_id="test-episodes")

    for _ in range(num_episodes):
        obs, info = env.reset()
        for _ in range(max_steps_per_episode):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    return env


def test_entropies_per_episode_empty_when_no_episodes() -> None:
    """
    If no episode has been completed, entropies_per_episode should return an empty list.
    """
    env = gym.make("CartPole-v1")
    env = gym_info.attach(env, preset="classic_control", run_id="no-episodes")

    episode_entropies = gym_info.entropies_per_episode(env)

    assert isinstance(episode_entropies, list)
    assert episode_entropies == []


def test_entropies_per_episode_lengths_and_types() -> None:
    """
    entropies_per_episode should return one entry per completed episode,
    each entry being an instance of gym_info.Entropies.
    """
    env = _run_cartpole_episodes(num_episodes=5)
    episode_entropies = gym_info.entropies_per_episode(env)
    summary = gym_info.summary(env)

    assert isinstance(episode_entropies, list)
    assert len(episode_entropies) == summary.num_episodes

    for ent in episode_entropies:
        assert isinstance(ent, gym_info.Entropies)


def test_entropies_per_episode_values_are_non_negative_and_finite() -> None:
    """
    All entropy values should be finite, non-negative, and H(A|S) <= H(A)
    for each episode (up to numerical tolerance).
    """
    env = _run_cartpole_episodes(num_episodes=3)
    episode_entropies = gym_info.entropies_per_episode(env)

    for ent in episode_entropies:
        # Non-negative and finite
        for value in (ent.H_S, ent.H_A, ent.H_A_given_S):
            assert isinstance(value, float)
            assert value >= 0.0
            assert math.isfinite(value)

        # Conditional entropy cannot exceed marginal entropy (up to small tolerance)
        assert ent.H_A_given_S <= ent.H_A + 1e-6
