from __future__ import annotations

import gymnasium as gym
import pandas as pd
import pytest

import gym_info


@pytest.fixture
def cartpole_env() -> gym.Env:
    """
    Instrumented CartPole environment for notebook UX tests.
    """
    env = gym.make("CartPole-v1")
    wrapped = gym_info.attach(
        env,
        preset="classic_control",
        run_id="notebook-test-run",
    )
    return wrapped


def test_episode_entropy_series_and_dataframe(cartpole_env: gym.Env) -> None:
    """
    episode_entropy_series and episode_entropy_dataframe must return
    coherent data structures with matching lengths and consistent keys/
    columns when at least one episode has completed.

    The functions should also handle the case with no completed episodes
    by returning empty, but well-formed, structures.
    """

    series_empty = gym_info.episode_entropy_series(cartpole_env)
    df_empty = gym_info.episode_entropy_dataframe(cartpole_env)

    assert set(series_empty.keys()) == {"episode", "H_S", "H_A", "H_A_given_S"}
    assert isinstance(df_empty, pd.DataFrame)
    assert list(df_empty.columns) == ["episode", "H_S", "H_A", "H_A_given_S"]
    assert len(series_empty["episode"]) == 0
    assert len(df_empty) == 0

    obs, info = cartpole_env.reset(seed=0)
    while len(gym_info.entropies_per_episode(cartpole_env)) == 0:
        action = cartpole_env.action_space.sample()
        obs, reward, terminated, truncated, info = cartpole_env.step(action)
        if terminated or truncated:
            obs, info = cartpole_env.reset()

    series = gym_info.episode_entropy_series(cartpole_env)
    df = gym_info.episode_entropy_dataframe(cartpole_env)

    assert set(series.keys()) == {"episode", "H_S", "H_A", "H_A_given_S"}
    assert list(df.columns) == ["episode", "H_S", "H_A", "H_A_given_S"]

    n_episodes = len(series["episode"])
    assert n_episodes > 0
    assert len(series["H_S"]) == n_episodes
    assert len(series["H_A"]) == n_episodes
    assert len(series["H_A_given_S"]) == n_episodes
    assert len(df) == n_episodes

    assert all(isinstance(e, int) for e in series["episode"])
    assert series["episode"] == sorted(series["episode"])


def test_entropy_report_html_and_include_filter(cartpole_env: gym.Env) -> None:
    """
    The HTML report must render a non-empty HTML snippet for notebook
    environments, and the include parameter of as_html must control
    which entropy columns appear in the tables.
    """
    obs, info = cartpole_env.reset(seed=0)
    for _ in range(50):
        action = cartpole_env.action_space.sample()
        obs, reward, terminated, truncated, info = cartpole_env.step(action)
        if terminated or truncated:
            obs, info = cartpole_env.reset()

    rep = gym_info.report(cartpole_env)

    html_all = rep._repr_html_()
    assert isinstance(html_all, str)
    assert "gym_info entropy report" in html_all
    assert "Global entropies" in html_all
    assert "Per-episode entropies" in html_all
    assert cartpole_env.env_id in html_all

    html_Hs = rep.as_html(include=("H_S",))

    assert "H(S)" in html_Hs
    assert "H(A)" not in html_Hs
    assert "H(A | S)" not in html_Hs
