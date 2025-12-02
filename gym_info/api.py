from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import gymnasium as gym

from .wrappers.info_wrapper import InfoMeasureWrapper
from .discretization.envs import make_env_discretization
from .discretization.trajectory import discretize_trajectory
from .measures.histogram import (
    histogram_entropies,
    episode_histogram_entropies,
)

E = TypeVar("E", bound=gym.Env)


@dataclass(frozen=True)
class Summary:
    """Lightweight summary of an instrumented environment run."""

    env_id: str
    run_id: str | None
    num_episodes: int
    num_steps: int


@dataclass(frozen=True)
class Entropies:
    """Entropy metrics computed from the collected trajectory."""

    H_S: float
    H_A: float
    H_A_given_S: float


def attach(
    env: E,
    *,
    preset: str | None = None,
    env_id: str | None = None,
    run_id: str | None = None,
) -> gym.Env:
    """Attach the information-measure wrapper to an environment."""
    wrapped = InfoMeasureWrapper(
        env,
        env_id=env_id,
        run_id=run_id,
        preset=preset,
    )
    return wrapped
    

def _as_info_wrapper(env: gym.Env) -> InfoMeasureWrapper:
    if not isinstance(env, InfoMeasureWrapper):
        msg = "gym_info.entropies expects an environment returned by gym_info.attach()."
        raise TypeError(msg)
    return env


def entropies(env: gym.Env) -> Entropies:
    """
    Compute entropy metrics H(S), H(A) and H(A | S) for the current run.

    The environment must be one returned by :func:`gym_info.attach`. The
    entropies are computed by:
        1) building an EnvDiscretization from the Gymnasium spaces,
        2) discretizing the collected trajectory, and
        3) computing histogram-based entropies in bits.
    """
    wrapped = _as_info_wrapper(env)

    discretization = make_env_discretization(
        wrapped,
        env_id=wrapped.env_id,
        preset=wrapped.preset,
    )

    trajectory = discretize_trajectory(
        wrapped.buffers,
        config=discretization,
        action_space=wrapped.action_space,
    )

    hist = histogram_entropies(trajectory)

    return Entropies(
        H_S=hist.H_S,
        H_A=hist.H_A,
        H_A_given_S=hist.H_A_given_S,
    )


def entropies_per_episode(env: gym.Env) -> list[Entropies]:
    """
    Compute entropy metrics H(S), H(A) and H(A | S) for each episode.

    The environment must be one returned by :func:`gym_info.attach`. The
    returned list has length equal to the number of episodes observed in
    the current run, ordered chronologically.
    """
    wrapped = _as_info_wrapper(env)

    discretization = make_env_discretization(
        wrapped,
        env_id=wrapped.env_id,
        preset=wrapped.preset,
    )

    trajectory = discretize_trajectory(
        wrapped.buffers,
        config=discretization,
        action_space=wrapped.action_space,
    )

    hist_list = episode_histogram_entropies(trajectory)

    return [
        Entropies(
            H_S=h.H_S,
            H_A=h.H_A,
            H_A_given_S=h.H_A_given_S,
        )
        for h in hist_list
    ]



def summary(env: gym.Env) -> Summary:
    """
    Build a lightweight summary of the collected run.

    If the environment is not instrumented, the summary will contain zeros.
    """
    if isinstance(env, InfoMeasureWrapper):
        env_id = env.env_id
        if env_id is None:
            spec = getattr(env, "spec", None)
            if spec is not None:
                env_id = getattr(spec, "id", "") or ""
        run_id = env.run_id
        num_episodes = env.buffers.num_episodes
        num_steps = env.buffers.num_steps
        return Summary(
            env_id=env_id or "",
            run_id=run_id,
            num_episodes=num_episodes,
            num_steps=num_steps,
        )

    return Summary(env_id="", run_id=None, num_episodes=0, num_steps=0)


def print_table(summary: Summary) -> None:
    """Render a simple textual table for the given summary (MVP version)."""
    print("env_id       :", summary.env_id)
    print("run_id       :", summary.run_id)
    print("num_episodes :", summary.num_episodes)
    print("num_steps    :", summary.num_steps)


def plot_entropies(summary: Summary) -> None:
    """
    Placeholder for future plotting of entropy metrics.

    Kept as a stub in this MVP.
    """
    print("gym_info.plot_entropies (stub)")
