from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

from .wrappers.info_wrapper import InfoMeasureWrapper
from .wrappers.info_wrapper import DiscretizationConfig
from .discretization.envs import make_env_discretization
from .discretization.trajectory import discretize_trajectory
from .measures.histogram import (
    histogram_entropies,
    episode_histogram_entropies,
)


@dataclass(slots=True)
class Summary:
    """
    Lightweight summary of a logged run.

    This is intentionally minimal: it only tracks identifiers and basic
    episode/step counts, which are cheap to compute and useful for quick
    inspection and reporting.
    """

    env_id: str | None
    run_id: str | None
    num_episodes: int
    num_steps: int


@dataclass(slots=True)
class Entropies:
    """
    Container for entropy metrics of a run or an episode.

    All values are measured in bits.
    """

    H_S: float
    H_A: float
    H_A_given_S: float


def _infer_env_id(env: gym.Env) -> str:
    """
    Infer a human-readable env_id from a Gymnasium environment.

    """
    spec = getattr(env, "spec", None)
    env_id = getattr(spec, "id", None)
    if isinstance(env_id, str) and env_id:
        return env_id
    return env.__class__.__name__


def attach(
    env: gym.Env,
    *,
    env_id: str | None = None,
    run_id: str | None = None,
    preset: str | None = None,
    obs_bins: int = 10,
    action_bins: int = 10,
) -> InfoMeasureWrapper:
    """
    Attach gym_info logging to a Gymnasium environment.

    If `env` is already an InfoMeasureWrapper, it is returned unchanged.

    Parameters
    ----------
    env:
        Base Gymnasium environment to wrap.
    env_id:
        Optional identifier for the environment. If None, it is inferred
        from `env.spec.id` when available, falling back to the class name.
    run_id:
        Optional run identifier (e.g., for experiment tracking).
    preset:
        Named discretization preset (e.g. "classic_control").
    obs_bins:
        Default number of bins per continuous observation dimension.
    action_bins:
        Default number of bins per continuous action dimension.
    """
    if isinstance(env, InfoMeasureWrapper):
        return env

    if env_id is None:
        env_id = _infer_env_id(env)

    discretization = DiscretizationConfig(
        obs_bins=obs_bins,
        action_bins=action_bins,
    )

    wrapped = InfoMeasureWrapper(
        env,
        env_id=env_id,
        run_id=run_id,
        preset=preset,
        discretization=discretization,
    )
    return wrapped


def _as_info_wrapper(env: gym.Env) -> InfoMeasureWrapper:
    """
    Ensure that the given environment is an InfoMeasureWrapper instance.
    """
    if not isinstance(env, InfoMeasureWrapper):
        msg = (
            "gym_info functions such as entropies() expect an environment "
            "returned by gym_info.attach()."
        )
        raise TypeError(msg)
    return env


def summary(env: gym.Env) -> Summary:
    """
    Build a lightweight summary for the given instrumented environment.
    """
    wrapped = _as_info_wrapper(env)
    buffers = wrapped.buffers

    inferred_env_id: str | None = wrapped.env_id
    if inferred_env_id is None and getattr(wrapped.env, "spec", None) is not None:
        inferred_env_id = getattr(wrapped.env.spec, "id", None)

    return Summary(
        env_id=inferred_env_id,
        run_id=wrapped.run_id,
        num_episodes=buffers.num_episodes,
        num_steps=buffers.num_steps,
    )


def entropies(env: gym.Env) -> Entropies:
    """
    Compute global entropies H(S), H(A) and H(A | S) for the current run.

    The environment must be one returned by :func:`gym_info.attach`. The
    entropies are computed by:

    1. Building an EnvDiscretization from the Gymnasium spaces and the
       user-provided discretization configuration.
    2. Discretizing the collected trajectory.
    3. Computing histogram-based entropies in bits.
    """
    wrapped = _as_info_wrapper(env)

    discretization = make_env_discretization(
        wrapped.env,
        env_id=wrapped.env_id,
        preset=wrapped.preset,
        obs_bins=wrapped.discretization.obs_bins,
        action_bins=wrapped.discretization.action_bins,
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
        wrapped.env,
        env_id=wrapped.env_id,
        preset=wrapped.preset,
        obs_bins=wrapped.discretization.obs_bins,
        action_bins=wrapped.discretization.action_bins,
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


def episode_entropy_series(env: gym.Env) -> dict[str, list[float]]:
    """
    Build an episode-wise time series of entropy metrics for a run.

    Returns a simple dictionary that is easy to feed into plotting code
    or to construct a pandas.DataFrame.

    Keys
    ----
    episode:
        List of episode indices [0, 1, 2, ...].
    H_S:
        H(S) per episode.
    H_A:
        H(A) per episode.
    H_A_given_S:
        H(A | S) per episode.
    """
    ent_list = entropies_per_episode(env)

    episodes = list(range(len(ent_list)))
    H_S = [e.H_S for e in ent_list]
    H_A = [e.H_A for e in ent_list]
    H_A_given_S = [e.H_A_given_S for e in ent_list]

    return {
        "episode": episodes,
        "H_S": H_S,
        "H_A": H_A,
        "H_A_given_S": H_A_given_S,
    }


def episode_entropy_dataframe(env: gym.Env) -> pd.DataFrame:
    """
    Return a pandas.DataFrame with episode-wise entropy metrics.

    Columns
    -------
    episode:
        Episode index (int).
    H_S:
        H(S) per episode.
    H_A:
        H(A) per episode.
    H_A_given_S:
        H(A | S) per episode.
    """
    data = episode_entropy_series(env)
    return pd.DataFrame(data)


def save_run_state(env: gym.Env, path: str | Path) -> None:
    """
    Serialize the logging state of an instrumented environment to disk.

    This uses :meth:`InfoMeasureWrapper.get_state` internally, so only
    data collected by gym_info (states, actions, episode boundaries and
    metadata) is saved. The underlying Gymnasium environment state is
    not persisted.

    Parameters
    ----------
    env:
        Environment returned by :func:`gym_info.attach`.
    path:
        Filesystem path where the state should be written, typically
        with a ``.pkl`` or similar extension.
    """
    wrapped = _as_info_wrapper(env)
    state = wrapped.get_state()

    target = Path(path)
    with target.open("wb") as f:
        pickle.dump(state, f)


def load_run_state(env: gym.Env, path: str | Path) -> None:
    """
    Restore the logging state of an instrumented environment from disk.

    This is the inverse of :func:`save_run_state`. It uses
    :meth:`InfoMeasureWrapper.set_state` internally.

    Parameters
    ----------
    env:
        Environment returned by :func:`gym_info.attach`.
    path:
        Filesystem path previously written by :func:`save_run_state`.
    """
    wrapped = _as_info_wrapper(env)

    source = Path(path)
    with source.open("rb") as f:
        state = pickle.load(f)

    wrapped.set_state(state)


def print_table(summary: Summary) -> None:
    """
    Render a simple textual table for the given summary.
    """
    print("env_id       :", summary.env_id)
    print("run_id       :", summary.run_id)
    print("num_episodes :", summary.num_episodes)
    print("num_steps    :", summary.num_steps)


def plot_entropies(
    env: gym.Env,
    *,
    entropies: tuple[str, ...] = ("H_S", "H_A", "H_A_given_S"),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot entropy metrics as a function of episode index.

    This helper computes episode-wise entropies using
    :func:`gym_info.entropies_per_episode` and plots the selected
    metrics against the episode index.

    Parameters
    ----------
    env:
        Environment returned by :func:`gym_info.attach`.
    entropies:
        Tuple specifying which entropy curves to plot. Valid entries are
        ``"H_S"``, ``"H_A"``, ``"H_A_given_S"``.
    ax:
        Optional Matplotlib Axes object to plot into. If None, a new
        figure and axes are created.

    Returns
    -------
    (fig, ax):
        The Matplotlib Figure and Axes used for the plot.
    """
    data = episode_entropy_series(env)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = data["episode"]
    label_map = {
        "H_S": "H(S)",
        "H_A": "H(A)",
        "H_A_given_S": "H(A|S)",
    }

    valid_keys = {"H_S", "H_A", "H_A_given_S"}
    for key in entropies:
        if key not in valid_keys:
            msg = (
                f"Unknown entropy key {key!r}. "
                "Valid options are 'H_S', 'H_A', 'H_A_given_S'."
            )
            raise ValueError(msg)
        ax.plot(x, data[key], label=label_map.get(key, key))

    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy (bits)")
    if entropies:
        ax.legend()
    ax.set_title("Entropy over episodes")

    return fig, ax
