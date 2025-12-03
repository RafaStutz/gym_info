from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


ObsArray = NDArray[np.float32]
Action = int | float | NDArray[np.float32]


@dataclass(slots=True)
class DiscretizationConfig:
    """
    User-level discretization configuration for an instrumented environment.

    - obs_bins: controls the discretization of the observation space (when Box).
    - action_bins: controls the discretization of the action space (when Box).

    Semantics:
    - None  -> use the library defaults;
    - int   -> same number of bins for all dimensions;
    - sequence of int -> one number of bins per dimension (after flattening).
    """

    obs_bins: int | Sequence[int] | None = None
    action_bins: int | Sequence[int] | None = None


@dataclass
class TrajectoryBuffers:
    """
    In-memory buffers for a single run of an instrumented environment.

    The indices stored in ``episode_start_indices`` and
    ``episode_end_indices`` refer to the step count of the run.
    """

    observations: list[ObsArray] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    episode_start_indices: list[int] = field(default_factory=list)
    episode_end_indices: list[int] = field(default_factory=list)
    num_steps: int = 0
    num_episodes: int = 0


class InfoMeasureWrapper(gym.Wrapper):
    """
    Gymnasium Wrapper that logs (state, action) pairs and episode metadata
    for subsequent computation of information-theoretic quantities such as
    H(S), H(A), and H(A|S).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        env_id: str | None = None,
        run_id: str | None = None,
        preset: str | None = None,
        discretization: DiscretizationConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        env:
            Gymnasium environment to be instrumented.
        env_id:
            Logical identifier for the environment (e.g., "CartPole-v1").
        run_id:
            Logical identifier for this run (e.g., "seed-0", "run-01").
        preset:
            Name of a discretization preset (kept for future extensions).
        discretization:
            Discretization configuration chosen by the user. If None, an
            empty configuration is used and library defaults apply.
        """
        super().__init__(env)

        self.env_id: str | None = env_id
        self.run_id: str | None = run_id
        self.preset: str | None = preset

        self.discretization: DiscretizationConfig = (
            discretization if discretization is not None else DiscretizationConfig()
        )

        self._buffers: TrajectoryBuffers = TrajectoryBuffers()

    def reset(self, **kwargs) -> tuple[ObsArray, dict[str, object]]:
        """
        Reset the environment and start a new episode in the buffers.

        Note
        ----
        We choose to record the initial observation as part of the data;
        the ``num_steps`` counter still represents the number of calls
        to ``step``.
        """
        obs, info = super().reset(**kwargs)
        obs_array = np.asarray(obs, dtype=np.float32)

        self._buffers.episode_start_indices.append(self._buffers.num_steps)
        self._buffers.observations.append(obs_array)

        return obs_array, dict(info)

    def step(
        self,
        action: Action,
    ) -> tuple[ObsArray, float, bool, bool, dict[str, object]]:
        """
        Take a step in the environment, logging the action, next state,
        and episode boundaries.

        The internal buffers are updated so that downstream code can
        reconstruct state and action sequences as well as episode spans.
        """
        self._buffers.actions.append(action)

        self._buffers.num_steps += 1

        obs, reward, terminated, truncated, info = super().step(action)
        obs_array = np.asarray(obs, dtype=np.float32)
        self._buffers.observations.append(obs_array)

        if terminated or truncated:
            self._buffers.episode_end_indices.append(self._buffers.num_steps)
            self._buffers.num_episodes += 1

        return (
            obs_array,
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    def clear_buffers(self) -> None:
        """
        Clear all collected trajectory data for the current run.

        After calling this method, any previously recorded observations,
        actions, and episode boundaries are discarded. The next call to
        ``reset`` will start a fresh run from an empty buffer state.
        """
        self._buffers = TrajectoryBuffers()

    def reset_run(self) -> None:
        """
        Reset the current run by discarding all collected trajectory data.

        This is a convenience alias for :meth:`clear_buffers`, provided
        for semantic clarity when the caller thinks in terms of runs
        rather than raw buffers.
        """
        self.clear_buffers()

    def get_state(self) -> dict[str, object]:
        """
        Return a serializable snapshot of the logging state for this wrapper.

        The returned dictionary contains enough information to reconstruct
        the trajectory buffers and high-level identifiers via
        :meth:`set_state`. It is intended to be used together with an
        external persistence mechanism (e.g., writing to disk).
        """
        if self._buffers.observations:
            obs_array = np.stack(self._buffers.observations, axis=0)
        else:
            obs_array = np.zeros((0, 0), dtype=np.float32)

        actions_array = np.array(self._buffers.actions, dtype=object)
        start_indices = np.asarray(self._buffers.episode_start_indices, dtype=np.int64)
        end_indices = np.asarray(self._buffers.episode_end_indices, dtype=np.int64)

        state: dict[str, object] = {
            "env_id": self.env_id,
            "run_id": self.run_id,
            "preset": self.preset,
            "discretization_obs_bins": self.discretization.obs_bins,
            "discretization_action_bins": self.discretization.action_bins,
            "observations": obs_array,
            "actions": actions_array,
            "episode_start_indices": start_indices,
            "episode_end_indices": end_indices,
            "num_steps": self._buffers.num_steps,
            "num_episodes": self._buffers.num_episodes,
        }
        return state

    def set_state(self, state: dict[str, object]) -> None:
        """
        Restore the logging state of this wrapper from a snapshot.

        The input must be a dictionary previously produced by
        :meth:`get_state`. It restores trajectory buffers, identifiers,
        and discretization configuration. The underlying Gym environment
        itself is not reset or advanced by this method.
        """
        self.env_id = state.get("env_id")  # type: ignore[assignment]
        self.run_id = state.get("run_id")  # type: ignore[assignment]
        self.preset = state.get("preset")  # type: ignore[assignment]

        obs_bins = state.get("discretization_obs_bins")
        action_bins = state.get("discretization_action_bins")
        self.discretization = DiscretizationConfig(
            obs_bins=obs_bins,  # type: ignore[arg-type]
            action_bins=action_bins,  # type: ignore[arg-type]
        )

        obs_array = np.asarray(state.get("observations"), dtype=np.float32)
        if obs_array.size == 0:
            observations: list[ObsArray] = []
        else:
            observations = [obs_array[i] for i in range(obs_array.shape[0])]

        actions_array = np.asarray(state.get("actions"), dtype=object)
        actions: list[Action] = list(actions_array.tolist())

        start_indices_array = np.asarray(
            state.get("episode_start_indices"),
            dtype=np.int64,
        )
        end_indices_array = np.asarray(
            state.get("episode_end_indices"),
            dtype=np.int64,
        )

        buffers = TrajectoryBuffers()
        buffers.observations = observations
        buffers.actions = actions
        buffers.episode_start_indices = start_indices_array.tolist()
        buffers.episode_end_indices = end_indices_array.tolist()
        buffers.num_steps = int(state.get("num_steps", 0))  # type: ignore[arg-type]
        buffers.num_episodes = int(
            state.get("num_episodes", 0),
        )  # type: ignore[arg-type]

        self._buffers = buffers

    @property
    def buffers(self) -> TrajectoryBuffers:
        """
        Return the trajectory buffers associated with this wrapper.

        These buffers contain all data collected during the run: states,
        actions, episode boundaries, and step/episode counters.
        """
        return self._buffers
