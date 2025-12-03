from __future__ import annotations

import math

import numpy as np

from gym_info.discretization.trajectory import DiscreteTrajectory
from gym_info.measures.histogram import (
    _entropy_from_counts,
    action_entropy,
    conditional_action_entropy,
    state_entropy,
)


def _make_trajectory(
    state_ids: np.ndarray,
    action_ids: np.ndarray,
) -> DiscreteTrajectory:
    """
    Helper to build a minimal DiscreteTrajectory from 1D state/action id arrays.

    """
    if state_ids.shape != action_ids.shape:
        msg = f"state_ids and action_ids must have same shape, got {state_ids.shape} and {action_ids.shape}"
        raise ValueError(msg)

    T = int(state_ids.shape[0])

    states = state_ids.reshape(T, 1).astype(np.int32)
    actions = action_ids.reshape(T, 1).astype(np.int32)

    # A single episode [0, T)
    episode_start_indices = (0,)
    episode_end_indices = (T,)

    return DiscreteTrajectory(
        states=states,
        actions=actions,
        episode_start_indices=episode_start_indices,
        episode_end_indices=episode_end_indices,
        num_steps=T,
        num_episodes=1,
    )


def test_entropy_uniform_distribution() -> None:
    """
    Discrete uniform distribution: H ~ log2(k).
    """
    k = 8
    counts = np.ones(k, dtype=np.int64)  # each symbol appears once
    H = _entropy_from_counts(counts, log_base=2.0)

    expected = math.log2(k)
    assert math.isclose(H, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_entropy_degenerate_distribution() -> None:
    """
    Totally concentrated distribution: H ~ 0.
    """
    counts = np.array([10, 0, 0, 0], dtype=np.int64)
    H = _entropy_from_counts(counts, log_base=2.0)

    assert H >= 0.0
    assert math.isclose(H, 0.0, abs_tol=1e-8)


def test_conditional_entropy_deterministic_action_given_state() -> None:
    """
    Case where the action is a deterministic function of the state: H(A|S) ~ 0.

    Example: states [0,0,1,1], actions = states.
    """
    state_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    action_ids = np.array([0, 0, 1, 1], dtype=np.int64)

    traj = _make_trajectory(state_ids, action_ids)

    H_S = state_entropy(traj)
    H_A = action_entropy(traj)
    H_A_given_S = conditional_action_entropy(traj)

    # H(S) and H(A) should be ~1 bit (two equiprobable categories)
    assert math.isclose(H_S, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(H_A, 1.0, rel_tol=1e-6, abs_tol=1e-6)

    # Deterministic action given state: H(A|S) ~ 0
    assert H_A_given_S >= 0.0
    assert math.isclose(H_A_given_S, 0.0, abs_tol=1e-8)


def test_conditional_entropy_independent_actions_and_states() -> None:
    """
    Case where the action is independent of the state: H(A|S) ~ H(A).

    Example: 2 states, 2 actions, joint uniform:
        (s,a) ∈ {(0,0),(0,1),(1,0),(1,1)}
    """
    state_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    action_ids = np.array([0, 1, 0, 1], dtype=np.int64)

    traj = _make_trajectory(state_ids, action_ids)

    H_A = action_entropy(traj)
    H_A_given_S = conditional_action_entropy(traj)

    # H(A) should be ~1 bit (two equiprobable categories)
    assert math.isclose(H_A, 1.0, rel_tol=1e-6, abs_tol=1e-6)

    # Independent actions and states: H(A|S) ~ H(A)
    assert math.isclose(H_A_given_S, H_A, rel_tol=1e-6, abs_tol=1e-6)
