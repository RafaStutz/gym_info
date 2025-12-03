from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from ..discretization.trajectory import DiscreteTrajectory, slice_trajectory
from ..discretization.core import IndexArray

# Default log base for entropy calculations
_DEFAULT_LOG_BASE = 2.0


@dataclass(frozen=True)
class HistogramEntropies:
    """
    Entropy metrics computed from a discrete trajectory using histograms.

    All values are measured in bits by default (log base 2).
    """

    H_S: float  # state entropy H(S)
    H_A: float  # action entropy H(A)
    H_A_given_S: float  # conditional action entropy H(A | S)


def _entropy_from_counts(
    counts: NDArray[np.int64],
    *,
    log_base: float = _DEFAULT_LOG_BASE,
) -> float:
    """
    Compute entropy from a vector of counts.

    Zero-count entries are ignored. The result is measured in the given
    logarithm base (base 2 by default => bits).
    """
    total = int(counts.sum())
    if total == 0:
        return 0.0

    probabilities = counts.astype(np.float64) / float(total)
    probabilities = probabilities[probabilities > 0.0]

    if probabilities.size == 0:
        return 0.0

    log_probs = np.log(probabilities)
    if log_base != math.e:
        log_probs = log_probs / math.log(log_base)

    entropy = -float(np.sum(probabilities * log_probs))
    return entropy


def _flatten_states(states: IndexArray) -> NDArray[np.int64]:
    """
    Map multidimensional discrete states to a 1D integer key.

    This is done via a stable hashing approach using a structured dtype,
    then viewing as int64. The absolute values of the keys are not
    meaningful, only equality/grouping properties are used.
    """
    if states.size == 0:
        return np.empty((0,), dtype=np.int64)

    contiguous = np.ascontiguousarray(states, dtype=np.int32)

    num_dims = contiguous.shape[1]
    dtype = np.dtype([("f" + str(i), np.int32) for i in range(num_dims)])
    structured = contiguous.view(dtype=dtype).reshape(-1)

    byte_view = structured.view(np.void)
    _, inverse_indices = np.unique(byte_view, return_inverse=True)
    return inverse_indices.astype(np.int64)


def _action_categories(actions: IndexArray) -> NDArray[np.int64]:
    """
    Map (T, d_a) action indices to a 1D integer key per time step.

    For Discrete action spaces, d_a = 1 and this simply flattens.
    For multidimensional actions, the same structured hashing strategy
    as for states is applied.
    """
    if actions.size == 0:
        return np.empty((0,), dtype=np.int64)

    if actions.ndim != 2:
        msg = f"actions must have shape (T, d_a), got shape {actions.shape}"
        raise ValueError(msg)

    if actions.shape[1] == 1:
        return actions.reshape(-1).astype(np.int64)

    contiguous = np.ascontiguousarray(actions, dtype=np.int32)
    num_dims = contiguous.shape[1]
    dtype = np.dtype([("f" + str(i), np.int32) for i in range(num_dims)])
    structured = contiguous.view(dtype=dtype).reshape(-1)
    byte_view = structured.view(np.void)
    _, inverse_indices = np.unique(byte_view, return_inverse=True)
    return inverse_indices.astype(np.int64)


def state_entropy(
    trajectory: DiscreteTrajectory,
    *,
    log_base: float = _DEFAULT_LOG_BASE,
) -> float:
    """
    Compute the entropy H(S) of the discrete state sequence in bits.

    The joint discrete state S_t is represented by the rows of
    trajectory.states. States that never appear do not contribute
    to the entropy.
    """
    if trajectory.num_steps == 0 or trajectory.states.size == 0:
        return 0.0

    state_keys = _flatten_states(trajectory.states)
    _, counts = np.unique(state_keys, return_counts=True)
    return _entropy_from_counts(counts, log_base=log_base)


def action_entropy(
    trajectory: DiscreteTrajectory,
    *,
    log_base: float = _DEFAULT_LOG_BASE,
) -> float:
    """
    Compute the entropy H(A) of the discrete action sequence in bits.

    The discrete action A_t is represented by the rows of
    trajectory.actions. For Discrete action spaces, this corresponds
    to the scalar action index at each time step.
    """
    if trajectory.num_steps == 0 or trajectory.actions.size == 0:
        return 0.0

    action_keys = _action_categories(trajectory.actions)
    _, counts = np.unique(action_keys, return_counts=True)
    return _entropy_from_counts(counts, log_base=log_base)


def conditional_action_entropy(
    trajectory: DiscreteTrajectory,
    *,
    log_base: float = _DEFAULT_LOG_BASE,
) -> float:
    """
    Compute the conditional entropy H(A | S) from a discrete trajectory in bits.

    This is defined as:

        H(A | S) = sum_s p(s) H(A | S = s)

    where the expectation is taken over the empirical state distribution
    induced by the trajectory.
    """
    if trajectory.num_steps == 0:
        return 0.0
    if trajectory.states.size == 0 or trajectory.actions.size == 0:
        return 0.0

    state_keys = _flatten_states(trajectory.states)
    action_keys = _action_categories(trajectory.actions)

    if state_keys.shape[0] != action_keys.shape[0]:
        msg = "State and action sequences must have the same length."
        raise ValueError(msg)

    joint = np.stack([state_keys, action_keys], axis=1)
    joint_contiguous = np.ascontiguousarray(joint, dtype=np.int64)
    dtype = np.dtype([("s", np.int64), ("a", np.int64)])
    structured_joint = joint_contiguous.view(dtype=dtype).reshape(-1)
    unique_joint, joint_counts = np.unique(structured_joint, return_counts=True)

    state_only = unique_joint["s"]
    unique_states, state_total_counts = np.unique(state_only, return_counts=True)

    total_steps = int(joint_counts.sum())
    if total_steps == 0:
        return 0.0

    state_index_map = {s: i for i, s in enumerate(unique_states)}
    per_state_counts: list[list[int]] = [[] for _ in unique_states]

    for s_value, joint_count in zip(state_only, joint_counts):
        idx = state_index_map[s_value]
        per_state_counts[idx].append(int(joint_count))

    conditional_entropy = 0.0
    for state_idx, counts_list in enumerate(per_state_counts):
        counts_array = np.array(counts_list, dtype=np.int64)
        H_A_given_s = _entropy_from_counts(counts_array, log_base=log_base)
        p_s = float(state_total_counts[state_idx]) / float(total_steps)
        conditional_entropy += p_s * H_A_given_s

    return conditional_entropy


def histogram_entropies(
    trajectory: DiscreteTrajectory,
    *,
    log_base: float = _DEFAULT_LOG_BASE,
) -> HistogramEntropies:
    """
    Convenience function that computes all histogram-based entropies
    for a given discrete trajectory.

    Returns H(S), H(A) and H(A | S), measured in bits by default.
    """
    H_S = state_entropy(trajectory, log_base=log_base)
    H_A = action_entropy(trajectory, log_base=log_base)
    H_A_given_S = conditional_action_entropy(trajectory, log_base=log_base)
    return HistogramEntropies(H_S=H_S, H_A=H_A, H_A_given_S=H_A_given_S)


def episode_histogram_entropies(
    trajectory: DiscreteTrajectory,
    *,
    log_base: float = _DEFAULT_LOG_BASE,
) -> list[HistogramEntropies]:
    """
    Compute histogram-based entropies separately for each completed episode.

    The returned list has length equal to the number of completed
    episodes. An episode is considered completed if it has both a start
    and an end index. Any currently running episode (started but not yet
    terminated) is ignored.
    """
    if trajectory.num_episodes == 0:
        return []

    starts = trajectory.episode_start_indices
    ends = trajectory.episode_end_indices

    n_pairs = min(len(starts), len(ends), trajectory.num_episodes)
    if n_pairs == 0:
        return []

    results: list[HistogramEntropies] = []

    for e in range(n_pairs):
        start = starts[e]
        end = ends[e]
        if start >= end:
            continue

        sub_traj = slice_trajectory(trajectory, start, end)
        ent = histogram_entropies(sub_traj, log_base=log_base)
        results.append(ent)

    return results
