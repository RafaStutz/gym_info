from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


ObsArray = NDArray[np.float32]
IndexArray = NDArray[np.int32]

# constants
_VECTOR_NDIM = 1  # expected ndim for 1D arrays (lower/upper)
_MATRIX_NDIM = 2  # expected ndim for batched observations (T, d)
_SINGLE_SAMPLE_BATCH = 1  # batch size when reshaping a single sample (d,) -> (1, d)
_ZERO_SPAN_REPLACEMENT = 1.0  # span used for degenerate dimensions (upper == lower)


@dataclass(frozen=True)
class BoxBinningConfig:
    """
    Uniform binning configuration for a Box space.

    Each dimension i is mapped from the continuous interval [lower[i], upper[i]]
    into n_bins[i] integer bins in {0, ..., n_bins[i] - 1}.
    """

    lower: ObsArray
    upper: ObsArray
    n_bins: tuple[int, ...]
    clip: bool = True

    def __post_init__(self) -> None:
        if self.lower.shape != self.upper.shape:
            msg = f"lower and upper must have the same shape, got {self.lower.shape} and {self.upper.shape}"
            raise ValueError(msg)
        if self.lower.ndim != 1:
            msg = f"lower and upper must be 1D arrays, got ndim={self.lower.ndim}"
            raise ValueError(msg)
        if len(self.n_bins) != self.lower.shape[0]:
            msg = f"n_bins length {len(self.n_bins)} does not match dimensionality {self.lower.shape[0]}"
            raise ValueError(msg)
        if any(b <= 0 for b in self.n_bins):
            msg = f"All bin counts must be positive, got {self.n_bins}"
            raise ValueError(msg)

    @property
    def ndim(self) -> int:
        return self.lower.shape[0]


@dataclass(frozen=True)
class EnvDiscretization:
    """
    Discretization configuration for both the observation and action spaces.

    If a space is already discrete, the corresponding config can be set to None.
    """

    env_id: str | None
    obs_config: BoxBinningConfig | None
    action_config: BoxBinningConfig | None


def _normalize_bins_argument(n_bins: int | Iterable[int], ndim: int) -> tuple[int, ...]:
    if isinstance(n_bins, int):
        return tuple(int(n_bins) for _ in range(ndim))
    bins_list = tuple(int(b) for b in n_bins)
    if len(bins_list) != ndim:
        msg = f"Expected {ndim} bin counts, got {len(bins_list)}"
        raise ValueError(msg)
    return bins_list


def make_box_binning_from_space(
    space: spaces.Box,
    n_bins: int | Iterable[int],
    *,
    clip: bool = True,
) -> BoxBinningConfig:
    """
    Create a BoxBinningConfig from a Gymnasium Box space and a bin specification.

    Parameters
    ----------
    space:
        The Box space to discretize. Its `low`, `high` and `shape` fields are used.
    n_bins:
        Either a single integer (same number of bins per dimension) or an
        iterable of length equal to the dimensionality of the space.
    clip:
        If True, observations outside [low, high] are clipped before binning.

    Returns
    -------
    BoxBinningConfig
        A configuration object that can be used with `discretize_box`.
    """
    if space.dtype is not None and np.issubdtype(space.dtype, np.integer):
        msg = "make_box_binning_from_space is intended for continuous Box spaces"
        raise TypeError(msg)

    flat_low = np.asarray(space.low, dtype=np.float32).reshape(-1)
    flat_high = np.asarray(space.high, dtype=np.float32).reshape(-1)

    if flat_low.shape != flat_high.shape:
        msg = f"Box low and high shapes do not match: {flat_low.shape} vs {flat_high.shape}"
        raise ValueError(msg)

    ndim = flat_low.shape[0]
    bins_tuple = _normalize_bins_argument(n_bins, ndim)

    return BoxBinningConfig(
        lower=flat_low, upper=flat_high, n_bins=bins_tuple, clip=clip
    )


def discretize_box(
    observations: ObsArray,
    config: BoxBinningConfig,
) -> IndexArray:
    """
    Discretize continuous observations according to a BoxBinningConfig.

    For dimensions where `config.lower` or `config.upper` are not finite
    (e.g. ±inf), effective bounds are derived from the observed data:
    the minimum and maximum values along that dimension are used.

    Parameters
    ----------
    observations:
        Array of observations of shape (T, d) or (d,), where d equals
        `config.ndim`.
    config:
        Binning configuration describing the target discrete grid.

    Returns
    -------
    IndexArray
        Integer bin indices of shape (T, d). Element [t, i] is in
        {0, ..., config.n_bins[i] - 1} and represents the bin index for
        dimension i at time t.
    """
    obs_array = np.asarray(observations, dtype=np.float32)
    if obs_array.ndim == _VECTOR_NDIM:
        obs_array = obs_array.reshape(_SINGLE_SAMPLE_BATCH, -1)
    if obs_array.ndim != _MATRIX_NDIM:
        msg = (
            f"observations must have shape (T, d) or (d,), got shape {obs_array.shape}"
        )
        raise ValueError(msg)
    if obs_array.shape[1] != config.ndim:
        msg = f"Dimensionality mismatch: observations have d={obs_array.shape[1]}, config.ndim={config.ndim}"
        raise ValueError(msg)

    # Start from the configured bounds, but work on local copies so we do not
    # mutate the configuration object.
    lower_eff = np.array(config.lower, dtype=np.float32, copy=True)
    upper_eff = np.array(config.upper, dtype=np.float32, copy=True)

    # For dimensions with non-finite bounds (inf, NaN), derive bounds from data.
    nonfinite_mask = ~np.isfinite(lower_eff) | ~np.isfinite(upper_eff)
    if np.any(nonfinite_mask):
        data_slice = obs_array[:, nonfinite_mask]
        data_min = data_slice.min(axis=0)
        data_max = data_slice.max(axis=0)
        lower_eff[nonfinite_mask] = data_min
        upper_eff[nonfinite_mask] = data_max

    if config.clip:
        obs_array = np.clip(obs_array, lower_eff, upper_eff)

    span = upper_eff - lower_eff
    span[span == 0.0] = _ZERO_SPAN_REPLACEMENT

    normalized = (obs_array - lower_eff) / span
    n_bins = np.asarray(config.n_bins, dtype=np.int32)
    indices = np.floor(normalized * n_bins).astype(np.int32)
    indices = np.clip(indices, 0, n_bins - 1)

    return indices
