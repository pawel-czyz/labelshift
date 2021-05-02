"""Common NumPy utilities for dealing with probabilities."""
import numpy as np
from numpy.typing import ArrayLike


def normalize_prevalences(raw_prevalences: ArrayLike, /) -> np.ndarray:
    """

    Args:
        raw_prevalences, will be normalized and reshaped

    Returns:
        prevalences, shape (1, n). Sums up to 1.

    Raises:
        ValueError, if any of the entries is 0
    """
    prevalences = np.array(raw_prevalences, dtype=float).reshape((1, -1))
    prevalences = prevalences / np.sum(prevalences)
    if not np.all(prevalences > 0):
        raise ValueError(
            f"All prevalences {prevalences} must be strictly greater than 0."
        )
    return prevalences
