"""Common NumPy utilities for dealing with probabilities."""

import numpy as np
from numpy.typing import ArrayLike


def normalize_prevalences(raw_prevalences: ArrayLike, /) -> np.ndarray:
    """Normalizes prevalence vector.

    Args:
        raw_prevalences, will be normalized and reshaped

    Returns:
        prevalences, shape (1, n). Sums up to 1.

    Raises:
        ValueError, if any of the entries is less or equal to 0
    """
    prevalences = np.array(raw_prevalences, dtype=float).reshape((1, -1))

    if np.sum(prevalences) <= 0.0:
        raise ValueError("Probabilities must sum up to a positive value.")

    prevalences = prevalences / np.sum(prevalences)

    return prevalences
