"""Used to calculate summary statistic of discrete data."""
from typing import Sequence

import numpy as np


def _validate_entries(n: int, values: Sequence[int]) -> np.ndarray:
    """Validate whether each entry is in the set {0, ..., n-1}.

    Args:
        n: number of different labels
        values: label of each data point.

    Returns:
        array of integers, the same as `values`

    Raises:
        ValueError if any element of `values` is not in the set {0, ..., n-1}.
    """
    if n <= 0:
        raise ValueError(f"n must be positive integer. Was {n}.")

    if isinstance(values, np.ndarray):
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("You need a sequence of integers.")
        if not values.shape == (len(values),):
            raise ValueError("The provided array has wrong shape.")

    values = np.asarray(values, dtype=int)

    if np.min(values) < 0 or np.max(values) > n - 1:
        raise ValueError(f"For n = {n} the entries must be in the set 0, ..., {n-1}.")

    return values


def count_values(n: int, values: Sequence[int]) -> np.ndarray:
    """Counts the occurrences of different labels.

    Args:
        n: number of different labels
        values: label of each data point. Each entry should be in the set {0, ..., n-1}.

    Returns:
        array of integers, shape (n,). At position `i` it has
          the number of occurrences of the label `i`.

    Raises:
        ValueError, if any element of `values` is not in {0, ..., n-1}
    """
    values = _validate_entries(n, values)

    hist = np.zeros(n, dtype=int)
    for i in values:
        hist[i] += 1

    return hist


def count_values_joint(
    n: int, k: int, ns: Sequence[int], ks: Sequence[int]
) -> np.ndarray:
    """Counts the joint occurrences.

    Args:
        n: number of different labels of type "N"
        k: number of different labels of type "K"
        ns: the "N" label for each example
        ks: the "K" label for each example, length the same as `ns`

    Returns:
        array of integers, shape (n, k). At position (i, j)
          there is the number of indices `u` such that (ns[u], ks[u]) = (i, j)

    Raises:
        ValueError, if `ns` and `ks` have length mismatch
        ValueError, if `ns` or `ks` contain elements not
          in the range {0, ..., n-1} and {0, ..., k-1}
    """
    ns = _validate_entries(n, ns)
    ks = _validate_entries(k, ks)

    hist = np.zeros((n, k), dtype=int)

    for i, j in zip(ns, ks):
        hist[i, j] += 1

    return hist
