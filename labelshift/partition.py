"""Partition of the real line into intervals."""
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

_INFINITY = np.inf


class RealLinePartition:
    """Partitions the real line into K (disjoint) intervals."""

    def __init__(self, breakpoints: Sequence[float]) -> None:
        """

        Args:
            breakpoints: points a_1, ..., a_{K-1} determine
              the partition into K intervals
              (-oo, a_1), [a_1, a_2), [a_2, a_3), ..., [a_{K-1}, +oo)
        """
        breakpoints = list(breakpoints)
        # Numer of intervals we want
        K = len(breakpoints) + 1

        # Make sure that the points are in ascending order
        # and that no two values are equal.
        # Note that we want k = 0, ...
        # and k + 1 = 1, ..., K-2,
        # as we have K-1 breakpoints
        # Hence, we want to have k = 0, ..., K-3
        for k in range(K - 2):
            assert breakpoints[k] < breakpoints[k + 1]

        self._intervals: List[Tuple[float, float]] = (
            [(-_INFINITY, breakpoints[0])]
            + [(breakpoints[k], breakpoints[k + 1]) for k in range(K - 2)]
            + [(breakpoints[-1], _INFINITY)]
        )

        assert (
            len(self._intervals) == K
        ), f"Expected {K} intervals, but got the following: {self._intervals}."

    def __len__(self) -> int:
        """Number of the intervals, K."""
        return len(self._intervals)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the labels of points in `X`.

        Args:
            X: points on the real line, shape (n, 1) or (n,)

        Returns:
            array of integers valued in {0, 1, ..., K-1} with the
              interval number, for each point in `X`
        """
        X = np.asarray(X).ravel()  # Shape (n,)
        return np.asarray([self._predict_point(x) for x in X], dtype=int)

    def _predict_point(self, x: float) -> int:
        """Predict the label (interval) for point `x`."""
        for k, (a, b) in enumerate(self._intervals):
            if a <= x < b:
                return k
        raise ValueError(f"No interval containing {x}.")

    def interval(self, k: int) -> Tuple[float, float]:
        """Returns the ends of the `k`th interval.

        Args:
            k: should be valued between 0 and K-1.

        Raises:
            IndexError, if k >= K (is out of range)
        """
        return self._intervals[k]


def gaussian_probability_masses(
    means: ArrayLike, sigmas: ArrayLike, partition: RealLinePartition
) -> np.ndarray:
    """
    Args:
        means: shape (L,)
        sigmas: shape (L,)
        partition: partition into K intervals

    Returns:
        matrix P(C|Y), shape (L, K). The (l,k)th entry is the probability mass
        of the `l`th Gaussian contained in the `k`th interval
    """
    means = np.asarray(means)
    sigmas = np.asarray(sigmas)
    L = len(means)
    assert means.shape == sigmas.shape == (L,)

    K = len(partition)
    p_c_cond_y = np.zeros((L, K))

    for l in range(L):  # noqa: E741
        for k in range(K):
            mu, sigma = means[l], sigmas[l]

            a, b = partition.interval(k)
            p_c_cond_y[l, k] = stats.norm.cdf(b, loc=mu, scale=sigma) - stats.norm.cdf(
                a, loc=mu, scale=sigma
            )

    return p_c_cond_y
