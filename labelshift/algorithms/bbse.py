"""Quantification with black-box shift estimators.

Based on:
  Zachary C. Lipton, Yu-Xiang Wang, Alexander J. Smola
  Detecting and Correcting for Label Shift with Black Box Predictors
  https://arxiv.org/pdf/1802.03916.pdf
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

import labelshift.interfaces.point_estimators as pe

# Constant for numerical solver of a linear system.
# See https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
_RCOND: float = 1e-4


def solve_system(
    matrix: np.ndarray,
    vector: np.ndarray,
    square_solver: bool,
    rcond: float = _RCOND,
) -> np.ndarray:
    """Finds `x` such that

    vector[i] = sum(matrix[i, j] * x[j])

    Args:
        matrix: shape (I, J)
        vector: shape (I,)
        square_solver: if True, we'll use `np.linalg.solve` instead
          of (less restrictive) `np.linalg.lstsq`.
          In this case one needs I = J.
        rcond: used to numerically solve a linear system. See:
          https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

    Returns:
        array of shape (J,)
    """
    if square_solver:
        return np.linalg.solve(a=matrix, b=vector)
    else:
        return np.linalg.lstsq(a=matrix, b=vector, rcond=rcond)[0]


def bbse_from_sufficient_statistic(
    n_y_and_c_labeled: ArrayLike,
    n_c_unlabeled: ArrayLike,
    p_y_labeled: Optional[ArrayLike] = None,
    enforce_square: bool = False,
    rcond: float = _RCOND,
) -> np.ndarray:
    """

    Args:
        n_y_and_c_labeled: the (l, k)th entry contains the number of examples
          in the labeled data set such that Y = l and C= f(X) = k. Shape (L, K)
        n_c_unlabeled: the `k`th entry contains the number of examples in the
          unlabeled data set such that Y = k. Shape (K,)
        p_y_labeled: the Y prevalence vector in the labeled data set, shape (L,).
          If not provided, it will be estimated from `n_y_and_c_labeled`.
        enforce_square: whether K = L is enforced. In this case, we will use a different
          solver, which can raise errors
        rcond: used to numerically solve a linear system. See:
          https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

    Note:
        In Lipton et al. (2018) paper there is a K = L requirement.
        You can use K >= L here, but we still require the rank to be at least L.
    """
    n_y_and_c_labeled = np.asarray(n_y_and_c_labeled, dtype=float)
    n_c_unlabeled = np.asarray(n_c_unlabeled, dtype=float)

    if enforce_square and n_y_and_c_labeled.shape[0] != n_y_and_c_labeled.shape[1]:
        raise ValueError(
            f"With enforce_square=1, you need a square shape. "
            f"It was {n_y_and_c_labeled.shape}."
        )

    if n_c_unlabeled.shape[0] != n_y_and_c_labeled.shape[1]:
        raise ValueError(f"Shape mismatch: {n_c_unlabeled} and {n_y_and_c_labeled}.")

    # Get the P_labeled(Y), shape (L,)
    if p_y_labeled is None:
        p_y_labeled = n_y_and_c_labeled.sum(axis=1)
        p_y_labeled = p_y_labeled / p_y_labeled.sum()
    else:
        p_y_labeled = np.asarray(p_y_labeled, dtype=float)
        if p_y_labeled.shape[0] != n_y_and_c_labeled.shape[0]:
            raise ValueError(
                f"Shape mismatch: {p_y_labeled.shape} and {n_y_and_c_labeled.shape}."
            )

    # P(Y and C) matrix, shape (L, K)
    p_y_and_c = n_y_and_c_labeled / n_y_and_c_labeled.sum()

    # P_unlabeled(C), shape (K,)
    p_c_unlabeled = n_c_unlabeled / n_c_unlabeled.sum()

    # Estimate for w = P_unlabeled(Y) / P_labeled(Y)
    w = solve_system(
        matrix=p_y_and_c.T,
        vector=p_c_unlabeled,
        square_solver=enforce_square,
        rcond=rcond,
    )

    return w * p_y_labeled


class BlackBoxShiftEstimator(pe.SummaryStatisticPrevalenceEstimator):
    """Black-Box Shift Estimator which can be applied to different data."""

    def __init__(
        self,
        p_y_labeled: Optional[ArrayLike] = None,
        enforce_square: bool = False,
        rcond: float = _RCOND,
    ) -> None:
        """
        Args:
          p_y_labeled: the Y prevalence vector in the labeled data set, shape (L,).
            If not provided, it will be estimated from `n_y_and_c_labeled`.
          enforce_square: whether K = L is enforced.
            In this case, we will use a different
            solver, which can raise errors
          rcond: used to numerically solve a linear system. See:
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
        """
        self._p_y_labeled = p_y_labeled
        self._enforce_square = enforce_square
        self._rcond = rcond

    def estimate_from_summary_statistic(
        self, /, statistic: pe.SummaryStatistic
    ) -> np.ndarray:
        """For more information see `bbse_from_sufficient_statistic`."""
        return bbse_from_sufficient_statistic(
            n_c_unlabeled=statistic.n_c_unlabeled,
            n_y_and_c_labeled=statistic.n_y_and_c_labeled,
            p_y_labeled=self._p_y_labeled,
            enforce_square=self._enforce_square,
            rcond=self._rcond,
        )
