"""Discrete categorical sampler."""
import math
import numpy as np
from numpy.typing import ArrayLike

from labelshift.interfaces.point_estimators import SummaryStatistic


class DiscreteSampler:
    """Samples from the discrete model P(C|Y)."""

    def __init__(
        self, p_y_labeled: ArrayLike, p_y_unlabeled: ArrayLike, p_c_cond_y: ArrayLike
    ) -> None:
        """
        Args:
            p_y_labeled: P_train(Y) vector, shape (L,)
            p_y_unlabeled: P_test(Y) vector, shape (L,)
            p_c_cond_y: P(C|Y), shape (L, K)
        """
        self._p_y_labeled = np.asarray(p_y_labeled)
        self._p_y_unlabeled = np.asarray(p_y_unlabeled)
        self._c_cond_y = np.asarray(p_c_cond_y)

        assert len(self._c_cond_y.shape) == 2
        self._L = self._c_cond_y.shape[0]
        self._K = self._c_cond_y.shape[1]

        assert self._p_y_labeled.shape == (self._L,)
        assert self._p_y_unlabeled.shape == (self._L,)

        assert np.min(self._p_y_labeled) >= 0
        assert np.min(self._p_y_unlabeled) >= 0
        assert np.min(self._c_cond_y) >= 0

        assert math.isclose(np.sum(self._p_y_labeled), 1.0)
        assert math.isclose(np.sum(self._p_y_unlabeled), 1.0)

        for label in range(self._L):
            s = self._c_cond_y[label, :].sum()
            assert math.isclose(s, 1.0)

    @property
    def p_y_labeled(self) -> np.ndarray:
        """P_labeled(Y) vector. Shape (size_Y,)"""
        return self._p_y_labeled

    @property
    def p_y_unlabeled(self) -> np.ndarray:
        """P_unlabeled(Y) vector. Shape (size_Y,)"""
        return self._p_y_unlabeled

    @property
    def p_c_cond_y(self) -> np.ndarray:
        """P(C | Y) matrix, shape (size_Y, size_C)"""
        return self._c_cond_y

    @property
    def p_c_unlabeled(self) -> np.ndarray:
        """P_unlabeled(C) vector, shape (size_C,)"""
        return self._c_cond_y.T @ self._p_y_unlabeled

    @property
    def p_c_labeled(self) -> np.ndarray:
        """P_labeled(C) vector, shape (size_C,)"""
        return self._c_cond_y.T @ self._p_y_labeled

    @property
    def size_Y(self) -> int:
        """How many Y there are."""
        return self._L

    def size_C(self) -> int:
        """How many C there are."""
        return self._K

    def sample_summary_statistic(
        self, n_labeled: int = 1000, n_unlabeled: int = 1000, seed: int = 42
    ) -> SummaryStatistic:
        """Samples the summary statistic from the model.

        Args:
            n_labeled: number of examples in the labeled data set
            n_unlabeled: number of examples in the unlabeled data set
            seed: random seed
        """
        rng = np.random.default_rng(seed)

        n_y = rng.multinomial(n_labeled, self._p_y_labeled)
        n_y_and_c = np.vstack(
            [rng.multinomial(n, p) for n, p in zip(n_y, self._c_cond_y)]
        )
        n_c = rng.multinomial(n_unlabeled, self._c_cond_y.T @ self._p_y_unlabeled)

        return SummaryStatistic(
            n_y_labeled=n_y,
            n_c_unlabeled=n_c,
            n_y_and_c_labeled=n_y_and_c,
        )


def almost_eye(y: int, c: int, diagonal: float = 1.0) -> np.ndarray:
    """Matrix P(C | Y) with fixed "diagonal" terms.

    Args:
        y: number of different Y labels
        c: number of different C predictions
        diagonal: the probability mass associated to the diagonal terms.
          The rest of the mass is evenly spread among other entries

    Example:
      For y = 2, c = 3, diagonal = 0.6, we expect the array

         ( 0.6, 0.2, 0.2 )
         ( 0.2, 0.6, 0.2 )

      For y = 3, c = 2, diagonal = 0.6, we expect the array
         ( 0.6, 0.4 )
         ( 0.4, 0.6 )
         ( 0.5, 0.5 )    # Note that there's no diagonal term here

    Returns:
        matrix of shape (L, C). The rows sum up to 1.
    """
    assert c > 1

    offdiagonal = (1 - diagonal) / (c - 1)

    arr = np.full((y, c), offdiagonal)
    for i in range(0, min(y, c)):
        arr[i, i] = diagonal

    for i in range(c, y):
        arr[i, :] = 1 / c

    return arr
