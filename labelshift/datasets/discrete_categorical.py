"""Discrete categorical sampler."""
import dataclasses
import math
from typing import Tuple, Any, Union, Optional

import numpy as np
from numpy.typing import ArrayLike

from labelshift.interfaces.point_estimators import SummaryStatistic

RNG = Any


@dataclasses.dataclass
class SummaryMultinomialStatistic:
    """
    
    Attributes:
        n_y: shape (L,)
        n_c: shape (K,)
        n_y_and_c: shape (L, K)
    """
    n_y: np.ndarray
    n_c: np.ndarray
    n_y_and_c: np.ndarray


class SimpleMultinomialSampler:
    def __init__(self, p_y: ArrayLike, p_c_cond_y: ArrayLike) -> None:
        """
        Args:
            p_y: P(Y) vector, shape (L,)
            p_c_cond_y: P(C|Y) matrix, shape (L, K)
        """
        self._p_y = np.asarray(p_y)
        self._p_c_cond_y = np.asarray(p_c_cond_y)

        assert len(self._p_c_cond_y.shape) == 2
        self._L = self._p_c_cond_y.shape[0]
        self._K = self._p_c_cond_y.shape[1]

        assert self._p_y.shape == (self._L,)

        assert np.min(self._p_y) >= 0
        assert np.min(self._p_c_cond_y) >= 0

        assert math.isclose(np.sum(self._p_y), 1.0)

        for label in range(self._L):
            s = self._p_c_cond_y[label, :].sum()
            assert math.isclose(s, 1.0)

    @property
    def p_y(self) -> np.ndarray:
        return self._p_y

    @property
    def p_c_cond_y(self) -> np.ndarray:
        return self._p_c_cond_y

    @property
    def p_c(self) -> np.ndarray:
        return np.einsum("lk,l->k", self._p_c_cond_y, self._p_y)

    @property
    def n_y(self) -> int:
        return self._L

    @property
    def n_c(self) -> int:
        return self._K

    def sample_summary_statistic(
        self, n: int, rng: Union[int, RNG]
    ) -> SummaryMultinomialStatistic:
        rng = np.random.default_rng(rng)

        n_y = rng.multinomial(n, self._p_y)
        n_y_and_c = np.vstack(
            [rng.multinomial(n, p) for n, p in zip(n_y, self._p_c_cond_y)]
        )
        n_c = n_y_and_c.sum(axis=0)

        assert n_c.shape == (self._K,)

        return SummaryMultinomialStatistic(
            n_c=n_c,
            n_y_and_c=n_y_and_c,
            n_y=n_y,
        )


class DiscreteSampler:
    def __init__(
        self,
        sampler_labeled: SimpleMultinomialSampler,
        sampler_unlabeled: SimpleMultinomialSampler,
    ) -> None:
        assert sampler_labeled.n_c == sampler_unlabeled.n_c
        assert sampler_labeled.n_y == sampler_unlabeled.n_y

        self.labeled = sampler_labeled
        self.unlabeled = sampler_unlabeled

    @property
    def n_y(self) -> int:
        return self.labeled.n_y

    @property
    def n_c(self) -> int:
        return self.labeled.n_c

    def sample_from_both(
        self, n_labeled: int, n_unlabeled: int, seed: int
    ) -> Tuple[SummaryMultinomialStatistic, SummaryMultinomialStatistic]:
        rng1, rng2 = [
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)
        ]

        return (
            self.labeled.sample_summary_statistic(n=n_labeled, rng=rng1),
            self.unlabeled.sample_summary_statistic(n=n_unlabeled, rng=rng2),
        )

    def sample_summary_statistic(
        self, n_labeled: int, n_unlabeled: int, seed: int
    ) -> SummaryStatistic:
        labeled, unlabeled = self.sample_from_both(
            n_labeled=n_labeled, n_unlabeled=n_unlabeled, seed=seed
        )
        return SummaryStatistic(
            n_y_labeled=labeled.n_y,
            n_y_and_c_labeled=labeled.n_y_and_c,
            n_c_unlabeled=unlabeled.n_c,
        )


def discrete_sampler_factory(
    p_y_labeled: ArrayLike,
    p_y_unlabeled: ArrayLike,
    p_c_cond_y_labeled: ArrayLike,
    p_c_cond_y_unlabeled: Optional[ArrayLike] = None,
) -> DiscreteSampler:
    p_c_cond_y_labeled = np.asarray(p_c_cond_y_labeled)

    if p_c_cond_y_unlabeled is None:
        p_c_cond_y_unlabeled = p_c_cond_y_labeled.copy()

    return DiscreteSampler(
        sampler_labeled=SimpleMultinomialSampler(
            p_y=p_y_labeled,
            p_c_cond_y=p_c_cond_y_labeled,
        ),
        sampler_unlabeled=SimpleMultinomialSampler(
            p_y=p_y_unlabeled,
            p_c_cond_y=p_c_cond_y_unlabeled,
        ),
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
