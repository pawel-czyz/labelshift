import dataclasses

import numpy as np
from numpy.typing import ArrayLike


@dataclasses.dataclass
class SummaryStatistic:
    n_y_labeled: np.ndarray
    n_y_and_c_labeled: np.ndarray
    n_c_unlabeled: np.ndarray


class DiscreteSampler:
    def __init__(
        self, p_y_labeled: ArrayLike, p_y_unlabeled: ArrayLike, p_y_c: ArrayLike
    ) -> None:
        self._p_y_labeled = np.asarray(p_y_labeled)
        self._p_y_unlabeled = np.asarray(p_y_unlabeled)
        self._p_y_c = np.asarray(p_y_c)

        assert len(self._p_y_c) == 2
        self._L = self._p_y_c.shape[0]
        self._K = self._p_y_c.shape[1]

        assert self._p_y_labeled.shape == (self._L,)
        assert self._p_y_unlabeled.shape == (self._L,)

    @property
    def size_Y(self) -> int:
        return self._L

    def size_C(self) -> int:
        return self._K

    def sample_summary_statistic(
        self, n_labeled: int = 1000, n_unlabeled: int = 1000, seed: int = 42
    ) -> SummaryStatistic:
        rng = np.random.default_rng(seed)

        n_y = rng.multinomial(n_labeled, self._p_y_labeled)
        n_y_and_c = np.vstack([rng.multinomial(n, p) for n, p in zip(n_y, self._p_y_c)])
        n_c = rng.multinomial(n_unlabeled, self._p_y_c.T @ self._p_y_unlabeled)

        return SummaryStatistic(
            n_y_labeled=n_y,
            n_c_unlabeled=n_c,
            n_y_and_c_labeled=n_y_and_c,
        )
