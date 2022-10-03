"""Protocols for point estimators for P_test(Y),
which may have access to different data modalities."""
import dataclasses
from typing import Protocol

import numpy as np


@dataclasses.dataclass
class SummaryStatistic:
    """Summary statistics of the generated data set.

    Attrs:
        n_y_labeled: array of shape (L,) with occurrences of Y
          in the labeled data set
        n_y_and_c_labeled: array of shape (L, K) with occurrences
          of pairs (Y, C) in labeled data set
        n_c_unlabeled: array of shape (K,) with histogram
          of occurrences of C in unlabeled data set
    """

    n_y_labeled: np.ndarray
    n_y_and_c_labeled: np.ndarray
    n_c_unlabeled: np.ndarray


class SummaryStatisticPrevalenceEstimator(Protocol):
    """Protocol for methods allowing a point estimate
    from summary statistics of the data."""

    def estimate_from_summary_statistic(
        self, /, statistic: SummaryStatistic
    ) -> np.ndarray:
        """
        Args:
            statistic: summary statistic

        Returns:
            prevalence vector P_test(Y), shape (L,)
        """
        raise NotImplementedError
