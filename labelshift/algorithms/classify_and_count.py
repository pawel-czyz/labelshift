"""Classify and Count algorithm."""

import numpy as np
from numpy.typing import ArrayLike

import labelshift.summary_statistic as summ
import labelshift.interfaces.point_estimators as pe


def classify_and_count_from_predictions(predictions: ArrayLike, /) -> np.ndarray:
    """Quantification via counting classifier results.

    Args:
        predictions: classifier outputs on the test set.
            Shape (n_samples, n_classes).

    Returns:
        test set prevalence, shape (n_classes,).

    Note:
        This is a very simple approach and may not work well in practice.
    """
    predictions = np.asarray(predictions)
    n_classes = predictions.shape[1]

    predicted_class = np.argmax(
        predictions, axis=1
    )  # Shape (n_samples,). Entries 0, ..., n_classes-1.

    return classify_and_count_from_labels(n_classes, predicted_class)


def classify_and_count_from_labels(n_classes: int, labels: ArrayLike) -> np.ndarray:
    """Quantification via counting classifier results.

    Args:
        n_classes: number of different label classes
        labels: for each example its label.
          The labels should be from the set {0, ..., n_classes-1}

    Returns:
        label prevalence vector, shape (n_classes,)
    """
    unnormalized = np.asarray(summ.count_values(n_classes, labels), dtype=float)

    return unnormalized / unnormalized.sum()


def classify_and_count_from_sufficient_statistic(
    n_c_unlabeled: ArrayLike,
) -> np.ndarray:
    """Applies the algorithm to sufficient statistic.

    Args:
        n_c_unlabeled: shape (K,), which should be the same as (L,)
          (see below for the explanation of this assumption)

    Note:
        This approximates P_test(Y) via P_test(C).
        Of course, it may be a very bad approximation
        if the classifier is very bad.

        Note that it assumes that the number of possible
        labels Y is the same as the number of available Cs.
    """
    n_c = np.asarray(n_c_unlabeled, dtype=float)
    return n_c / n_c.sum()


class ClassifyAndCount(pe.SummaryStatisticPrevalenceEstimator):
    """The class implementing several standard interfaces, which
    is based on the simple "classify and count" algorithm.

    Note:
        This algorithm essentially assumes that C and Y are the same.
    """

    def estimate_from_summary_statistic(
        self, /, statistic: pe.SummaryStatistic
    ) -> np.ndarray:
        """Implementation of the standard method.

        For more information see `classify_and_count_from_summary_statistic`.
        """
        return classify_and_count_from_sufficient_statistic(
            n_c_unlabeled=statistic.n_c_unlabeled
        )
