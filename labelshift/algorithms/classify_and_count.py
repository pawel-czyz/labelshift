"""Classify and Count algorithm."""
import numpy as np
from numpy.typing import ArrayLike

import labelshift.summary_statistic as summ


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
