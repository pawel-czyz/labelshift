"""Classify and Count algorithm."""
import numpy as np
from numpy.typing import ArrayLike


def classify_and_count(predictions: ArrayLike, /) -> np.ndarray:
    """Quantification via counting classifier results.

    Args:
        predictions: classifier outputs on the test set.
            Shape (n_samples, n_classes).

    Returns:
        test set prevalences, shape (n_classes,).

    Note:
        This is a very simple approach and may not work well in practice.
    """
    predictions = np.asarray(predictions)
    n_samples, n_classes = predictions.shape

    predicted_class = np.argmax(
        predictions, axis=1
    )  # Shape (n_samples,). Entries 0, ..., n_classes-1.
    histogram = np.array(
        [(predicted_class == i).sum() for i in range(n_classes)], dtype=float
    )  # Shape (n_classes,).

    return histogram / np.sum(histogram)
