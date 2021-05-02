"""Classify and Count algorithm."""
import numpy as np
from numpy.typing import ArrayLike

import labelshift.algorithms.base as base


class ClassifyAndCount(base.AbstractQuantificationAlgorithm):
    """Simplest version of the Classify and Count algorithm.

    TODO(pawel-czyz): Bibliography.

    Note:
        This is the simplest baseline possible which is usually wrong.

    Attributes:
        predict, returns the vector of prevalences
    """

    def predict(self, /, predictions: ArrayLike) -> np.ndarray:
        """Returns inferred prevalences.

        Args:
            predictions: classifier outputs. Shape (n_samples, n_classes).

        Returns:
            prevalences, shape (n_classes,).
                All entries are non-negative and they sum up to 1.
        """
        predictions = np.asarray(predictions)
        n_samples, n_classes = predictions.shape

        predicted_class = np.argmax(
            predictions, 1
        )  # Shape (n_samples,). Entries 0, ..., n_classes-1.
        histogram = np.array(
            [(predicted_class == i).sum() for i in range(self.n_classes)], dtype=float
        )

        return histogram / np.sum(histogram)
