import numpy as np

import labelshift.algorithms.base as base


class ClassifyAndCount(base.AbstractQuantificationAlgorithm):
    """Simplest version of the Classify and Count algorithm.
    
    TODO(pczyz): Bibliography.
    
    Attributes:
        n_classes (int), number of classes
        predict, returns the vector of prevalences
    """

    def __init__(self, n_classes: int) -> None:
        """
        Args:
            n_classes: number of classes
        """
        self.n_classes: int = n_classes

    def _predict(self, /, predictions: np.ndarray) -> np.ndarray:
        predicted_class = np.argmax(
            predictions, 1)  # Shape (n_samples,). Entries 0, ..., n_classes-1.
        histogram = [(predicted_class == i).sum() for i in range(n_classes)]
        return np.array(histogram, dtype=int)


class AdjustedClassifyAndCount(base.BaseQuantificationAlgorithm):
    pass
