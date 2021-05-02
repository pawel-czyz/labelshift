import abc
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike


class AbstractQuantificationAlgorithm(abc.ABC):
    """An abstract quantification algorithm.

    Attributes:
        predict, returns the vector of prevalences  
    
    """

    @abc.abstractmethod
    def _predict(self, /, predictions: ArrayLike) -> np.ndarray:
        pass

    def predict(self, /, predictions: ArrayLike) -> np.ndarray:
        predictions = np.array(predictions)
        return np.array(self._predict(predictions), dtype=float)


class BaseQuantificationAlgorithm(AbstractQuantificationAlgorithm):

    def __init__(self) -> None:
        self._n_classes: Optional[int] = None

    @property
    def n_classes(self) -> Optional[int]:
        return self._n_classes

    def fit(self, predictions: ArrayLike, labels: ArrayLike) -> None:
        """

        Args:
            predictions: one-hot encoded predictions of the classifier on the test data set. Shape (n_examples, n_classes).
            labels: true labels of the data. Shape (n_examples,).
        """
        predictions, labels = np.array(predictions), np.array(labels)
        assert len(predictions) == len(
            labels), "Number of examples must be the same."

        self._n_classes: int = predictions.shape[1]
        self._fit(predictions, labels)

    def predict(self, /, predictions: ArrayLike) -> np.ndarray:
        """

        Args:
            predictions: one-hot encoded predictions of the classifier on the data set with unknown label prevalences. Shape (n_examples, n_classes).

        Returns:
            prevalences, shape (n_classes,)
        """
        predictions = np.array(predictions)

        if predictions.shape[1] != self.n_classes:
            raise ValueError(
                f"The shape of predictions should be {(-1, self.n_classes)}. Was {predictions.shape}."
            )

        return self._predict(predictions)

    @abc.abstractmethod
    def _fit(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """

        Args:
            predictions: one-hot encoded predictions of the classifier on the test data set. Shape (n_examples, n_classes).
            labels: true labels of the data. Shape (n_examples,).
        """
        pass

    @abc.abstractmethod
    def _predict(self, /, predictions: np.ndarray) -> np.ndarray:
        """

        Args:
            predictions: one-hot encoded predictions of the classifier on the data set with unknown label prevalences. Shape (n_examples, n_classes).

        Returns:
            prevalences, shape (n_classes,)
        """
        pass
