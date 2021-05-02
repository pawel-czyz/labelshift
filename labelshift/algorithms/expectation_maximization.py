"""Expectation Maximization algorithm."""
import warnings
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike

import labelshift.algorithms.base as base
import labelshift.recalibrate as recalib


class ExpectationMaximization(base.BaseQuantificationAlgorithm):
    """Expectation maximization algorithm proposed in

    M. Saerens et al., Adjusting the outputs of a classifier to
        new a priori probabilities: A simple procedure.
        Neur. Comput.14, 1 (2002), 21–41.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prior_prevalences: Optional[np.ndarray] = None  # Shape (n_classes,)

    def _fit(self, /, predictions: np.ndarray, labels: np.ndarray) -> None:
        histogram = np.array(
            [np.sum(labels == i) for i in range(self.n_classes)], dtype=float
        )

        self._prior_prevalences = histogram / np.sum(histogram)

    def _predict(self, /, predictions: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def normalize_prevalences(raw_prevalences: ArrayLike, /) -> np.ndarray:
    """

    Args:
        raw_prevalences, will be normalized and reshaped

    Returns:
        prevalences, shape (1, n). Sums up to 1.

    Raises:
        ValueError, if any of the entries is 0
    """
    prevalences = np.array(raw_prevalences, dtype=float).reshape((1, -1))
    prevalences = prevalences / np.sum(prevalences)
    if not np.all(prevalences > 0):
        raise ValueError(
            f"All prevalences {prevalences} must be strictly greater than 0."
        )
    return prevalences


def expectation_maximization(
    predictions: ArrayLike,
    training_prevalences: ArrayLike,
    *,
    initial_prevalences: Optional[ArrayLike] = None,
    max_steps: int = 10000,
    atol: float = 0.01,
) -> np.ndarray:
    """

    Args:
        predictions: shape (n_samples, n_classes)
        prevalences: prevalences in the training data set. Shape (n,), (n,1) or (1, n).
            Will be normalized.
        initial_prevalences: starting prevalences for optimization. If not provided,
            the training prevalences are used. Shape (n,), (n,1) or (1, n).
    """
    predictions = np.asarray(predictions, dtype=float)

    training_prevalences: np.ndarray = normalize_prevalences(
        training_prevalences
    )  # Shape (1, n_classes)

    if initial_prevalences is not None:
        test_prevalences = normalize_prevalences(initial_prevalences)
    else:
        test_prevalences = training_prevalences.copy()

    for _ in range(max_steps):
        new_predictions: np.ndarray = recalib.recalibrate(
            predictions, training=training_prevalences, test=test_prevalences
        )
        new_prevalences: np.ndarray = np.sum(new_predictions, axis=0).reshape(
            1, -1
        ) / len(new_predictions)

        if np.allclose(new_prevalences, test_prevalences, atol=0.01, rtol=0):
            return new_prevalences.ravel()
        else:
            test_prevalences = new_prevalences

    warnings.warn(f"Required accuracy not reached in {max_steps}.")
    return test_prevalences
