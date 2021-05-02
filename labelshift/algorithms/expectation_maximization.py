"""Expectation Maximization algorithm."""
import warnings
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike

import labelshift.probability as prob
import labelshift.recalibrate as recalib


def expectation_maximization(
    predictions: ArrayLike,
    training_prevalences: ArrayLike,
    *,
    initial_prevalences: Optional[ArrayLike] = None,
    max_steps: int = 10000,
    atol: float = 0.01,
) -> np.ndarray:
    """Expectation maximization algorithm, as described in

    M. Saerens et al., Adjusting the outputs of a classifier to
        new a priori probabilities: A simple procedure.
        Neur. Comput.14, 1 (2002), 21--41.

    Args:
        predictions: test set probability predictions. Shape (n_samples, n_classes).
        prevalences: prevalences in the training data set. Shape (n,), (n,1) or (1, n).
            Will be normalized.
        initial_prevalences: starting prevalences for optimization. Will be normalized.
            If not provided, the training prevalences are used.
            Shape (n,), (n,1) or (1, n).

    Returns:
        test set prevalences, shape (n,).
    """
    predictions = np.asarray(predictions, dtype=float)

    training_prevalences: np.ndarray = prob.normalize_prevalences(
        training_prevalences
    )  # Shape (1, n_classes)

    if initial_prevalences is not None:
        test_prevalences = prob.normalize_prevalences(initial_prevalences)
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
            test_prevalences = new_prevalences
            break
        test_prevalences = new_prevalences

    warnings.warn(f"Required accuracy not reached in {max_steps}.")
    return test_prevalences.ravel()
