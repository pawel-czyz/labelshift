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
        prevalences: prevalences in the training data set.
            Shape (n_classes,), (n_classes, 1) or (1, n_classes). Will be normalized.
        initial_prevalences: starting prevalences for optimization.
            If not provided, the training prevalences are used.
            Shape (n_classes,), (n_classes, 1) or (1, n_classes). Will be normalized.
        max_steps: maximal number of iteration steps
        atol: desired accuracy (for early stopping)

    Returns:
        test set prevalences, shape (n_classes,).
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
        old_prevalences = test_prevalences.copy()

        new_predictions: np.ndarray = recalib.recalibrate(
            predictions, training=training_prevalences, test=test_prevalences
        )
        test_prevalences: np.ndarray = np.sum(new_predictions, axis=0).reshape(
            1, -1
        ) / len(new_predictions)

        # Check if converged
        if np.allclose(old_prevalences, test_prevalences, atol=0.01, rtol=0):
            break

    warnings.warn(
        RuntimeWarning(f"Required accuracy not reached in {max_steps} steps.")
    )
    return test_prevalences.ravel()
