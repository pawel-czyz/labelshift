"""Recalibration utilities under the prior probability shift assumption."""
import numpy as np
from numpy.typing import ArrayLike


def recalibrate(
    predictions: ArrayLike, *, training: ArrayLike, test: ArrayLike
) -> np.ndarray:
    """Recalibrate the probabilities predicted by a classifier
    under the prior probability shift assumption.

    See Equation 2.4 in M. Saerens et al., Adjusting the outputs of a classifier
        to new a priori probabilities: A simple procedure.
        Neur. Comput.14, 1 (2002), 21--41.

    Args:
        predictions: array with classifier predictions, shape (n_samples, n_classes).
        training_prevalences: the ith component is the probability of observing
            class i in the training data set. Shape (n_classes,).
        test_prevalences: the ith component is the probability of observing
            class i in the test data set. Shape (n_classes,).

    Returns:
        recalibrated predictions. Shape (n_samples, n_classes).

    Note:
        If the classifier is biased towards some classes, this bias will be increased.
    """
    predictions = np.array(predictions, dtype=float)
    training_prevalences = np.array(training, dtype=float).reshape((1, -1))
    test_prevalences = np.array(test, dtype=float).reshape((1, -1))

    assert (
        predictions.shape[1]
        == training_prevalences.shape[1]
        == test_prevalences.shape[1]
    ), "Shapes are not compatible."

    recalibrated = predictions * test_prevalences / training_prevalences
    recalibrated = recalibrated / np.sum(recalibrated, axis=1)[:, None]

    return recalibrated
