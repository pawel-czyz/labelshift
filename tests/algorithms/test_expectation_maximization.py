from typing import Sequence
import numpy as np
import numpy.testing as nptest
import pytest

import labelshift as ls


def test_too_short() -> None:
    """Check is warning appears when the optimization time is too short."""
    with pytest.warns(RuntimeWarning):
        ls.expectation_maximization([[0.1, 0.3]], [0.5, 0.5], max_steps=1)


def get_predictions(prevalences: Sequence[int]) -> np.ndarray:
    """Constructs one-hot predictions for given prevalences.

    Args:
        prevalences: the number of examples to generate for each class

    Returns
        one-hot classifier predictions. Shape (n_samples, n_classes),
            where `n_samples` is the sum of `prevalences` entries
            and `n_classes` is its length
    """
    n_classes = len(prevalences)

    # Identity matrix. Get perfect prediction for class i as classifier[i].
    classifier = np.eye(n_classes, dtype=float)

    predictions = sum(
        [[classifier[i].tolist()] * k for i, k in enumerate(prevalences)], []
    )

    return np.asarray(predictions)


def test_perfect_classifier_no_shift() -> None:
    """Check if we can rediscover the true prevalence if no shift
    is present and the classifier is perfect."""

    training_samples = [60, 40, 10]

    predictions = get_predictions(training_samples)

    prevalences = ls.expectation_maximization(
        predictions, training_samples, initial_prevalences=[1.0, 1.0, 1.0]
    )

    # Training set prevalences.
    prevalences1 = np.array(training_samples) / sum(training_samples)
    nptest.assert_allclose(prevalences, prevalences1)
