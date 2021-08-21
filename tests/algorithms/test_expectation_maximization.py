"""Tests for Expectation Maximization."""
import numpy as np
import numpy.testing as nptest
import pytest

from labelshift import algorithms


def test_too_short() -> None:
    """Check is warning appears when the optimization time is too short."""
    with pytest.warns(RuntimeWarning):
        algorithms.expectation_maximization([[0.1, 0.3]], [0.5, 0.5], max_steps=1)


def test_perfect_classifier_no_shift(construct_predictions) -> None:
    """Check if we can rediscover the true prevalence if no shift
    is present and the classifier is perfect."""

    training_samples = [60, 40, 10]
    predictions = construct_predictions(training_samples)

    prevalences = algorithms.expectation_maximization(
        predictions, training_samples, initial_prevalences=[1.0, 1.0, 1.0]
    )

    # Training set prevalences.
    prevalences1 = np.array(training_samples) / sum(training_samples)
    nptest.assert_allclose(prevalences, prevalences1)
