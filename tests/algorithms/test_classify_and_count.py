"""Classify and count algorithm."""

import numpy as np
import numpy.testing as nptest

import labelshift.algorithms.classify_and_count as cc


# TODO(pawel-czyz): This could be refactored into a fixture.
predictions = [
    [0.4, 0.6],
    [0.7, 0.3],
    [0.1, 0.9],
]


def test_list() -> None:
    """Test if works for a list."""
    prevalences = cc.classify_and_count_from_predictions(predictions)
    nptest.assert_allclose(prevalences, [1 / 3, 2 / 3])


def test_array() -> None:
    """Test if works for a numpy array."""
    prevalences = cc.classify_and_count_from_predictions(np.array(predictions))
    nptest.assert_allclose(prevalences, [1 / 3, 2 / 3])
