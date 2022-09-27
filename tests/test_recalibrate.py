"""Tests for recalibration."""
from typing import Tuple
import numpy as np
import numpy.testing as nptest
import pytest

import labelshift.recalibrate as rc


@pytest.mark.parametrize("shape", ((3, 5), (2, 10), (3, 9)))
def test_training_and_test_the_same(shape: Tuple[int, int], set_random) -> None:
    """The training and test distributions do not differ, so no
    recalibration is needed."""
    n_samples, n_classes = shape

    predictions = np.random.rand(n_samples, n_classes)
    # Normalize predictions, so that each row sums up to 1.
    predictions = predictions / np.sum(predictions, axis=1)[:, None]

    prevalences = np.random.rand(n_classes)
    prevalences = prevalences / np.sum(prevalences)

    recalibrated = rc.recalibrate(predictions, training=prevalences, test=prevalences)

    nptest.assert_allclose(predictions, recalibrated)


def test_known_values() -> None:
    """Simple case."""
    train_prev = [0.5, 0.5]
    test_prev = [0.9, 0.1]

    predictions = [
        [0.5, 0.5],
        [0.1, 0.9],
    ]

    calibrated = [
        [0.9, 0.1],
        [0.5, 0.5],
    ]
    calibrated1 = rc.recalibrate(predictions, training=train_prev, test=test_prev)
    nptest.assert_allclose(calibrated, calibrated1)
