"""Tests for recalibration."""
from typing import Tuple
import numpy as np
import numpy.testing as nptest
import pytest

import labelshift as ls


@pytest.mark.parametrize("shape", ((3, 5), (2, 10), (3, 9)))
def test_training_and_test_the_same(shape: Tuple[int, int]) -> None:
    """The training and test distributions do not differ, so no
    recalibration is needed."""
    n_samples, n_classes = shape

    predictions = np.random.rand(n_samples, n_classes)
    # Normalize predictions, so that each row sums up to 1.
    predictions = predictions / np.sum(predictions, axis=1)[:, None]

    prevalences = np.random.rand(n_classes)
    prevalences = prevalences / np.sum(prevalences)

    recalibrated = ls.recalibrate(predictions,
                                  training=prevalences,
                                  test=prevalences)

    nptest.assert_allclose(predictions, recalibrated)
