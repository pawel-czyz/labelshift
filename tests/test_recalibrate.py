from typing import Tuple
import numpy as np
import pytest

import labelshift as ls


@pytest.mark.parametrize("shape", ((3, 5), (2, 10), (3, 9)))
def test_training_and_test_the_same(shape: Tuple[int, int]) -> None:
    """The training and test distributions do not differ, so no
    recalibration is needed."""
    n_samples, n_classes = shape

    predictions = np.random.rand(n_samples, n_classes)
    prevalences = np.random.rand(n_classes)

    recalibrated = ls.recalibrate(predictions,
                                  training=prevalences,
                                  test=prevalences)

    assert (predictions == recalibrated).all()
