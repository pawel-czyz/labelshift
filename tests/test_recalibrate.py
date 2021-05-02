import numpy as np
import pytest

import quantipy as qp


@pytest.mark.parametrize("shape", (3, 5), (2, 10), (3, 9))
def test_training_and_test_the_same(shape: tuple[int, int]) -> None:
    n_samples, n_classes = shape

    predictions = np.random.rand(n_samples, n_classes)
    prevalences = np.random.rand(n_classes)

    recalibrated = qp.recalibrate(predictions, training=prevalences, test=prevalences)

    assert (predictions == recalibrated).all()

