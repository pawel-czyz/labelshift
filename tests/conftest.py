"""Fixtures for pytests."""
import random
from typing import Sequence

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random() -> None:
    """Fixture fixing the random seeds."""
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def construct_predictions():
    """Fixture for constructing one-hot predictions. For signature check
    the wrapped function below."""

    def _construct_predictions(prevalences: Sequence[int]) -> np.ndarray:
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

    return _construct_predictions
