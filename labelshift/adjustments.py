"""Predictions adjustments."""
import numpy as np
from numpy.typing import ArrayLike


def label_hardening(predictions: ArrayLike, /) -> np.ndarray:
    """Converts soft-label predictions into one-hot vectors.

    Args:
        predictions: soft labels, shape (n_samples, n_classes)

    Returns:
        hardened predictions, shape (n_samples, n_classes)
    """
    predictions = np.asarray(predictions)
    _, n_classes = predictions.shape

    max_label = np.argmax(predictions, axis=1)  # Shape (n_samples,)
    # Use the identity matrix for one-hot vectors
    eye = np.eye(n_classes, dtype=float)
    return np.asarray([eye[label] for label in max_label])
