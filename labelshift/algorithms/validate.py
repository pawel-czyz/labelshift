"""Preprocessing and validation methods."""

from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike


def validate(
    *,
    train_predictions: ArrayLike,
    train_labels: ArrayLike,
    test_predictions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """

    Args:
        train_predictions: shape (n_train_samples, n_classes)
        train_labels: shape (n_train_samples,) (or compatible)
        test_predictions: shape (n_test_samples, n_classes)

    Returns:
        train_predictions, shape (n_train_samples, n_classes)
        train_labels, shape (n_train_samples,), dtype int.
        test_predictions, shape (n_test_samples, n_classes)
        n_classes

    Raises:
        ValueError, if n_classes is less than 2
            or train and test sets have incompatible shapes
            or train_labels doesn't have entries in range {0, ..., n_classes-1}
    """
    # Convert to numpy arrays
    train_predictions = np.asarray(train_predictions, dtype=float)
    test_predictions = np.asarray(test_predictions, dtype=float)
    train_labels = np.array(
        train_labels, dtype=int
    ).ravel()  # Shape (n_train_samples,).

    # Check if shapes or predictions are compatible
    n_train, n_test = train_predictions.shape[1], test_predictions.shape[1]

    if n_test != n_train:
        raise ValueError(
            f"The number of classes in training and test set must be the same "
            f"({n_test} != {n_train})."
        )

    # Check if there are at least two classes
    if n_train < 2:
        raise ValueError(f"Number of classes must be at least 2. Was {n_train}.")

    # Check if the labels are in the right range
    in_bounds = (0 <= train_labels) & (train_labels <= n_train - 1)
    if not np.all(in_bounds):
        raise ValueError(
            f"Labels {train_labels} out ouf bounds. "
            f"Must be between 0 and {n_train - 1}."
        )

    return train_predictions, train_labels, test_predictions, n_train
