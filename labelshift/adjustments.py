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


def project_onto_probability_simplex(v: ArrayLike, /) -> np.ndarray:
    """Projects a point onto the probability simplex.

    The code is adapted from Mathieu Blondel's BSD-licensed
    implementation accompanying the paper

    Large-scale Multiclass Support Vector Machine Training
    via Euclidean Projection onto the Simplex,
    Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
    ICPR 2014.
    http://www.mblondel.org/publications/mblondel-icpr2014.pdf

    Args:
        v: point in n-dimensional space, shape (n,)

    Returns:
        projection of v onto (n-1)-dimensional
          probability simplex, shape (n,)
    """
    v = np.asarray(v)
    n = len(v)

    # Sort the values in the descending order
    u = np.sort(v)[::-1]

    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(v - theta, 0)
