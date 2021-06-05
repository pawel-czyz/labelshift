import numpy as np
from numpy import linalg
from numpy.typing import ArrayLike

import labelshift.algorithms.validate as validate


def ratio_estimator(
    *,
    train_predictions: ArrayLike,
    train_labels: ArrayLike,
    test_predictions: ArrayLike,
    restrict: bool = False,
) -> np.ndarray:
    """General ratio estimator, as described in Remark 5
    of
    A. Fernandes Vaz et al., Quantification Under Prior Probability
        Shift: the Ratio Estimator and its Extensions.
        Journal of Machine Learning Research 20 (2019), 1--33

    This generalizes the Adjusted Classify and Count estimator
    (predictions are in the set :math:`{0, 1}`) described in

    G. Forman. Quantifying counts and costs via classification.
    Data Mining and Knowledge Discovery 17(2) (2008), 164--206

    and the quantifier proposed in

    A. Bella et al., Quantification via probability estimators.
    IEEE 10th International Conference on Data Mining (ICDM) (2010), 737--742

    where the predictions are "soft" in the interval :math:`[0, 1]`.

    Args:
        train_predictions: predictions of the classifier on the training set.
            Shape (n_samples, n_classes)
        train_labels: labels of the training set examples.
            Shape (n_samples,), (n_samples, 1) or (1, n_samples).
            Entries should be from the set {0, 1, ..., n_classes-1}.
        test_predictions: predictions of the classifier on the test test.

    Returns:
        prevalences on the test set, shape (n_classes,).

    Note:
        This is the unrestricted version.
    """
    # Convert to numpy arrays, check whether shapes align, and get the number of classes
    train_predictions, train_labels, test_predictions, n = validate.validate(
        train_predictions=train_predictions,
        train_labels=train_predictions,
        test_predictions=test_predictions,
    )

    # Shape (n-1,)
    g_hat = _g_hat_vector(test_predictions)
    # Shape (n-1, n)
    G_hat = _G_hat_matrix(
        train_predictions=train_predictions, train_labels=train_labels, n=n
    )

    # The second equation (total probability equals 1) can be merged with this one
    # via adding an additional component to g_hat and an additional row to G_hat
    vector = np.ones(n)
    vector[: n - 1] = g_hat

    matrix = np.ones(n, n)
    matrix[: n - 1, :] = G_hat

    # Unresticted thetas
    thetas = linalg.solve(matrix, vector)

    if restrict:
        raise NotImplementedError("Restricted thetas not optimized yet.")

    return thetas


def _g_hat_vector(test_predictions: ArrayLike) -> np.ndarray:
    """Calculates the :math:`\\hat g` vector from Vaz et al.

    Args:
        test_predictions: shape (n_samples, n_classes)

    Returns:
        :math:`\\hat g` vector, shape (n_classes-1,)

    Note:
        It has shape (n_classes-1,), i.e. it doesn't store
        the information about Y = 0 label."""
    return np.average(test_predictions, axis=0)[1:]


def _G_hat_matrix(
    train_predictions: np.ndarray, train_labels: np.ndarray, n: int
) -> np.ndarray:
    """Calculates the :math:`\\hat G` matrix from Vaz et al.

    Args:
        train_predictions: shape (n_samples, n)
        train_labels: shape (n_samples,)
        n: number of classes

    Returns:
        :math:`\\hat G` matrix, shape (n-1, n)

    Note:
        `n` must be at least 2
    """
    matrix = np.zeros((n - 1, n), dtype=float)

    for label in range(n):
        index = train_labels == label
        predictions_for_label = train_predictions[index, :]

        matrix[:, label] = np.average(predictions_for_label, axis=0)[1:]

    return matrix
