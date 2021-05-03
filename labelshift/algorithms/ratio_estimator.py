import numpy as np
from numpy.typing import ArrayLike


def ratio_estimator(
    *,
    train_predictions: ArrayLike,
    train_labels: ArrayLike,
    test_predictions: ArrayLike,
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
    """
    pass
