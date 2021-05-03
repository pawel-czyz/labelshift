import numpy as np


def ratio_estimator() -> np.ndarray:
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


    Returns:
        prevalences, shape (n_classes,)

    """
    pass
