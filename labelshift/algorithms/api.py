"""The quantification algorithms API.

Use as:

>>> import labelshift.algorithms.api as algo
"""
from labelshift.algorithms.bayesian_discrete import DiscreteCategoricalMAPEstimator
from labelshift.algorithms.bbse import BlackBoxShiftEstimator
from labelshift.algorithms.classify_and_count import ClassifyAndCount
from labelshift.algorithms.ratio_estimator import InvariantRatioEstimator

__all__ = [
    "BlackBoxShiftEstimator",
    "ClassifyAndCount",
    "DiscreteCategoricalMAPEstimator",
    "InvariantRatioEstimator",
]
