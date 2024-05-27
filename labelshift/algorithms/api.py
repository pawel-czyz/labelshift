"""The quantification algorithms API.

Use as:

>>> import labelshift.algorithms.api as algo
"""

from labelshift.algorithms.bayesian_discrete import (
    DiscreteCategoricalMeanEstimator,
    SamplingParams,
)
from labelshift.algorithms.bbse import BlackBoxShiftEstimator
from labelshift.algorithms.classify_and_count import ClassifyAndCount
from labelshift.algorithms.ratio_estimator import InvariantRatioEstimator
from labelshift.interfaces.point_estimators import SummaryStatistic

__all__ = [
    "BlackBoxShiftEstimator",
    "ClassifyAndCount",
    "DiscreteCategoricalMeanEstimator",
    "InvariantRatioEstimator",
    "SummaryStatistic",
    "SamplingParams",
]
