"""Python package for label shift.

Exports:
    BaseQuantificationAlgorithm, a base class for quantification algorithm. All algorithms to be benchmarked methods should sublass from it
"""
from labelshift.algorithms.base import BaseQuantificationAlgorithm

from labelshift.recalibrate import recalibrate

__all__ = [
    "recalibrate",
]