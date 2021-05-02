"""The quantification benchmark package.

Exports:
    BaseQuantificationAlgorithm, a base class for quantification algorithm. All algorithms to be benchmarked methods should sublass from it
"""
from qantipy.base import BaseQuantificationAlgorithm

from quantipy.recalibrate import recalibrate

__all__ = [
    "recalibrate",
]