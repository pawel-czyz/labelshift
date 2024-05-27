"""The experimental utilities."""

from typing import TypeVar, Optional

from labelshift.experiments.timer import Timer
from labelshift.experiments.names import generate_name

_T = TypeVar("_T")


def calculate_value(*, overwrite: Optional[_T], default: _T) -> _T:
    if overwrite is None:
        return default
    else:
        return overwrite


__all__ = [
    "Timer",
    "calculate_value",
    "generate_name",
    "calculate_value",
]
