"""The experimental utilities."""

from typing import TypeVar, Optional

from labelshift.experiments.timer import Timer

_T = TypeVar("_T")


def calculate_value(*, overwrite: Optional[_T], default: _T) -> _T:
    if overwrite is None:
        return default
    else:
        return overwrite


__all__ = [
    "Timer",
    "calculate_value",
    "calculate_value",
]
