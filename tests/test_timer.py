"""Tests of the `timer` submodule."""

import time

import pytest

from labelshift.experiments.timer import Timer


@pytest.mark.parametrize("t1", [0.1, 0.2])
@pytest.mark.parametrize("t2", [0.05, 0.1])
def test_timer(t1: float, t2: float) -> None:
    """Simple test with sleeping for `t` seconds."""
    timer = Timer()
    time.sleep(t1)
    t_ = timer.check()

    assert t_ == pytest.approx(t1, abs=0.01)

    timer.reset()
    time.sleep(t2)
    t_ = timer.check()

    assert t_ == pytest.approx(t2, abs=0.01)
