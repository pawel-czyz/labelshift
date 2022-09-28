"""Tests of the scoring submodule."""
from typing import List

import numpy as np
import pytest

import labelshift.scoring as sc


def error_functions() -> List[sc.MulticlassQuantificationError]:
    """Returns the list of implemented metrics."""
    return [
        sc.AbsoluteError(),
        sc.NormalizedAbsoluteError(),
        sc.RelativeAbsoluteError(),
        sc.BrayCurtisDissimilarity(),
        sc.HellingerDistance(),
        sc.SymmetrisedKLDivergence(),
    ]


@pytest.mark.parametrize("metric", error_functions())
@pytest.mark.parametrize("n", [2, 10])
@pytest.mark.parametrize("seed", range(3))
def test_zero_error_from_itself(
    metric: sc.MulticlassQuantificationError, n: int, seed: int
) -> None:
    """Tests whether we have `M(p, p) = 0` for
    a metric `M` and several distributions `p`"""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=n)
    p = p / p.sum()

    assert metric.error(p, p) == pytest.approx(0)


@pytest.mark.parametrize(
    "metric",
    [
        sc.AbsoluteError(),
        sc.NormalizedAbsoluteError(),
        sc.HellingerDistance(),
        sc.BrayCurtisDissimilarity(),
    ],
)
@pytest.mark.parametrize("n", [2, 10])
@pytest.mark.parametrize("seed", range(3))
def test_between_0_and_1(
    metric: sc.MulticlassQuantificationError, n: int, seed: int
) -> None:
    """Tests on some random examples whether the metric is between 0 and 1."""
    rng = np.random.default_rng(seed)

    p = rng.uniform(0, 1, size=n)
    q = rng.uniform(0, 1, size=n)

    p, q = p / p.sum(), q / q.sum()

    assert metric.error(p, q) > 0
    assert metric.error(p, q) < 1
