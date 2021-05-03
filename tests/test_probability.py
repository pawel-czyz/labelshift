import numpy.testing as nptest
import pytest

import labelshift.probability as prob


def test_shape() -> None:
    a = [1, 7, 10]
    prevalences = prob.normalize_prevalences(a)
    assert prevalences.shape == (1, 3), "Shape mismatch."


def test_values() -> None:
    a = [1, 3, 4, 2]
    prevalences = prob.normalize_prevalences(a)
    nptest.assert_allclose(prevalences.ravel(), [0.1, 0.3, 0.4, 0.2])


def test_error_raised() -> None:
    with pytest.raises(ValueError):
        prob.normalize_prevalences([0.1, 0.0, 0.3])


@pytest.mark.parametrize("p", (0.1, 0.2, -1.0))
def test_nonnormalizable(p: float) -> None:
    with pytest.raises(ValueError):
        prob.normalize_prevalences([-p, p])
