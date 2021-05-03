"""Tests for the auxilary probability submodule."""
import numpy.testing as nptest
import pytest

import labelshift.probability as prob


def test_shape() -> None:
    """Tests if the shape returned by `normalize_prevalences` is correct."""
    a = [1, 7, 10]
    prevalences = prob.normalize_prevalences(a)
    assert prevalences.shape == (1, 3), "Shape mismatch."


def test_values() -> None:
    """Checks the normalized value."""
    a = [1, 3, 4, 2]
    prevalences = prob.normalize_prevalences(a)
    nptest.assert_allclose(prevalences.ravel(), [0.1, 0.3, 0.4, 0.2])


def test_error_raised() -> None:
    """Checks if an error is raised for non-positive probability."""
    with pytest.raises(ValueError):
        prob.normalize_prevalences([0.1, 0.0, 0.3])


@pytest.mark.parametrize("p", (0.1, 0.2, -1.0))
def test_nonnormalizable(p: float) -> None:
    """Entries sum up to 0, so that it is not normalizable."""
    with pytest.raises(ValueError):
        prob.normalize_prevalences([-p, p])
