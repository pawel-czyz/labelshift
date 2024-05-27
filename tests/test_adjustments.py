"""Tests for adjustment submodule."""

import numpy as np
import numpy.testing as nptest
import pytest

import labelshift.adjustments as adj


# TODO(pawel-czyz): These tests can be refactored with parametrization fixture.
def test_label_hardening() -> None:
    """Test if soft-labels are properly converted
    into one-hot vectors."""

    softlabels = [
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
    ]

    onehot = [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    onehot1 = adj.label_hardening(softlabels)
    nptest.assert_allclose(onehot1, onehot)


def test_label_hardening_square() -> None:
    """Test if soft-labels are properly converted
    into one-hot vectors. Square matrix."""

    softlabels = [
        [0.1, 0.9],
        [0.2, 0.8],
    ]

    onehot = [
        [0.0, 1.0],
        [0.0, 1.0],
    ]

    onehot1 = adj.label_hardening(softlabels)
    nptest.assert_allclose(onehot1, onehot)


class TestOntoProbabilitySimplex:
    """Tests for projections onto the probability simplex."""

    @pytest.mark.parametrize("n", range(4, 10))
    @pytest.mark.parametrize("seed", range(3))
    def test_preserved(self, n: int, seed: int) -> None:
        """Tests whether a randomly sampled point of the simplex is preserved."""
        rng = np.random.default_rng(seed)
        point = rng.dirichlet(alpha=np.ones(n))

        assert point.shape == (n,)
        assert point.sum() == pytest.approx(1)

        assert adj.project_onto_probability_simplex(point) == pytest.approx(point)

    @pytest.mark.parametrize("p1", (0.0, 0.1, 0.5))
    @pytest.mark.parametrize("distance", (-2, 0.2, 12))
    def test_2d_case(self, p1: float, distance: float) -> None:
        """This test uses the fact that 1-simplex is in 2D space and it's easy to
        construct a point on a perpendicular ray.

        Args:
            p1: parametrizes the 1-simplex, between 0 and 1
            distance: oriented (can be negative) distance from the simplex
              to the generated point
        """
        simplex_point = np.array([1 - p1, p1])
        point = distance * np.ones(2) + simplex_point

        assert adj.project_onto_probability_simplex(point) == pytest.approx(
            simplex_point
        )
