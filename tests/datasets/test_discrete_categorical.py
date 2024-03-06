"""Tests for the discrete categorical sampler."""
from typing import Sequence

import numpy as np
import pytest

import labelshift.datasets.discrete_categorical as dc


class TestAlmostEye:
    """The tests for `almost_eye` method."""

    @pytest.mark.parametrize("y", (2, 5))
    @pytest.mark.parametrize("c", (2, 3, 8))
    @pytest.mark.parametrize("diagonal", (0.2, 0.5))
    def test_shape_sum(self, y: int, c: int, diagonal: float) -> None:
        """Tests the shape of the generated matrix and whether each row sums up to 1."""
        m = dc.almost_eye(y=y, c=c, diagonal=diagonal)

        print(m)
        assert m.shape == (y, c), f"Shapes differ: {m.shape} != {(y, c)}."

        assert m.sum(axis=1).shape == (
            y,
        ), f"Shape is terribly wrong: {m.sum(axis=1).shape}"
        assert m.sum(axis=1) == pytest.approx(
            np.ones(y)
        ), f"P(C|Y) is wrong: {m.sum(axis=1)}"

        assert np.min(m) >= 0, f"Minimum: {np.min(m)} should be non-negative"

    def test_simple_example_more_c_than_y(self) -> None:
        """Test on a simple example what to expect."""
        obtained = dc.almost_eye(y=2, c=3, diagonal=0.6)
        expected = np.asarray(
            [
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
            ]
        )
        assert obtained == pytest.approx(expected)

    def test_simple_example_more_y_than_c(self) -> None:
        """Another example, this one is trickier."""
        obtained = dc.almost_eye(y=3, c=2, diagonal=0.6)
        expected = np.asarray(
            [
                [0.6, 0.4],
                [0.4, 0.6],
                [0.5, 0.5],
            ]
        )
        assert obtained == pytest.approx(expected)


class TestDiscreteSampler:
    """Tests for the `DiscreteSampler` class."""

    @pytest.mark.parametrize("n_labeled", (5000, 10000))
    @pytest.mark.parametrize(
        "p_y",
        (np.array([0.1, 0.5, 0.4]), np.array([0.6, 0.2, 0.2]), np.array([0.2, 0.8])),
    )
    @pytest.mark.parametrize("n_c", (2, 3))
    def test_p_y_labeled(self, n_c: int, n_labeled: int, p_y: Sequence[float]) -> None:
        """Tests whether the P_train(Y) looks alright, basing on empirical counts
        and large sample size."""
        rng = np.random.default_rng(111)

        p_c_cond_y = rng.dirichlet(alpha=np.ones(n_c), size=len(p_y))
        p_y_unlabeled = np.ones_like(p_y) / len(p_y)

        sampler = dc.discrete_sampler_factory(
            p_y_labeled=p_y,
            p_y_unlabeled=p_y_unlabeled,
            p_c_cond_y_labeled=p_c_cond_y,
        )

        summary_stats = sampler.sample_summary_statistic(
            n_labeled=n_labeled, n_unlabeled=1, seed=12
        )

        assert summary_stats.n_y_labeled.sum() == n_labeled
        assert summary_stats.n_y_labeled / n_labeled == pytest.approx(p_y, abs=0.02)

        assert summary_stats.n_y_and_c_labeled.sum(axis=1) == pytest.approx(
            summary_stats.n_y_labeled
        )

    def test_p_c(self) -> None:
        """Tests whether P_train(C) and P_test(C) are alright,
        using a trivial P(C|Y) matrix."""
        p_y_labeled = [0.1, 0.9]
        p_y_unlabeled = [0.2, 0.8]
        p_c_cond_y = [
            [1.0, 0, 0],
            [0.0, 1, 0],
        ]

        sampler = dc.discrete_sampler_factory(
            p_y_labeled=p_y_labeled,
            p_y_unlabeled=p_y_unlabeled,
            p_c_cond_y_labeled=p_c_cond_y,
        )

        assert sampler.labeled.p_c == pytest.approx(np.array([0.1, 0.9, 0.0]))
        assert sampler.unlabeled.p_c == pytest.approx(np.array([0.2, 0.8, 0.0]))

    @pytest.mark.parametrize("n_y", (2, 5))
    @pytest.mark.parametrize("n_c", (2, 3))
    def test_create_right_fields(self, n_c: int, n_y: int) -> None:
        """Tests whether the right fields/attributes are available."""
        rng = np.random.default_rng(111)

        p_c_cond_y = rng.dirichlet(alpha=np.ones(n_c), size=n_y)
        p_y_unlabeled = rng.dirichlet(alpha=np.ones(n_y))
        p_y_labeled = rng.dirichlet(alpha=np.ones(n_y))

        sampler = dc.discrete_sampler_factory(
            p_y_labeled=p_y_labeled,
            p_y_unlabeled=p_y_unlabeled,
            p_c_cond_y_labeled=p_c_cond_y,
        )

        assert sampler.labeled.p_c_cond_y == pytest.approx(p_c_cond_y)
        assert sampler.unlabeled.p_c_cond_y == pytest.approx(p_c_cond_y)

        assert sampler.labeled.p_y == pytest.approx(p_y_labeled)
        assert sampler.unlabeled.p_y == pytest.approx(p_y_unlabeled)

        assert sampler.labeled.p_c.shape == (n_c,)
        assert sampler.unlabeled.p_c.shape == (n_c,)
