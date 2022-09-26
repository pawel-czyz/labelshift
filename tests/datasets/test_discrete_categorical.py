from typing import Sequence

import numpy as np
import pytest

import labelshift.datasets.discrete_categorical as dc


class TestAlmostEye:
    """The tests for `almost_eye` method."""

    @pytest.mark.parametrize("y", (2, 5))
    @pytest.mark.parametrize("c", (2, 3, 8))
    @pytest.mark.parametrize("diagonal", (0.2, 0.5))
    def test_shape_sum_max(self, y: int, c: int, diagonal: float) -> None:
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

    def test_simple_example(self) -> None:
        obtained = dc.almost_eye(y=2, c=3, diagonal=0.6)
        expected = np.asarray(
            [
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
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
        rng = np.random.default_rng(111)

        p_c_cond_y = rng.dirichlet(alpha=np.ones(n_c), size=len(p_y))
        p_y_unlabeled = np.ones_like(p_y) / len(p_y)

        sampler = dc.DiscreteSampler(
            p_y_labeled=p_y,
            p_y_unlabeled=p_y_unlabeled,
            p_c_cond_y=p_c_cond_y,
        )

        summary_stats = sampler.sample_summary_statistic(
            n_labeled=n_labeled, n_unlabeled=1
        )

        assert summary_stats.n_y_labeled.sum() == n_labeled
        assert summary_stats.n_y_labeled / n_labeled == pytest.approx(p_y, abs=0.02)

        assert summary_stats.n_y_and_c_labeled.sum(axis=1) == pytest.approx(
            summary_stats.n_y_labeled
        )

    def test_p_c(self) -> None:
        p_y_labeled = [0.1, 0.9]
        p_y_unlabeled = [0.2, 0.8]
        p_c_cond_y = [
            [1.0, 0, 0],
            [0.0, 1, 0],
        ]

        sampler = dc.DiscreteSampler(
            p_y_labeled=p_y_labeled, p_y_unlabeled=p_y_unlabeled, p_c_cond_y=p_c_cond_y
        )

        assert sampler.p_c_labeled == pytest.approx(np.array([0.1, 0.9, 0.0]))
        assert sampler.p_c_unlabeled == pytest.approx(np.array([0.2, 0.8, 0.0]))

    @pytest.mark.parametrize("n_y", (2, 5))
    @pytest.mark.parametrize("n_c", (2, 3))
    def test_create_right_fields(self, n_c: int, n_y: int) -> None:
        rng = np.random.default_rng(111)

        p_c_cond_y = rng.dirichlet(alpha=np.ones(n_c), size=n_y)
        p_y_unlabeled = rng.dirichlet(alpha=np.ones(n_y))
        p_y_labeled = rng.dirichlet(alpha=np.ones(n_y))

        sampler = dc.DiscreteSampler(
            p_y_labeled=p_y_labeled,
            p_y_unlabeled=p_y_unlabeled,
            p_c_cond_y=p_c_cond_y,
        )

        assert sampler.p_c_cond_y == pytest.approx(p_c_cond_y)
        assert sampler.p_y_labeled == pytest.approx(p_y_labeled)
        assert sampler.p_y_unlabeled == pytest.approx(p_y_unlabeled)
