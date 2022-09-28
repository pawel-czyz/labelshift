"""Tests for the BBSE submodule."""
import pytest
import numpy as np

import labelshift.algorithms.bbse as bbse
import labelshift.datasets.discrete_categorical as dc


def invertible_matrix(n: int, rng, jitter: float = 1e-3) -> np.ndarray:
    """Sample from the Wishart distribution."""
    rng = np.random.default_rng(rng)

    m = rng.normal(0, 1, size=(n, n)) + jitter * np.eye(n)

    if np.linalg.matrix_rank(m) < n:
        return invertible_matrix(n, rng)
    else:
        return m


class TestSolveSystem:
    """Tests for the solve_system."""

    @pytest.mark.parametrize("n", (2, 3, 10))
    @pytest.mark.parametrize("enforce_squares", (True, False))
    @pytest.mark.parametrize("seed", range(4))
    def test_solve_simple_square(
        self, n: int, enforce_squares: bool, seed: int
    ) -> None:
        """Test whether a good solution is found for random square matrices
        and random vectors.

        Tests also that the transposed matrix doesn't work,
        to avoid transposition errors.
        """
        rng = np.random.default_rng(seed)

        a = invertible_matrix(n, rng=rng)
        assert a.shape == (n, n)
        assert a != pytest.approx(a.T)

        b = rng.uniform(0, 1, size=n)

        x = bbse.solve_system(matrix=a, vector=b, square_solver=enforce_squares)
        assert x.shape == (n,)

        b_ = np.einsum("ij,j->i", a, x)
        assert b == pytest.approx(b_)

        x_wrong = bbse.solve_system(matrix=a.T, vector=b, square_solver=enforce_squares)
        assert np.einsum("ij,j->i", a, x_wrong) != pytest.approx(b)


def test_bbse_from_sufficient_statistic(
    n_labeled: int = 20_000, n_unlabeled: int = 20_000
) -> None:
    """Generates the data according to the P(C|Y) model."""
    p_y_labeled = np.asarray([0.4, 0.6])
    p_y_unlabeled = np.asarray([0.7, 0.3])

    p_c_cond_y = np.asarray(
        [
            [0.95, 0.04, 0.01],
            [0.04, 0.95, 0.01],
        ]
    )

    sampler = dc.DiscreteSampler(
        p_y_labeled=p_y_labeled, p_y_unlabeled=p_y_unlabeled, p_c_cond_y=p_c_cond_y
    )
    statistic = sampler.sample_summary_statistic(
        n_labeled=n_labeled, n_unlabeled=n_unlabeled, seed=111
    )

    p_y_estimated = bbse.from_sufficient_statistic(
        n_y_and_c_labeled=statistic.n_y_and_c_labeled,
        n_c_unlabeled=statistic.n_c_unlabeled,
    )

    assert p_y_estimated == pytest.approx(p_y_unlabeled, abs=0.01)
