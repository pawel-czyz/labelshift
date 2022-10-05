"""Tests of the Gaussian mixture module."""
import numpy as np
import pytest

import labelshift.datasets.gaussian_mixture as gm


class TestPosteriorProbabilities:
    @pytest.mark.parametrize("n_components", [1, 3])
    @pytest.mark.parametrize("dim", [1, 4])
    def test_sums_up_to_one(self, n_components: int, dim: int, seed: int = 0) -> None:
        """Tests whether the posterior P(Y|X=x) sums up to one."""
        rng = np.random.default_rng(seed)
        means = rng.uniform(size=(n_components, dim))
        sigmas = [np.eye(dim) for _ in range(n_components)]

        distribution = gm.GaussianMixturePXGivenY(means=means, covariances=sigmas)

        posterior = gm.posterior_probabilities(
            p_y=np.ones(n_components) / n_components,
            xs=rng.uniform(size=(1, dim)),
            p_x_cond_y=distribution,
        )
        assert posterior.shape == (1, n_components), f"Shape is wrong: {posterior}"
        assert posterior.sum() == pytest.approx(1.0), f"Sum is wrong: {posterior}"


# TODO(Pawel): Add more tests.
