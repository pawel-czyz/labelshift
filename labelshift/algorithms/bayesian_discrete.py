"""Categorical discrete Bayesian model for quantification."""

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax
import pydantic
from typing import Optional

import labelshift.interfaces.point_estimators as pe


class SamplingParams(pydantic.BaseModel):
    """Settings for the MCMC sampler."""

    warmup: pydantic.PositiveInt = pydantic.Field(default=500)
    samples: pydantic.PositiveInt = pydantic.Field(default=1000)
    chains: pydantic.PositiveInt = pydantic.Field(default=1)


P_TRAIN_Y: str = "P_train(Y)"
P_TEST_Y: str = "P_test(Y)"
P_TEST_C: str = "P_test(C)"
P_C_COND_Y: str = "P(C|Y)"


def model(summary_statistic, alpha: float = 1.0):
    n_y_labeled = summary_statistic.n_y_labeled
    n_y_and_c_labeled = summary_statistic.n_y_and_c_labeled
    n_c_unlabeled = summary_statistic.n_c_unlabeled
    K = len(n_c_unlabeled)
    L = len(n_y_labeled)

    pi = numpyro.sample(P_TRAIN_Y, dist.Dirichlet(alpha * jnp.ones(L)))
    pi_ = numpyro.sample(P_TEST_Y, dist.Dirichlet(alpha * jnp.ones(L)))
    p_c_cond_y = numpyro.sample(
        P_C_COND_Y, dist.Dirichlet(alpha * jnp.ones(K).repeat(L).reshape(L, K))
    )

    N_y = numpyro.sample(
        "N_y", dist.Multinomial(jnp.sum(n_y_labeled), pi), obs=n_y_labeled
    )

    with numpyro.plate("plate", L):
        numpyro.sample("F_yc", dist.Multinomial(N_y, p_c_cond_y), obs=n_y_and_c_labeled)

    p_c = numpyro.deterministic(P_TEST_C, jnp.einsum("yc,y->c", p_c_cond_y, pi_))
    numpyro.sample(
        "N_c", dist.Multinomial(jnp.sum(n_c_unlabeled), p_c), obs=n_c_unlabeled
    )


class DiscreteCategoricalMeanEstimator(pe.SummaryStatisticPrevalenceEstimator):
    """A version of Bayesian quantification which finds the mean solution.

    Note that it runs the MCMC sampler in the backend.
    """

    P_TRAIN_Y = P_TRAIN_Y
    P_TEST_Y = P_TEST_Y
    P_TEST_C = P_TEST_C
    P_C_COND_Y = P_C_COND_Y

    def __init__(
        self,
        params: Optional[SamplingParams] = None,
        seed: int = 42,
        alpha: float = 1.0,
    ) -> None:
        if params is None:
            params = SamplingParams()
        self._params = params
        self._seed = seed
        self._mcmc = None

        if alpha <= 0:
            raise ValueError("Concentration parameter alpha has to be positive.")
        self._alpha = alpha

    def sample_posterior(self, /, statistic: pe.SummaryStatistic):
        """Returns the samples from the MCMC sampler."""
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model),
            num_warmup=self._params.warmup,
            num_samples=self._params.samples,
            num_chains=self._params.chains,
        )
        rng_key = jax.random.PRNGKey(self._seed)
        mcmc.run(rng_key, summary_statistic=statistic, alpha=self._alpha)
        self._mcmc = mcmc
        return mcmc.get_samples()

    def estimate_from_summary_statistic(
        self, /, statistic: pe.SummaryStatistic
    ) -> np.ndarray:
        """Returns the mean prediction."""
        samples = self.sample_posterior(statistic)[P_TEST_Y]
        return np.array(samples.mean(axis=0))

    def get_mcmc(self):
        """Returns the MCMC object."""
        if self._mcmc is None:
            raise ValueError("Run `sample_posterior` to obtain MCMC samples first.")
        return self._mcmc
