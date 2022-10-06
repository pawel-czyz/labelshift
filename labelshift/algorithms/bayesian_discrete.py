"""Categorical discrete Bayesian model for quantification.

Proposed in
   TODO(Pawel): Add citation to pre-print after AISTATS reviews.
"""
from typing import cast, NewType, Optional, Union

import arviz as az
import numpy as np
import pydantic
import pymc
import pymc as pm

from numpy.typing import ArrayLike

import labelshift.interfaces.point_estimators as pe


P_TRAIN_Y: str = "P_train(Y)"
P_TEST_Y: str = "P_test(Y)"
P_TEST_C: str = "P_test(C)"
P_C_COND_Y: str = "P(C|Y)"


class SamplingParams(pydantic.BaseModel):
    """Settings for the MCMC sampler."""

    draws: pydantic.PositiveInt = pydantic.Field(default=1000)
    chains: pydantic.PositiveInt = pydantic.Field(default=4)
    random_seed: int = 20


DiscreteBayesianQuantificationModel = NewType(
    "DiscreteBayesianQuantificationModel", pm.Model
)


def dirichlet_alphas(L: int, alpha: Union[float, ArrayLike]) -> np.ndarray:
    """Convenient initialization of alpha (pseudocounts)
    parameters of the Dirichlet prior.

    Args:
        alpha: either an array of shape (L,) or a float.
          If a float, vector (alpha, alpha, ..., alpha)
          is created

    Returns:
         alphas, shape (L,)
    """
    if isinstance(alpha, float):
        return np.ones(L) * alpha
    else:
        alpha = np.asarray(alpha)
        assert alpha.shape == (L,)
        return alpha


def build_model(
    n_y_and_c_labeled: ArrayLike,
    n_c_unlabeled: ArrayLike,
    alpha_p_y_labeled: Union[float, ArrayLike] = 1.0,
    alpha_p_y_unlabeled: Union[float, ArrayLike] = 1.0,
) -> DiscreteBayesianQuantificationModel:
    """Builds the discrete Bayesian quantification model,
     basing on the sufficient statistic of the data.

    Args:
        n_y_and_c_labeled: histogram of Y and C labels in the labeled data set,
            shape (L, K)
        n_c_unlabeled: histogram of C in the unlabeled data set, shape (K,)
    """
    n_y_and_c_labeled = np.asarray(n_y_and_c_labeled)
    n_y_labeled = n_y_and_c_labeled.sum(axis=1)
    n_c_unlabeled = np.asarray(n_c_unlabeled)

    assert len(n_y_and_c_labeled.shape) == 2
    L, K = n_y_and_c_labeled.shape

    assert n_y_labeled.shape == (L,)
    assert n_c_unlabeled.shape == (K,)

    alpha_p_y_labeled = dirichlet_alphas(L, alpha_p_y_labeled)
    alpha_p_y_unlabeled = dirichlet_alphas(L, alpha_p_y_unlabeled)

    model = pm.Model()
    with model:
        # Prior on pi, pi_, phi
        pi = pm.Dirichlet(P_TRAIN_Y, alpha_p_y_labeled)
        pi_ = pm.Dirichlet(P_TEST_Y, alpha_p_y_unlabeled)
        p_c_cond_y = pm.Dirichlet(P_C_COND_Y, np.ones(K), shape=(L, K))

        # Note: we need to silence unused variable error (F841)

        # Sample N_y from P_train(Y)
        N_y = pm.Multinomial(  # noqa: F841
            "N_y", np.sum(n_y_labeled), p=pi, observed=n_y_labeled
        )

        # Sample the rows
        F_yc = pm.Multinomial(  # noqa: F841
            "F_yc", n_y_labeled, p=p_c_cond_y, observed=n_y_and_c_labeled
        )

        # Sample from P_test(C) = P(C | Y) P_test(Y)
        p_c = pm.Deterministic(P_TEST_C, p_c_cond_y.T @ pi_)
        N_c = pm.Multinomial(  # noqa: F841
            "N_c", np.sum(n_c_unlabeled), p=p_c, observed=n_c_unlabeled
        )

    return cast(DiscreteBayesianQuantificationModel, model)


def sample_from_bayesian_discrete_model_posterior(
    model: DiscreteBayesianQuantificationModel,
    sampling_params: Optional[SamplingParams] = None,
) -> az.InferenceData:
    """Inference in the Bayesian model

    Args:
        model: built model
        sampling_params: sampling parameters, will be passed to PyMC's sampling method
    """
    sampling_params = SamplingParams() if sampling_params is None else sampling_params

    with model:
        inference_data = pm.sample(
            random_seed=sampling_params.random_seed,
            chains=sampling_params.chains,
            draws=sampling_params.draws,
        )

    return inference_data


class DiscreteCategoricalMeanEstimator(pe.SummaryStatisticPrevalenceEstimator):
    """A version of Bayesian quantification which finds the mean solution.

    Note that it runs the MCMC sampler in the backend.
    """

    def __init__(self) -> None:
        """Not implemented yet."""
        raise NotImplementedError

    def estimate_from_summary_statistic(
        self, /, statistic: pe.SummaryStatistic
    ) -> np.ndarray:
        """Returns the mean prediction."""
        raise NotImplementedError


class DiscreteCategoricalMAPEstimator(pe.SummaryStatisticPrevalenceEstimator):
    """A version of Bayesian quantification
    which finds the Maximum a Posteriori solution."""

    def __init__(
        self, max_eval: int = 10_000, alpha_unlabeled: Union[float, ArrayLike] = 1.0
    ) -> None:
        """
        Args:
            max_eval: maximal number of evaluations of the posterior
              during the optimization to find the MAP
        """
        self._max_eval = max_eval
        self._alpha_unlabeled = alpha_unlabeled

    def estimate_from_summary_statistic(
        self, /, statistic: pe.SummaryStatistic
    ) -> np.ndarray:
        """Finds the Maximum a Posteriori (MAP)."""
        model = build_model(
            n_c_unlabeled=statistic.n_c_unlabeled,
            n_y_and_c_labeled=statistic.n_y_and_c_labeled,
            alpha_p_y_unlabeled=self._alpha_unlabeled,
        )
        with model:
            optimal = pymc.find_MAP(maxeval=self._max_eval)
        return optimal[P_TEST_Y]
