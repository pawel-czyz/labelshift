"""Categorical discrete Bayesian model for quantification.

Proposed in
   TODO(Pawel): Add citation to pre-print after AISTATS reviews.
"""
from typing import cast, NewType, Optional

import arviz as az
import numpy as np
import pydantic
import pymc as pm

from numpy.typing import ArrayLike

P_TRAIN_Y: str = "P_train(Y)"
P_TEST_Y: str = "P_test(Y)"
P_TEST_C: str = "P_test(C)"
P_C_COND_Y: str = "P(C|Y)"


class SamplingParams(pydantic.BaseModel):
    """Settings for the MCMC sampler."""

    draws: pydantic.PositiveInt = pydantic.Field(default=1000)
    chains: pydantic.PositiveInt = pydantic.Field(default=4)
    random_seed: int = 20


DiscreteBayesianQuantificationModel = NewType("", pm.Model)


def build_model(
    n_y_labeled: ArrayLike,
    n_y_and_c_labeled: ArrayLike,
    n_c_unlabeled: ArrayLike,
    alpha_p_y_labeled: Optional[ArrayLike] = None,
    alpha_p_y_unlabeled: Optional[ArrayLike] = None,
) -> DiscreteBayesianQuantificationModel:
    """Builds the discrete Bayesian quantification model,
     basing on the sufficient statistic of the data.

    Args:
        n_y_labeled: histogram of Y labels in the visible data set  shape (L,),
            where L is the number of labels
        n_y_and_c_labeled: histogram of Y and C labels in the labeled data set,
            shape (L, K)
        n_c_unlabeled: histogram of C in the unlabeled data set, shape (K,)
    """
    n_y_labeled = np.asarray(n_y_labeled)
    n_y_and_c_labeled = np.asarray(n_y_and_c_labeled)
    n_c_unlabeled = np.asarray(n_c_unlabeled)

    assert len(n_y_and_c_labeled.shape) == 2
    L, K = n_y_and_c_labeled.shape

    assert n_y_labeled.shape == (L,)
    assert n_c_unlabeled.shape == (K,)

    alpha_p_y_labeled = (
        np.ones(L) if alpha_p_y_labeled is None else np.asarray(alpha_p_y_labeled)
    )
    alpha_p_y_unlabeled = (
        np.ones(L) if alpha_p_y_unlabeled is None else np.asarray(alpha_p_y_unlabeled)
    )

    assert alpha_p_y_labeled.shape == (L,)
    assert alpha_p_y_unlabeled.shape == (L,)

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
