"""This is a toy Bayesian quantification algorithm,
which assumes that the data were generated according
to a mixture of Gaussians.

Note:
    This algorithm models the data in 1D.
"""
from typing import Optional, Sequence, Tuple

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike

MEANS: str = "mu"
SIGMAS: str = "sigma"
P_UNLABELED_Y: str = "P_unabeled(Y)"


def build_model(
    labeled_data: Sequence[ArrayLike],
    unlabeled_data: ArrayLike,
    mean_params: Tuple[float, float] = (0.0, 1.0),
    sigma_param: float = 1.0,
    alpha: Optional[ArrayLike] = None,
) -> pm.Model:
    """Builds a PyMC model for Bayesian quantification for 1D data
    assumed to be sampled from a mixture of normals.

    Args:
        labeled_data: a list of arrays. The ith list should
          contain the X samples from the ith component, so that
          the length of this list is (n_components,).
          Each of the inside arrays should be of shape
          (n_samples_per_given_component,)
          and, of course, these may be of different lengths
        unlabeled_data: the X samples from the unlabeled
          data distribution. Shape (n_unlabeled,)
        mean_params: used to initialize the prior on the component means
        sigma_param: used to initialize the prior on the component sigmas
        alpha: used to initialize the Dirichlet prior on P_unlabeled(Y).
          Shape (n_components,)

    Returns:
        a PyMC model with the following variables:
          `MEANS`: the vector with the mean of each component
          `SIGMAS`: the vector with the sigma of each component
          `P_UNLABELED_Y`: the prevalence vector
    """
    unlabeled_data = np.asarray(unlabeled_data)
    # We only support 1D mixtures
    assert unlabeled_data.shape == (len(unlabeled_data),)

    n_y = len(labeled_data)
    if alpha is None:
        alpha = np.ones(n_y)
    else:
        alpha = np.asarray(alpha)

    if alpha.shape != (n_y,):
        raise ValueError(f"Shape mismatch: {alpha.shape} != ({n_y},).")

    with pm.Model() as gaussian_mixture:
        mu = pm.Normal("mu", mu=mean_params[0], sigma=mean_params[1], shape=n_y)
        sigma = pm.HalfNormal("sigma", sigma=sigma_param, shape=n_y)

        # For each component we have some samples, which can be used
        # to constrain the mean and sigma of this distribution
        for i in range(n_y):
            pm.Normal(
                f"X_labeled{i}", mu=mu[i], sigma=sigma[i], observed=labeled_data[i]
            )

        p_unlabeled_y = pm.Dirichlet("P_unlabeled(Y)", alpha)
        # We sample the data points
        pm.NormalMixture(
            "X_unlabeled", w=p_unlabeled_y, mu=mu, sigma=sigma, observed=unlabeled_data
        )

    return gaussian_mixture
