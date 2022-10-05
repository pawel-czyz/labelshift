from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


class PXGivenY(Protocol):
    """This is a protocol defining the P(X|Y) distribution,
    where X is a vector-valued random variable and Y is a discrete variable.
    """

    @property
    def dim_x(self) -> int:
        """Dimension of the vector space in which X takes values."""
        raise NotImplementedError

    @property
    def number_y(self) -> int:
        """The number of possible labels Y."""
        raise NotImplementedError

    def p_x_cond_y(self, xs: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """Calculates the probability density function p(x|y).

        Args:
            xs: vector of points `x`, shape (n_points, dim_x)
            ys: vector of labels `y`, shape (n_points,)

        Returns
            vector of probability density function values p(X=x | Y=y),
              for each point. Shape (n_points,)
        """
        raise NotImplementedError

    def sample_p_x_cond_y(self, ys: ArrayLike, rng) -> np.ndarray:
        """Samples from P(X|Y) for given Y.

        Args:
            ys: labels `y`, shape (n_points,)
            rng: random number generator

        Returns:
            a random vector from P(X|Y=y), for each y in `ys`.
              Shape (n_points, dim_x)
        """
        raise NotImplementedError


class GaussianMixturePXGivenY(PXGivenY):
    """Generative process in which each P(X|Y)
    is a multivariate normal distribution.
    """

    def __init__(self, means: ArrayLike, covariances: ArrayLike) -> None:
        """
        Args:
            means: mean of the Gaussian associated to each label,
              shape (number_y, dim_x)
            covariances: covariance of the Gaussian associated to each label,
              shape (number_y, dim_x, dim_x)
        """
        means = np.asarray(means)
        covariances = np.asarray(covariances)
        self._dim_x = means.shape[1]
        self._number_y = means.shape[0]

        assert covariances.shape == (means.shape[0], means.shape[1], means.shape[1])

        self._distributions = [
            stats.multivariate_normal(
                mean=means[y, :],
                cov=covariances[y, :, :],
                allow_singular=False,
            )
            for y in range(self._number_y)
        ]

    @property
    def number_y(self) -> int:
        """The number of possible labels Y."""
        return self._number_y

    @property
    def dim_x(self) -> int:
        """Dimension of the vector space in which X takes values."""
        return self._dim_x

    def p_x_cond_y(self, xs: ArrayLike, ys: ArrayLike) -> np.ndarray:
        """The PDF p(X = x | Y = y) for given points."""
        return np.asarray([self._distributions[y].pdf(x) for x, y in zip(xs, ys)])

    def sample_p_x_cond_y(self, ys: ArrayLike, rng) -> np.ndarray:
        """Random samples from distributions associated to the labels."""
        rng = np.random.default_rng(rng)
        return np.vstack([self._distributions[y].rvs(random_state=rng) for y in ys])


def posterior_probabilities(
    p_y: ArrayLike, xs: ArrayLike, p_x_cond_y: PXGivenY
) -> np.ndarray:
    """Bayes rule for inference of P(Y|X).

    Args:
        p_y: prior probabilities (number_y,)
        xs: covariates, shape (n_points, dim_x)
        p_x_cond_y: a model of P(X|Y). Note that it needs to be
          compatible with `number_y` and `dim_x` shapes
          of the other arguments

    Returns:
        posterior probability P(Y|X) for each data point.
          Shape (n_points, number_y)
    """
    raise NotImplementedError
