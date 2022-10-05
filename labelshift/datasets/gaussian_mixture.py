"""Model used for working with exact probabilities
in the Gaussian mixture model."""
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

    def p_x_cond_y_fixed(self, xs: ArrayLike, y: int) -> np.ndarray:
        """Calculates the probability density function p(x|y).

        Args:
            xs: vector of points `x`, shape (n_points, dim_x)
            y: label `y`

        Returns
            vector of probability density function values p(X=x | Y=y),
              for each point `x`. Shape (n_points,)
        """
        raise NotImplementedError

    def p_x_cond_y(self, xs: ArrayLike) -> np.ndarray:
        """Calculates the probability density function p(x|y)
        for all possible `y`.

        Args:
            xs: vector of points `x`, shape (n_points, dim_x)

        Returns:
            probability density function values p(X=x | Y=y),
              for each point `x` and label `y`.
              Shape (n_points, number_y).
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

        if covariances.shape != (means.shape[0], means.shape[1], means.shape[1]):
            raise ValueError(f"Covariance matrix has wrong shape: {covariances.shape}")

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

    def p_x_cond_y_fixed(self, xs: ArrayLike, y: int) -> np.ndarray:
        """See the parent class."""
        xs = np.asarray(xs)
        dist = self._distributions[y]
        return dist.pdf(xs)

    def p_x_cond_y(self, xs: ArrayLike) -> np.ndarray:
        """See the parent class."""
        xs = np.asarray(xs)
        return np.vstack(
            [self.p_x_cond_y_fixed(xs, y=y) for y in range(self.number_y)]
        ).T

    def sample_p_x_cond_y(self, ys: ArrayLike, rng) -> np.ndarray:
        """See the parent class."""
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
    p_y = np.asarray(p_y)
    xs = np.asarray(xs)
    assert p_y.shape == (p_x_cond_y.number_y,)
    assert xs.shape[1] == p_x_cond_y.dim_x

    # P(X=x, Y=y) = P(X = x | Y = y) * P(Y=y)
    # calculated pointwise.
    # Shape (n_points, number_y)
    p_x_and_y = p_x_cond_y.p_x_cond_y(xs) * p_y[None, :]

    # We need to normalize p(x, y) to get p(y|x)
    return p_x_and_y / p_x_and_y.sum(axis=1)[None, :]
