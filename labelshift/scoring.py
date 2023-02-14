"""Evaluation metrics for quantification methods.

See Section 4 of
P. Gonzalez, A. Castano, N.V. Chawla, J. J. del Coz,
A Review on Quantification Learning,
ACM Computing Surveys, Vol. 50, No. 5. DOI: https://dl.acm.org/doi/10.1145/3117807
"""
from typing import cast, Protocol

import numpy as np
from numpy.typing import ArrayLike

import labelshift.probability as prob


class MulticlassQuantificationError(Protocol):
    """Base class for scoring multi-class quantification methods.
    Every child class needs to implement `_calculate_error` method.

    Methods:
        error: measures the error of an estimate of true prevalences
    """

    def _calculate_error(self, true: np.ndarray, estimated: np.ndarray) -> float:
        """The main method `error()` is a wrapper around this one.

        Args:
            true: array with true prevalences, shape (n,)
            estimated: array with estimated prevalences, shape (n,)

        Returns:
            mismatch measure (error of the estimate)
        """
        raise NotImplementedError

    def error(self, true: ArrayLike, estimated: ArrayLike) -> float:
        """Score a method basing on true and found prevalences.

        Args:
            true: ground truth prevalences. Will be normalized.
            estimated: an estimate for the prevalences. Will be normalized.

        Returns:
            mismatch measure (error of the estimate)

        Raises:
            ValueError, if the number of classes in `true` and `estimated`
                is different
        """
        true = prob.normalize_prevalences(true).ravel()
        estimated = prob.normalize_prevalences(estimated).ravel()

        if len(true) != len(estimated):
            raise ValueError(
                f"Number of classes mismatch: {len(true)} != {len(estimated)}."
            )

        return self._calculate_error(true=true, estimated=estimated)


class AbsoluteError(MulticlassQuantificationError):
    """Standard mean absolute error.

    Ranges between :math:`0` (perfect match) and :math:`2 \\cdot (1 - m) / l`, where
    :math:`m` is the smallest entry in the ground truth prevalence vector
    and :math:`l` is the number of classes.

    See also:
        NormalizedAbsoluteError
    """

    def _calculate_error(self, true: np.ndarray, estimated: np.ndarray) -> float:
        return cast(float, np.mean(np.abs(true - estimated)))


class NormalizedAbsoluteError(MulticlassQuantificationError):
    """Normalized version of the mean absolute error,
    ranging between 0 (best) and 1 (worst).

    It was introduced in:

    W. Gao, F. Sebastiani,
    From classification to quantification in tweet sentiment analysis.
    Soc. Netw. Anal. Min. 6, 1 (2016), 1--22.

    See also:
        AbsoluteError
    """

    def _calculate_error(self, true: np.ndarray, estimated: np.ndarray) -> float:
        sum_absolute_error = np.sum(np.abs(true - estimated))
        normalization = 2 * (1 - np.min(true))

        return sum_absolute_error / normalization


def _smooth_vector(prevalences: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """Apply smoothing to prevalence vector.

    Args:
        prevalences: prevalence vector, shape (n,)
        smoothing: smoothing constant, value 0 implies no smoothing

    Returns
        smoothed prevalence vector, shape (n,)
    """
    numerator = prevalences + smoothing
    denominator = smoothing * len(prevalences) + np.sum(prevalences)

    return numerator / denominator


class RelativeAbsoluteError(MulticlassQuantificationError):
    """Relative absolute error with smoothing.

    It was discussed in:

    W. Gao, F. Sebastiani,
    From classification to quantification in tweet sentiment analysis.
    Soc. Netw. Anal. Min. 6, 1 (2016), 1--22.

    and

    G. Da San Martino, W. Gao, and F. Sebastiani, Ordinal text quantification.
    In Proceedings of the International ACM SIGIR Conference on Research and Development
    in Information Retrieval (2016). 937--940

    Properties:
        smoothing: float, the smoothing constant
    """

    def __init__(self, smoothing: float = 0.0) -> None:
        """
        Args:
            smoothing: smoothing constant (needed if the prevalence of any class
                in the ground truth distribution is 0)
        """
        self.smoothing = smoothing

    def _calculate_error(self, true: np.ndarray, estimated: np.ndarray) -> float:
        # Smooth the values
        p = _smooth_vector(true, smoothing=self.smoothing)
        p_hat = _smooth_vector(estimated, smoothing=self.smoothing)

        return cast(float, np.mean(np.abs(p_hat - p) / p))


class BrayCurtisDissimilarity(MulticlassQuantificationError):
    """Bray-Curtis dissimilarity error, ranging between 0 (the best) and 1 (the worst).

    Note:
        This is not metric, as it does not satisfy the triangle inequality.

    Introduced in

    J. R. Bray and J. T. Curtis.
    An ordination of the upland forest communities of southern Wisconsin.
    Ecol. Monogr. 27, 4 (1957), 325--349
    """

    def _calculate_error(self, true: np.ndarray, estimated: np.ndarray) -> float:
        numerator = np.sum(np.abs(true - estimated))
        denominator = np.sum(true + estimated)
        return numerator / denominator


class HellingerDistance(MulticlassQuantificationError):
    """Hellinger distance, ranging between 0 (the best) and 1 (the worst).

    For more information see:

    https://en.wikipedia.org/wiki/Hellinger_distance#Discrete_distributions
    """

    def _calculate_error(self, true: np.ndarray, estimated: np.ndarray) -> float:
        s = np.sum((np.sqrt(true) - np.sqrt(estimated)) ** 2)

        return (s / 2) ** 0.5


class KLDivergence(MulticlassQuantificationError):
    """Kullblack-Leibler divergence

    KL (true || estimated)
    """

    def error(self, true: ArrayLike, estimated: ArrayLike) -> float:
        return cast(float, np.sum(true * np.log(true / estimated)))


class SymmetrisedKLDivergence(MulticlassQuantificationError):
    """Symmetrised Kullback-Leibler divergence:

    KL( p || q ) + KL( q || p)

    It is also called Jeffreys divergence and has been described here:

    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence

    Note:
        We use natural logarithm here, so the unit is nat, rather than bit.
    """

    def __init__(self) -> None:
        self._kl = KLDivergence()

    def _calculate_error(self, true: np.ndarray, estimated: np.ndarray) -> float:
        return self._kl.error(true, estimated) + self._kl.error(estimated, true)


class FisherRaoDistance(MulticlassQuantificationError):
    def error(self, true: ArrayLike, estimated: ArrayLike) -> float:
        bhattacharyya_coefficient = np.dot(np.sqrt(true), np.sqrt(estimated))
        return 2 * np.arccos(bhattacharyya_coefficient)
