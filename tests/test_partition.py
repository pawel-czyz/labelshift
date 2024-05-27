"""Tests of the `partition` submodule."""

import numpy as np
import pytest
from scipy import stats

import labelshift.partition as part


class TestRealLinePartition:
    """Tests of the partition class."""

    @pytest.mark.parametrize("k", (2, 3, 10))
    def test_length(self, k: int) -> None:
        """Tests whether the length of the generated partition
        is right.

        Args:
            k: the length of the partition
        """
        breakpoints = np.arange(k - 1)

        partition = part.RealLinePartition(breakpoints)

        assert len(partition) == k
        for i in range(k):
            a, b = partition.interval(i)
            assert a < b

        for i in range(k, k + 10):
            with pytest.raises(IndexError):
                partition.interval(k)

    @pytest.mark.parametrize("a", [0.1, -0.5, 0.8])
    def test_two_intervals(self, a: float) -> None:
        """We use one breakpoint to split the real line
        into two halves.

        Args:
            a: breakpoint
        """
        partition = part.RealLinePartition([a])

        assert len(partition) == 2
        assert partition.interval(0) == (-np.inf, a)
        assert partition.interval(1) == (a, np.inf)

        with pytest.raises(IndexError):
            partition.interval(2)


class TestGaussianProbabilityMasses:
    """Tests of `gaussian_probability_masses` function."""

    @pytest.mark.parametrize("mean", [0.1, 0.6, 10.0])
    @pytest.mark.parametrize("sigma", [1.0, 2.0])
    @pytest.mark.parametrize("a", [-0.3, 0.1, 1.0])
    def test_one_gaussian_halves(self, mean: float, sigma: float, a: float) -> None:
        """We have L = 1 and K = 2."""

        result = part.gaussian_probability_masses(
            means=[mean], sigmas=[sigma], partition=part.RealLinePartition([a])
        )
        assert result.shape == (1, 2)
        assert result[0][0] == pytest.approx(stats.norm.cdf(a, loc=mean, scale=sigma))
        assert result.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("K", (2, 3, 10))
    @pytest.mark.parametrize("L", (1, 2, 5))
    def test_shape(self, K: int, L: int) -> None:
        """Tests if the generated shape is right."""
        partition = part.RealLinePartition(np.linspace(0, 1, K - 1))
        assert len(partition) == K

        means = np.linspace(-5, 5, L)
        sigmas = np.linspace(0.1, 1.0, L)

        output = part.gaussian_probability_masses(
            means=means, sigmas=sigmas, partition=partition
        )

        assert output.shape == (L, K)
