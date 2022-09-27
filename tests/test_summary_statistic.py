"""Tests for the module calculating the summary statistic in the discrete case."""
import numpy as np
import pytest

import labelshift.summary_statistic as summ


class TestCountValues:
    """Tests for `count_values`"""

    def test_simple(self) -> None:
        """Test for a simple example with three classes."""
        values = [0, 1, 0, 2, 0]

        obtained = summ.count_values(3, values)
        assert obtained.shape == (3,)
        assert obtained.dtype == int
        assert obtained == pytest.approx([3, 1, 1])

        assert summ.count_values(4, values) == pytest.approx([3, 1, 1, 0])

    @pytest.mark.parametrize("bad_value", [-1, 6, 10])
    @pytest.mark.parametrize("n", (2, 5))
    def test_out_of_range(self, n: int, bad_value: int) -> None:
        """Test with value which is not in the set {0, ..., n-1}."""
        values = [0, 1, bad_value]

        with pytest.raises(ValueError):
            summ.count_values(2, values)

    @pytest.mark.parametrize("n", [3, 5])
    def test_range(self, n: int) -> None:
        """Test for range (expected: ones)."""
        values = list(range(n))
        assert summ.count_values(n, values) == pytest.approx(np.ones_like(values))

    def test_non_integer(self) -> None:
        """Test for non-integer array: we are looking for a type error."""
        values = np.asarray([0.0, 1.0, 2.0, 0.0])

        with pytest.raises(TypeError):
            summ.count_values(3, values)


class TestCountValuesJoint:
    """Tests for `count_values_joint`."""

    def test_simple(self) -> None:
        """Tests for a simple example, with N = 2, K = 3."""
        expected = np.asarray(
            [
                [1, 0, 2],
                [0, 2, 1],
            ]
        )

        ns = [0, 0, 0, 1, 1, 1]
        ks = [0, 2, 2, 1, 1, 2]

        obtained = summ.count_values_joint(
            n=2,
            k=3,
            ns=ns,
            ks=ks,
        )

        assert obtained.shape == expected.shape
        assert np.issubdtype(obtained.dtype, np.integer)

        assert expected == pytest.approx(obtained)
