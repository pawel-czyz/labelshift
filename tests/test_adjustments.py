"""Tests for adjustment submodule."""
import numpy.testing as nptest

import labelshift as ls


# TODO(pawel-czyz): These tests can be refactored with parametrization fixture.
def test_label_hardening() -> None:
    """Test if soft-labels are properly converted
    into one-hot vectors."""

    softlabels = [
        [0.1, 0.9],
        [0.8, 0.2],
        [0.3, 0.7],
    ]

    onehot = [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    onehot1 = ls.label_hardening(softlabels)
    nptest.assert_allclose(onehot1, onehot)


def test_label_hardening_square() -> None:
    """Test if soft-labels are properly converted
    into one-hot vectors. Square matrix."""

    softlabels = [
        [0.1, 0.9],
        [0.2, 0.8],
    ]

    onehot = [
        [0.0, 1.0],
        [0.0, 1.0],
    ]

    onehot1 = ls.label_hardening(softlabels)
    nptest.assert_allclose(onehot1, onehot)
