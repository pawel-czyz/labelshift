"""Fixtures for pytests."""
import numpy
import pytest
import random


@pytest.fixture
def set_random():
    """Fixture fixing the random seeds."""
    random.seed(0)
    numpy.random.seed(0)
