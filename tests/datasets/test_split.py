"""Tests of `labelshift.datasets.split`."""

import numpy as np
import pytest
from sklearn import datasets

import labelshift.datasets.split as split


@pytest.mark.parametrize("seed_prevalence", range(5))
@pytest.mark.parametrize("n_classes", [2, 10])
def test_split_dataset(
    seed_prevalence: int, n_classes: int, seed_split: int = 0
) -> None:
    """
    Args:
        seed_prevalence: seed used to draw the prevalence vectors
    """
    rng = np.random.default_rng(seed_prevalence)
    dataset = datasets.load_digits(n_class=n_classes)

    spec = split.SplitSpecification(
        train=rng.choice(range(4, 20), size=n_classes).tolist(),
        valid=rng.choice(range(4, 20), size=n_classes).tolist(),
        test=rng.choice(range(4, 20), size=n_classes).tolist(),
    )

    split_dataset = split.split_dataset(
        dataset=dataset, specification=spec, random_seed=seed_split
    )

    assert len(split_dataset.train_x) == sum(spec.train)
    assert len(split_dataset.valid_x) == sum(spec.valid)
    assert len(split_dataset.test_x) == sum(spec.test)

    assert len(split_dataset.train_x) == len(split_dataset.train_y)
    assert len(split_dataset.valid_x) == len(split_dataset.valid_y)
    assert len(split_dataset.test_x) == len(split_dataset.test_y)
