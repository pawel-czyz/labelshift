"""Utilities for working with NumPy datasets."""
import dataclasses
from typing import List, Protocol

import numpy as np
import pydantic


class IDataset(Protocol):
    """Interface for the dataset used. Note that it's designed
    to be compatible with the scikit-learn's datasets."""

    @property
    def data(self) -> np.ndarray:
        """The covariates X. Shape (n_samples, ...)."""
        raise NotImplementedError

    @property
    def target(self) -> np.ndarray:
        """The target labels Y. Shape (n_samples,)."""
        raise NotImplementedError


def n_classes(dataset: IDataset) -> int:
    """Calculates the number of classes in the dataset."""
    return len(np.unique(dataset.target))


@dataclasses.dataclass
class SplitDataset:
    """This class represents training, validation, and test datasets."""

    train_x: np.ndarray
    train_y: np.ndarray

    valid_x: np.ndarray
    valid_y: np.ndarray

    test_x: np.ndarray
    test_y: np.ndarray


class SplitSpecification(pydantic.BaseModel):
    """Contains the specification about the class prevalences
    in each of training, validation, and test datasets.

    Each of the lists should be of length `n_classes`
    and at `y`th position the required number of instances
    in the given dataset should be given.
    """

    train: List[int]
    valid: List[int]
    test: List[int]


def _calculate_number_required(specification: SplitSpecification, label: int) -> int:
    """Calculates the required number of instances of label `label`
    according to `specification`."""
    return (
        specification.train[label]
        + specification.valid[label]
        + specification.test[label]
    )


def split_dataset(
    dataset: IDataset, specification: SplitSpecification, random_seed: int
) -> SplitDataset:
    """Splits `dataset` according to `specification`."""
    n_labels = n_classes(dataset)

    if set(np.unique(dataset.target)) != set(range(n_labels)):
        raise ValueError(
            f"Labels must be 0-indexed integers: {dataset.target_names} != "
            f"{set(range(n_labels))}."
        )
    if {
        len(specification.train),
        len(specification.valid),
        len(specification.train),
    } != {n_labels}:
        raise ValueError("Wrong length of the specification.")

    rng = np.random.default_rng(random_seed)

    train_indices: List[np.ndarray] = []
    valid_indices: List[np.ndarray] = []
    test_indices: List[np.ndarray] = []

    for label in range(n_labels):
        # Take the index of the points in `dataset` corresponding to the `label`
        # Shape (n_points_with_this_label,).
        index: np.ndarray = np.asarray(dataset.target == label).nonzero()[0]

        n_required = _calculate_number_required(
            specification=specification, label=label
        )

        if n_required > index.shape[0]:
            raise ValueError(
                f"According to specification one needs {n_required} data points "
                f"of label {label},"
                f"but only {index.shape[0]} are available."
            )

        # Shuffle index
        shuffled_index = rng.permutation(index)

        # Get the required data points from this data set
        # according to the permuted index
        n_train = specification.train[label]
        n_valid = specification.valid[label]
        n_test = specification.test[label]

        # Silence errors related to spacings around :
        index_train = shuffled_index[:n_train]
        index_valid = shuffled_index[n_train : n_train + n_valid]  # noqa: E203
        index_test = shuffled_index[
            n_train + n_valid : n_train + n_valid + n_test  # noqa: E203
        ]

        assert len(index_train) == n_train
        assert len(index_valid) == n_valid
        assert len(index_test) == n_test

        train_indices.append(index_train)
        valid_indices.append(index_valid)
        test_indices.append(index_test)

    train_index = rng.permutation(np.hstack(train_indices))
    valid_index = rng.permutation(np.hstack(valid_indices))
    test_index = rng.permutation(np.hstack(test_indices))

    return SplitDataset(
        train_x=dataset.data[train_index, ...],
        train_y=dataset.target[train_index],
        valid_x=dataset.data[valid_index, ...],
        valid_y=dataset.target[valid_index],
        test_x=dataset.data[test_index, ...],
        test_y=dataset.target[test_index],
    )
