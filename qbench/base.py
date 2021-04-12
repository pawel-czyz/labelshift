import abc


class BaseQuantificationAlgorithm(abc.ABC):
    def __init__(self, n_classes: int) -> None:
        self.n_classes: int = n_classes

    @abc.abstractmethod
    def fit(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """

        Args:
            predictions: one-hot encoded predictions of the classifier on the test data set. Shape (n_examples, n_classes).
            labels: true labels of the data. Shape (n_examples,).
        """
        pass

    @abc.abstractmethod:
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """

        Args:
            predictions: one-hot encoded predictions of the classifier on the data set with unknown label prevalences. Shape (n_examples, n_classes).

        Returns:
            prevalences, shape (n_classes,)
        """
        pass

