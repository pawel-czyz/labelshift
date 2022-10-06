import numpy as np
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier

import labelshift.datasets.split as split


def main() -> None:
    dataset = sklearn.datasets.load_digits(n_class=3)

    random_seed: int = 20
    n_training_examples: int = 40
    n_labeled_examples: int = 30
    n_unlabeled_examples: int = 20
    prevalence_labeled: np.ndarray = np.ones(3) / 3
    prevalence_unlabeled: np.ndarray = np.asarray([0.1, 0.3, 0.6])

    specification = split.SplitSpecification(
        train=np.asarray(prevalence_labeled * n_training_examples, dtype=int).tolist(),
        valid=np.asarray(prevalence_labeled * n_labeled_examples, dtype=int).tolist(),
        test=np.asarray(
            prevalence_unlabeled * n_unlabeled_examples, dtype=int
        ).tolist(),
    )

    datasets = split.split_dataset(
        dataset=dataset, specification=specification, random_seed=random_seed
    )

    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(datasets.train_x, datasets.train_y)

    labeled_predictions = classifier.predict(datasets.valid_x)
    unlabeled_predictions = classifier.predict(datasets.test_x)

    print(labeled_predictions)
    print(unlabeled_predictions)


if __name__ == "__main__":
    main()
