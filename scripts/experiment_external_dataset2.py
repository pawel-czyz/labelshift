import enum

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import labelshift.datasets.split as split
import labelshift.summary_statistic as summ

import labelshift.algorithms.api as algos
import labelshift.algorithms.ratio_estimator as re
import labelshift.algorithms.bayesian_discrete as bay
from labelshift.algorithms.expectation_maximization import expectation_maximization

plt.rcParams.update({"font.size": 14})


class Algorithm(enum.Enum):
    EM = "EM"
    CC = "CC"
    BBSE_HARD = "BBSE"
    RATIO_HARD = "IR: hard"
    RATIO_SOFT = "IR: soft"


def get_estimate(
    algorithm: Algorithm,
    n_y_c_labeled: np.ndarray,
    n_c_unlabeled: np.ndarray,
    y_labeled: np.ndarray,
    prob_c_labeled: np.ndarray,
    prob_c_unlabeled: np.ndarray,
    labeled_prevalence: np.ndarray,
) -> np.ndarray:
    """Function running the (point) prevalence estimator.

    Args:
        algorithm: estimator
        n_y_c_labeled: matrix with counts of predictions and true values, shape (L, K)
        n_c_unlabeled: vector with prediction counts on unlabeled data set, shape (K,)
        y_labeled: true labels in the labeled data set, shape (N,)
        prob_c_labeled: predictions of the classifier on the labeled data set, shape (N, K)
        prob_c_unlabeled: predictions of the classifier on the unlabeled data set, shape (N', K)
        labeled_prevalence: prevalence vector on the labeled distribution, shape (L,)
    """
    summary_statistic = algos.SummaryStatistic(
        n_y_labeled=None, n_y_and_c_labeled=n_y_c_labeled, n_c_unlabeled=n_c_unlabeled
    )

    if algorithm == Algorithm.EM:
        return expectation_maximization(
            predictions=prob_c_unlabeled, training_prevalences=labeled_prevalence
        )
    elif algorithm == Algorithm.CC:
        return algos.ClassifyAndCount().estimate_from_summary_statistic(
            summary_statistic
        )
    elif algorithm == Algorithm.BBSE_HARD:
        return algos.BlackBoxShiftEstimator(
            p_y_labeled=labeled_prevalence
        ).estimate_from_summary_statistic(summary_statistic)
    elif algorithm == Algorithm.RATIO_HARD:
        return algos.InvariantRatioEstimator(
            restricted=True
        ).estimate_from_summary_statistic(summary_statistic)
    elif algorithm == Algorithm.RATIO_SOFT:
        return re.calculate_vector_and_matrix_from_predictions(
            unlabeled_predictions=prob_c_unlabeled,
            labeled_predictions=prob_c_labeled,
            labeled_ground_truth=y_labeled,
        )
    else:
        raise ValueError(f"Algorithm {algorithm} not recognized.")


def main() -> None:
    L = 2
    K = L
    dataset = sklearn.datasets.load_breast_cancer()
    print(len(dataset.target))

    ymax: float = 7.0
    random_seed: int = 20
    n_training_examples: int = 200
    n_labeled_examples: int = 100
    n_unlabeled_examples: int = 150
    prevalence_labeled: np.ndarray = np.ones(2) / 2
    prevalence_unlabeled: np.ndarray = np.asarray([0.3, 0.7])

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

    # classifier = DecisionTreeClassifier(random_state=random_seed + 1)
    classifier = RandomForestClassifier(random_state=random_seed + 1)
    # classifier = LogisticRegression(random_state=random_seed + 1)
    classifier.fit(datasets.train_x, datasets.train_y)

    # The count values
    n_y_c_labeled = summ.count_values_joint(
        L, K, datasets.valid_y, classifier.predict(datasets.valid_x)
    )
    n_c_unlabeled = summ.count_values(K, classifier.predict(datasets.test_x))

    labeled_probabilities = classifier.predict_proba(datasets.valid_x)
    unlabeled_probabilities = classifier.predict_proba(datasets.test_x)

    with bay.build_model(
        n_y_and_c_labeled=n_y_c_labeled,
        n_c_unlabeled=n_c_unlabeled,
    ):
        idata = pm.sample()

    fig, ax = plt.subplots(figsize=(6, 4))
    _, ax_trash = plt.subplots()

    az.plot_posterior(idata, ax=[ax, ax_trash], var_names=bay.P_TEST_Y)
    ax.set_title(r"$\pi'_1$ posterior")

    ax.vlines(
        x=prevalence_unlabeled[0],
        ymin=0,
        ymax=ymax,
        label="Ground truth",
        colors=["k"],
        linestyles=["--"],
    )

    linestyles = [
        "dashdot",
        (0, (1, 1)),
        "solid",
        "dashed",
        (0, (3, 10, 1, 10)),
    ]

    for i, alg in enumerate(Algorithm):
        print(alg)
        estimate = get_estimate(
            algorithm=alg,
            n_y_c_labeled=n_y_c_labeled,
            n_c_unlabeled=n_c_unlabeled,
            y_labeled=datasets.valid_y,
            prob_c_labeled=labeled_probabilities,
            prob_c_unlabeled=unlabeled_probabilities,
            labeled_prevalence=prevalence_labeled,
        )

        ax.vlines(
            estimate[0],
            ymin=0,
            ymax=ymax,
            label=alg.value,
            colors=[f"C{i+2}"],
            linestyles=[linestyles[i]],
        )

        print(estimate)

    fig.legend()
    fig.tight_layout()
    fig.savefig("plot_cancer.pdf")


if __name__ == "__main__":
    main()
