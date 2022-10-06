"""Experiment in which we directly sample from the P(C|Y) distribution."""
import argparse
import enum
from pathlib import Path
from typing import List

import pydantic

import labelshift.interfaces.point_estimators as pe
import labelshift.datasets.discrete_categorical as dc
import labelshift.algorithms.api as algo
from labelshift.timer import Timer


class Algorithm(enum.Enum):
    CLASSIFY_AND_COUNT = "ClassifyAndCount"
    RATIO_ESTIMATOR = "RatioEstimator"
    BBSE = "BlackBoxShiftEstimator"
    BAYESIAN = "BayesianMAP"
    EXPECTATION_MAXIMIZATION = "ExpectationMaximization"


class DiscreteSamplerConfig(pydantic.BaseModel):
    p_y_labeled: List[pydantic.NonNegativeFloat]
    p_y_unlabeled: List[pydantic.NonNegativeFloat]
    p_c_cond_y: List[List[pydantic.NonNegativeFloat]]

    n_labeled: pydantic.PositiveInt
    n_unlabeled: pydantic.PositiveInt

    random_seed: int = pydantic.Field(
        default=0, description="Random seed to be used " "to generate the data set."
    )


def get_estimator(
    algorithm: Algorithm, restricted: bool, alpha: float
) -> pe.SummaryStatisticPrevalenceEstimator:
    if algorithm == Algorithm.CLASSIFY_AND_COUNT:
        return algo.ClassifyAndCount()
    elif algorithm == Algorithm.RATIO_ESTIMATOR:
        return algo.InvariantRatioEstimator(restricted=restricted)
    elif algorithm == Algorithm.BBSE:
        return algo.BlackBoxShiftEstimator()
    elif algorithm == Algorithm.BAYESIAN:
        return algo.DiscreteCategoricalMAPEstimator(alpha_unlabeled=alpha)
    else:
        raise ValueError(f"Algorithm {algorithm} not recognized.")


def get_sufficient_statistic(config: DiscreteSamplerConfig) -> dc.SummaryStatistic:
    """Samples the sufficient statistic for the data set according to the specification."""
    sampler = dc.DiscreteSampler(
        p_y_labeled=config.p_y_labeled,
        p_y_unlabeled=config.p_y_unlabeled,
        p_c_cond_y=config.p_c_cond_y,
    )

    return sampler.sample_summary_statistic(
        n_labeled=config.n_labeled,
        n_unlabeled=config.n_unlabeled,
        seed=config.random_seed,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-labeled", type=int, default=1000, help="Number of labeled examples."
    )
    parser.add_argument(
        "--n-unlabeled", type=int, default=1000, help="Number of unlabeled examples."
    )
    parser.add_argument(
        "--quality", type=float, default=0.8, help="Quality of the classifier."
    )
    parser.add_argument(
        "--pi-labeled",
        type=float,
        default=0.5,
        help="Prevalence in the labeled data set.",
    )
    parser.add_argument(
        "--pi-unlabeled",
        type=float,
        default=0.5,
        help="Prevalence in the unlabeled data set.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed to sample the data."
    )
    parser.add_argument("--algorithm", type=Algorithm, default=Algorithm.BAYESIAN)
    parser.add_argument("--output", type=Path, default="output.json")

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--restricted", type=bool, default=False)

    return parser


class Result(pydantic.BaseModel):
    algorithm: Algorithm
    sampler: DiscreteSamplerConfig
    true: List[float]
    estimated: List[float]
    time: float


def main() -> None:
    """The main function of the experiment."""
    parser = create_parser()
    args = parser.parse_args()

    qual = args.quality
    assert 0 < qual < 1
    assert 0 < args.pi_labeled < 1
    assert 0 < args.pi_unlabeled < 1

    true_pi_unlabeled = [args.pi_unlabeled, 1 - args.pi_unlabeled]

    sampler_config = DiscreteSamplerConfig(
        p_y_labeled=[args.pi_labeled, 1 - args.pi_labeled],
        p_y_unlabeled=true_pi_unlabeled,
        p_c_cond_y=[
            [qual, 1 - qual],
            [1 - qual, qual],
        ],
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        random_seed=args.seed,
    )
    sufficient_statistic = get_sufficient_statistic(sampler_config)

    estimator = get_estimator(
        algorithm=args.algorithm, alpha=args.alpha, restricted=args.restricted
    )

    timer = Timer()

    estimate = estimator.estimate_from_summary_statistic(sufficient_statistic)
    elapsed_time = timer.check()

    result = Result(
        algorithm=args.algorithm,
        sampler=sampler_config,
        true=true_pi_unlabeled,
        estimated=estimate.tolist(),
        time=elapsed_time,
    )

    with open(args.output, "w") as f:
        f.write(result.json())

    print(result)
    print("Finished.")


if __name__ == "__main__":
    main()
