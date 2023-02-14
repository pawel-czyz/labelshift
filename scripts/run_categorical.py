"""Sample data directly from P(C|Y) distribution and run specified quantification estimator."""
import argparse
import dataclasses
import enum
from pathlib import Path
from typing import List

import pydantic

import labelshift.interfaces.point_estimators as pe
import labelshift.datasets.discrete_categorical as dc
import labelshift.algorithms.api as algo
import labelshift.experiments.api as exp


class Algorithm(enum.Enum):
    CLASSIFY_AND_COUNT = "CC"
    RATIO_ESTIMATOR = "IR"
    BBSE = "BBSE"
    BAYESIAN = "MAP"


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


def get_estimator(
    algorithm: Algorithm, restricted: bool, alpha: float
) -> pe.SummaryStatisticPrevalenceEstimator:
    if algorithm == Algorithm.CLASSIFY_AND_COUNT:
        return algo.ClassifyAndCount()
    elif algorithm == Algorithm.RATIO_ESTIMATOR:
        return algo.InvariantRatioEstimator(restricted=restricted, enforce_square=False)
    elif algorithm == Algorithm.BBSE:
        return algo.BlackBoxShiftEstimator(enforce_square=False)
    elif algorithm == Algorithm.BAYESIAN:
        return algo.DiscreteCategoricalMAPEstimator(alpha_unlabeled=alpha)
    else:
        raise ValueError(f"Algorithm {algorithm} not recognized.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-labeled", type=int, default=1000, help="Number of labeled examples."
    )
    parser.add_argument(
        "--n-unlabeled", type=int, default=1000, help="Number of unlabeled examples."
    )
    parser.add_argument(
        "--quality",
        type=float,
        default=0.85,
        help="Quality of the classifier on the labeled data.",
    )
    parser.add_argument(
        "--quality-unlabeled",
        type=float,
        default=None,
        help="Quality of the classifier on the unlabeled data."
        "Can be used to assess model misspecification. "
        "If None, the quality will be the same for both labeled"
        "and unlabeled data set (no misspecification).",
    )
    parser.add_argument("--L", type=int, default=2, help="Number of classes L.")
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="Number of available predictions. Default: the same as L.",
    )
    parser.add_argument(
        "--pi-labeled",
        type=float,
        default=None,
        help="Prevalence of the first class in the labeled data set. Default: 1/L (uniform).",
    )
    parser.add_argument(
        "--pi-unlabeled",
        type=float,
        default=0.5,
        help="Prevalence of the first class in the unlabeled data set.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed to sample the data."
    )
    parser.add_argument("--algorithm", type=Algorithm, default=Algorithm.BAYESIAN)
    parser.add_argument(
        "--output", type=Path, default=Path(f"{exp.generate_name()}.json")
    )
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--restricted", type=bool, default=True)

    parser.add_argument("--dry-run", action="store_true")

    return parser


class Arguments(pydantic.BaseModel):
    output_path: Path
    pi_labeled: pydantic.confloat(gt=0, lt=1)
    pi_unlabeled: pydantic.confloat(gt=0, lt=1) = pydantic.Field(
        description="Prevalence of the "
    )
    n_labels: pydantic.PositiveInt = pydantic.Field(description="Number of labels, L.")
    n_predictions: pydantic.PositiveInt = pydantic.Field(
        description="Number of predictions, K."
    )


def parse_args(args) -> Arguments:
    raise NotImplementedError


class Result(pydantic.BaseModel):
    algorithm: Algorithm
    alpha: float
    true: List[float]
    estimated: List[float]
    time: float
    quality: float
    quality_prime: float


def dry_run(args: Arguments) -> None:
    print("-- Dry run --\nUsed settings:")
    print(args)
    print("Exiting...")


def main() -> None:
    """The main function of the experiment."""
    raw_args = create_parser().parse_args()
    args: Arguments = parse_args(raw_args)

    if raw_args.dry_run:
        dry_run(args)
        return

    L = args.L
    K = args.K

    qual = args.quality
    qual_prime = args.quality_prime

    assert 0 < qual < 1
    assert 0 < qual_prime < 1
    assert 0 < args.pi_unlabeled < 1

    pi_labeled = 1 / L if args.pi_labeled is None else args.pi_labeled

    true_pi_labeled = dc.almost_eye(L, L, diagonal=pi_labeled)[0, :]
    true_pi_unlabeled = dc.almost_eye(L, L, diagonal=args.pi_unlabeled)[0, :]
    quality_matrix = dc.almost_eye(
        y=L,
        c=K,
        diagonal=qual,
    )
    quality_prime_matrix = dc.almost_eye(
        y=L,
        c=K,
        diagonal=qual_prime,
    )

    sampler_config_labeled = DiscreteSamplerConfig(
        p_y_labeled=true_pi_labeled.tolist(),
        p_y_unlabeled=true_pi_unlabeled.tolist(),
        p_c_cond_y=quality_matrix.tolist(),
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        random_seed=args.seed,
    )
    sufficient_statistic_labeled = get_sufficient_statistic(sampler_config_labeled)

    sampler_config_unlabeled = DiscreteSamplerConfig(
        p_y_labeled=true_pi_labeled.tolist(),
        p_y_unlabeled=true_pi_unlabeled.tolist(),
        p_c_cond_y=quality_prime_matrix.tolist(),
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        random_seed=args.seed,
    )
    sufficient_statistic_unlabeled = get_sufficient_statistic(sampler_config_unlabeled)

    sufficient_statistic_misspecified = pe.SummaryStatistic(
        # Labeled data set has one summary statistic
        n_y_labeled=sufficient_statistic_labeled.n_y_labeled,
        n_y_and_c_labeled=sufficient_statistic_labeled.n_y_and_c_labeled,
        # And unlabeled one has another one...
        n_c_unlabeled=sufficient_statistic_unlabeled.n_c_unlabeled,
    )

    estimator = get_estimator(
        algorithm=args.algorithm, alpha=args.alpha, restricted=args.restricted
    )

    timer = exp.Timer()

    estimate = estimator.estimate_from_summary_statistic(
        sufficient_statistic_misspecified
    )
    elapsed_time = timer.check()

    result = Result(
        algorithm=args.algorithm,
        alpha=args.alpha,
        true=true_pi_unlabeled.tolist(),
        estimated=estimate.tolist(),
        time=elapsed_time,
        quality=qual,
        quality_prime=qual_prime,
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(exist_ok=True, parents=True)
        output_path = args.output_dir / args.output
    else:
        output_path = args.output

    with open(output_path, "w") as f:
        f.write(result.json())

    print(result)
    print("Finished.")


if __name__ == "__main__":
    main()
