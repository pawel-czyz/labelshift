"""Sample data directly from P(C|Y) distribution and run specified quantification estimator."""
import argparse
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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-labeled", type=int, default=1_000, help="Number of labeled examples."
    )
    parser.add_argument(
        "--n-unlabeled", type=int, default=1_000, help="Number of unlabeled examples."
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
        "--prevalence-labeled",
        type=float,
        default=None,
        help="Prevalence of the first class in the labeled data set. Default: 1/L (uniform).",
    )
    parser.add_argument(
        "--prevalence-unlabeled",
        type=float,
        default=None,
        help="Prevalence of the first class in the unlabeled data set. Default: 1/L (uniform).",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed to sample the data."
    )
    parser.add_argument("--algorithm", type=Algorithm, default=Algorithm.BAYESIAN)
    parser.add_argument(
        "--output", type=Path, default=Path(f"{exp.generate_name()}.json")
    )
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument(
        "--bayesian-alpha",
        type=float,
        default=1.0,
        help="Dirichlet prior specification for the Bayesian quantification.",
    )
    parser.add_argument(
        "--restricted",
        type=bool,
        default=True,
        help="Whether to use restricted invariant ratio estimator.",
    )

    parser.add_argument(
        "--tag", type=str, default="", help="Can be used to tag the run."
    )

    parser.add_argument("--dry-run", action="store_true")

    return parser


class EstimatorArguments(pydantic.BaseModel):
    bayesian_alpha: float
    restricted: bool


class Arguments(pydantic.BaseModel):
    p_y_labeled: pydantic.confloat(gt=0, lt=1)
    p_y_unlabeled: pydantic.confloat(gt=0, lt=1)

    quality_labeled: pydantic.confloat(ge=0, le=1)
    quality_unlabeled: pydantic.confloat(ge=0, le=1)

    n_y: pydantic.PositiveInt = pydantic.Field(description="Number of labels, L.")
    n_c: pydantic.PositiveInt = pydantic.Field(description="Number of predictions, K.")

    n_labeled: pydantic.PositiveInt
    n_unlabeled: pydantic.PositiveInt

    seed: int

    algorithm: Algorithm
    tag: str
    estimator_arguments: EstimatorArguments


def parse_args(args) -> Arguments:
    n_y = args.L
    n_c = exp.calculate_value(overwrite=args.K, default=n_y)

    quality_unlabeled = exp.calculate_value(
        overwrite=args.quality_unlabeled, default=args.quality
    )

    p_y_labeled = exp.calculate_value(
        overwrite=args.prevalence_labeled, default=1 / n_y
    )
    p_y_unlabeled = exp.calculate_value(
        overwrite=args.prevalence_unlabeled, default=1 / n_y
    )

    return Arguments(
        p_y_labeled=p_y_labeled,
        p_y_unlabeled=p_y_unlabeled,
        quality_labeled=args.quality,
        quality_unlabeled=quality_unlabeled,
        n_y=n_y,
        n_c=n_c,
        seed=args.seed,
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        algorithm=args.algorithm,
        tag=args.tag,
        estimator_arguments=EstimatorArguments(
            bayesian_alpha=args.bayesian_alpha,
            restricted=args.restricted,
        ),
    )


def create_sampler(args: Arguments) -> dc.DiscreteSampler:
    L = args.n_y
    p_y_labeled = dc.almost_eye(L, L, diagonal=args.p_y_labeled)[0, :]
    p_y_unlabeled = dc.almost_eye(L, L, diagonal=args.p_y_unlabeled)[0, :]

    p_c_cond_y_labeled = dc.almost_eye(
        y=L,
        c=args.n_c,
        diagonal=args.quality_labeled,
    )
    p_c_cond_y_unlabeled = dc.almost_eye(
        y=L,
        c=args.n_c,
        diagonal=args.quality_unlabeled,
    )

    return dc.discrete_sampler_factory(
        p_y_labeled=p_y_labeled,
        p_y_unlabeled=p_y_unlabeled,
        p_c_cond_y_labeled=p_c_cond_y_labeled,
        p_c_cond_y_unlabeled=p_c_cond_y_unlabeled,
    )


def get_estimator(args: Arguments) -> pe.SummaryStatisticPrevalenceEstimator:
    if args.algorithm == Algorithm.CLASSIFY_AND_COUNT:
        if args.n_c != args.n_y:
            raise ValueError("For classify and count you need K = L.")
        return algo.ClassifyAndCount()
    elif args.algorithm == Algorithm.RATIO_ESTIMATOR:
        return algo.InvariantRatioEstimator(
            restricted=args.estimator_arguments.restricted, enforce_square=False
        )
    elif args.algorithm == Algorithm.BBSE:
        return algo.BlackBoxShiftEstimator(enforce_square=False)
    elif args.algorithm == Algorithm.BAYESIAN:
        return algo.DiscreteCategoricalMAPEstimator(
            alpha_unlabeled=args.estimator_arguments.bayesian_alpha
        )
    else:
        raise ValueError(f"Algorithm {args.algorithm} not recognized.")


class Result(pydantic.BaseModel):
    p_y_unlabeled_true: List[float]
    p_y_unlabeled_estimate: List[float]
    time: float
    algorithm: Algorithm

    input_arguments: Arguments


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

    sampler = create_sampler(args)

    summary_statistic = sampler.sample_summary_statistic(
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        seed=args.seed,
    )

    estimator = get_estimator(args)
    timer = exp.Timer()
    estimate = estimator.estimate_from_summary_statistic(summary_statistic)
    elapsed_time = timer.check()

    result = Result(
        algorithm=args.algorithm,
        time=elapsed_time,
        p_y_unlabeled_true=sampler.unlabeled.p_y.tolist(),
        p_y_unlabeled_estimate=estimate.tolist(),
        input_arguments=args,
    )

    if raw_args.output_dir is not None:
        raw_args.output_dir.mkdir(exist_ok=True, parents=True)
        output_path = raw_args.output_dir / raw_args.output
    else:
        output_path = raw_args.output

    with open(output_path, "w") as f:
        f.write(result.json())

    print(result)
    print("Finished.")


if __name__ == "__main__":
    main()
