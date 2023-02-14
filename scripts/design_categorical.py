"""Experimental design for the categorical experiment.

Use it to generate a list of commands to be run."""
from pathlib import Path
from typing import Optional

DIRECTORY = Path("data/generated/categorical_experiment")

ESTIMATOR_CONFIGURATIONS = {
    "MAP-1": "--algorithm MAP --bayesian-alpha 1",
    "MAP-2": "--algorithm MAP --bayesian-alpha 2",
    "CC": "--algorithm CC",
    "IR": "--algorithm IR --restricted true",
    "BBSE": "--algorithm BBSE",
}

N_SEEDS: int = 2

N_LABELED: int = 1_000
N_UNLABELED: int = 500
QUALITY_LABELED: float = 0.85
PI_UNLABELED: float = 0.7
L: int = 5
K: int = 5


def command(
    estimator_key: str,
    seed: int,
    output_dir: Path,
    n_y: int = L,
    n_c: int = K,
    n_labeled: int = N_LABELED,
    n_unlabeled: int = N_UNLABELED,
    quality_labeled: float = QUALITY_LABELED,
    quality_unlabeled: Optional[float] = None,
    pi_unlabeled: float = PI_UNLABELED,
) -> str:
    estimator_args = ESTIMATOR_CONFIGURATIONS[estimator_key]

    quality_unlabeled_str = (
        "" if quality_unlabeled is None else f"--quality-unlabeled {quality_unlabeled}"
    )

    print(
        f"python scripts/run_categorical.py "
        f"--n-labeled {n_labeled} --n-unlabeled {n_unlabeled} "
        f"--quality {quality_labeled} {quality_unlabeled_str} "
        f"--prevalence-unlabeled {pi_unlabeled} "
        f"--seed {seed} "
        f"--output-dir {output_dir} "
        f"--K {n_y} --L {n_c} "
        f"--tag {estimator_key} {estimator_args}"
    )


def experiment_change_prevalence() -> None:
    """Fix L = K = 5 and change pi'_1."""
    for seed in range(N_SEEDS):
        for pi_unlabeled in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for algorithm in ESTIMATOR_CONFIGURATIONS.keys():
                output_dir = (
                    DIRECTORY / "change_prevalence" / f"{algorithm}-{pi_unlabeled}"
                )
                command(
                    output_dir=output_dir,
                    pi_unlabeled=pi_unlabeled,
                    seed=seed,
                    estimator_key=algorithm,
                )


def experiment_change_n_unlabeled() -> None:
    """Change N'."""
    for seed in range(N_SEEDS):
        for n_unlabeled in [10, 50, 100, 500, 1000, 10000]:
            for algorithm in ESTIMATOR_CONFIGURATIONS.keys():
                output_dir = (
                    DIRECTORY / "change_n_unlabeled" / f"{algorithm}-{n_unlabeled}"
                )
                command(
                    n_unlabeled=n_unlabeled,
                    seed=seed,
                    estimator_key=algorithm,
                    output_dir=output_dir,
                )


def experiment_change_k() -> None:
    """Change K, keeping L fixed."""
    for seed in range(N_SEEDS):
        for n_c in [2, 3, 5, 7, 9]:
            for algorithm in ESTIMATOR_CONFIGURATIONS.keys():
                output_dir = DIRECTORY / "change_k" / f"{algorithm}-{n_c}"
                command(
                    seed=seed,
                    output_dir=output_dir,
                    estimator_key=algorithm,
                    n_c=n_c,
                )


def experiment_change_jointly_l_and_k() -> None:
    """Jointly change L = K."""
    for seed in range(N_SEEDS):
        for lk in [2, 3, 5, 7, 9, 10]:
            for algorithm in ESTIMATOR_CONFIGURATIONS.keys():
                output_dir = DIRECTORY / "change_jointly_lk" / f"{algorithm}-{lk}"
                command(
                    seed=seed,
                    estimator_key=algorithm,
                    output_dir=output_dir,
                    n_c=lk,
                    n_y=lk,
                )


def experiment_change_quality() -> None:
    """Change quality."""
    for seed in range(N_SEEDS):
        for quality in [0.55, 0.65, 0.75, 0.85, 0.95]:
            for algorithm in ESTIMATOR_CONFIGURATIONS.keys():
                output_dir = DIRECTORY / "change_quality" / f"{algorithm}-{quality}"
                command(
                    quality_labeled=quality,
                    seed=seed,
                    estimator_key=algorithm,
                    output_dir=output_dir,
                )


def experiment_misspecified() -> None:
    """Change quality in the unlabeled population, so that the model is misspecified."""
    for seed in range(N_SEEDS):
        for quality_prime in [0.45, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95]:
            for algorithm in ESTIMATOR_CONFIGURATIONS.keys():
                output_dir = DIRECTORY / "misspecified" / f"{algorithm}-{quality_prime}"
                command(
                    quality_unlabeled=quality_prime,
                    seed=seed,
                    output_dir=output_dir,
                    estimator_key=algorithm,
                )


def main() -> None:
    experiment_change_prevalence()
    experiment_change_n_unlabeled()
    experiment_change_quality()
    experiment_change_jointly_l_and_k()
    experiment_change_k()
    experiment_misspecified()


if __name__ == "__main__":
    main()
