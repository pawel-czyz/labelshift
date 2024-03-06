"""
Fixed q = 0.85, N' = 500 and changed the prevalence π'1 in range {0.5, 0.6, 0.7, 0.8, 0.9}.
"""
algorithms = [
    "ClassifyAndCount",
    "RatioEstimator",
    "BlackBoxShiftEstimator",
    "BayesianMAP",
]


def main() -> None:
    n_labeled = 1000
    n_unlabeled = 500
    quality = 0.85
    n_seeds = 30

    pi_labeled = 0.5

    for pi_unlabeled in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for seed in range(n_seeds):
            for algorithm in algorithms:
                try:
                    output_dir = f"experiment1-1/{algorithm}"
                    command = f"python scripts/experiment_categorical.py --n-labeled {n_labeled} --n-unlabeled {n_unlabeled} --quality {quality} --pi-labeled {pi_labeled} --pi-unlabeled {pi_unlabeled} --seed {seed} --algorithm {algorithm} --output-dir {output_dir}"
                    print(command)
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    main()
