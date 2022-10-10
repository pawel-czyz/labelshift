"""
Fixed q = 0.85, π′1 = 0.7 and changed N' in range
{10, 50, 100, 500, 1000, 10000}
"""
algorithms = [
    "ClassifyAndCount",
    "RatioEstimator",
    "BlackBoxShiftEstimator",
    "BayesianMAP",
]


def main() -> None:
    n_labeled = 1000
    quality = 0.85
    n_seeds = 10

    pi_unlabeled = 0.7
    pi_labeled = 0.5

    for n_unlabeled in [10, 50, 100, 500, 1000, 10000]:
        for seed in range(n_seeds):
            for algorithm in algorithms:
                try:
                    output_dir = f"experiment1-2/{algorithm}"
                    command = f"python scripts/experiment_categorical.py --n-labeled {n_labeled} --n-unlabeled {n_unlabeled} --quality {quality} --pi-labeled {pi_labeled} --pi-unlabeled {pi_unlabeled} --seed {seed} --algorithm {algorithm} --output-dir {output_dir}"
                    print(command)
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    main()
