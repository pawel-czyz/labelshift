"""
Fixed π′1 = 0.7, N'= 500 and changed q in range
{0.55, 0.65, 0.75, 0.85, 0.95}
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
    n_seeds = 30

    pi_unlabeled = 0.7
    pi_labeled = 0.5

    for quality in [0.55, 0.65, 0.75, 0.85, 0.95]:
        for seed in range(n_seeds):
            for algorithm in algorithms:
                try:
                    output_dir = f"experiment1-3/{algorithm}"
                    command = f"python scripts/experiment_categorical.py --n-labeled {n_labeled} --n-unlabeled {n_unlabeled} --quality {quality} --pi-labeled {pi_labeled} --pi-unlabeled {pi_unlabeled} --seed {seed} --algorithm {algorithm} --output-dir {output_dir}"
                    print(command)
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    main()
