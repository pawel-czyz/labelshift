import json
import string
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


rename_dict = {
    "ClassifyAndCount": "CC",
    "RatioEstimator": "IR",
    "BlackBoxShiftEstimator": "BBSE",
    "BayesianMAP": "MAP",
}

hue_order = [
    "CC",
    "IR",
    "BBSE",
    "MAP",
]


def file_to_row(file):
    with open(file) as f:
        x = json.load(f)
    return {
        "Algorithm": rename_dict[x["algorithm"]],
        "true": x["true"][0],
        "estimated": x["estimated"][0],
        "quality": x["sampler"]["p_c_cond_y"][0][0],
        "n_labeled": x["sampler"]["n_labeled"],
        "n_unlabeled": x["sampler"]["n_unlabeled"],
    }


def experiment_directory_to_dataframe(experiment_directory) -> pd.DataFrame:
    files = list(
        Path(experiment_directory).rglob(
            "*.json",
        )
    )
    df = pd.DataFrame([file_to_row(f) for f in files])
    df["error"] = df["estimated"] - df["true"]
    return df


def main() -> None:
    fig, axs = plt.subplots(3, 1, figsize=(4, 12), sharey=False)

    experiment1 = "experiment1-1"
    df1 = experiment_directory_to_dataframe(experiment1)
    sns.boxplot(
        df1, x="true", y="error", hue="Algorithm", ax=axs[0], hue_order=hue_order
    )
    axs[0].set_xlabel(r"Prevalence $\pi'_1$")
    axs[0].set_ylabel(r"Signed difference $\hat \pi'_1 - \pi'_1$")

    experiment2 = "experiment1-2"
    df2 = experiment_directory_to_dataframe(experiment2)
    sns.boxplot(
        df2, x="n_unlabeled", y="error", hue="Algorithm", ax=axs[1], hue_order=hue_order
    )

    axs[1].set_xlabel(r"Unlabeled data set size $N'$")
    axs[1].set_ylabel(r"Signed difference $\hat \pi'_1 - \pi'_1$")
    axs[1].legend([], [], frameon=False)

    experiment3 = "experiment1-3"
    df3 = experiment_directory_to_dataframe(experiment3)
    sns.boxplot(
        df3, x="quality", y="error", hue="Algorithm", ax=axs[2], hue_order=hue_order
    )

    axs[2].set_xlabel(r"Classifier quality $q$")
    axs[2].set_ylabel(r"Signed difference $\hat \pi'_1 - \pi'_1$")
    axs[2].legend([], [], frameon=False)

    for n, ax in enumerate(axs):
        ax.text(
            -0.1,
            1.1,
            string.ascii_uppercase[n],
            transform=ax.transAxes,
            size=20,
            weight="bold",
        )

    sns.move_legend(axs[0], "lower left")  # , bbox_to_anchor=(1, 1))

    fig.tight_layout()
    fig.savefig("experiment1.pdf")


if __name__ == "__main__":
    main()
