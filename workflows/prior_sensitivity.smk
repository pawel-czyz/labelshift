# --------------------------------------------------------------------
# ---  Prior sensitivity check for binary quantification problems  ---
# --------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import json
from contextlib import redirect_stdout
from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd


import labelshift.algorithms.api as algo
import labelshift.experiments.api as exp
import labelshift.datasets.discrete_categorical as dc

workdir: "generated/prior_sensitivity"



@dataclass
class DataSetting:
    scalar_p_y_labeled: float
    scalar_p_y_unlabeled: float

    quality_labeled: float
    quality_unlabeled: float

    n_y: int
    n_c: int

    n_labeled: int
    n_unlabeled: int

    @property
    def p_y_labeled(self) -> np.ndarray:   
        return dc.almost_eye(self.n_y, self.n_y, diagonal=self.scalar_p_y_labeled)[0, :]
    
    @property
    def p_y_unlabeled(self) -> np.ndarray:
        return dc.almost_eye(self.n_y, self.n_y, diagonal=self.scalar_p_y_unlabeled)[0, :]

    @property
    def p_c_cond_y_labeled(self) -> np.ndarray:
        return dc.almost_eye(
            y=self.n_y,
            c=self.n_c,
            diagonal=self.quality_labeled,
        )
    
    @property
    def p_c_cond_y_unlabeled(self) -> np.ndarray:
        return dc.almost_eye(
        y=self.n_y,
        c=self.n_c,
        diagonal=self.quality_unlabeled,
    )

    def create_sampler(self) -> dc.DiscreteSampler:
        return dc.discrete_sampler_factory(
            p_y_labeled=self.p_y_labeled,
            p_y_unlabeled=self.p_y_unlabeled,
            p_c_cond_y_labeled=self.p_c_cond_y_labeled,
            p_c_cond_y_unlabeled=self.p_c_cond_y_unlabeled,
    )


def generate_data_setting(
    n_labeled: int = 1000,
    n_unlabeled: int = 500,
    quality: float = 0.85,
    quality_unlabeled: float | None = None,
    L: int = 5,
    K: int | None = None,
    prevalence_labeled: float | None = None,
    prevalence_unlabeled: float | None = 0.7,
) -> DataSetting:
    n_y = L
    n_c = exp.calculate_value(overwrite=K, default=n_y)

    quality_unlabeled = exp.calculate_value(
        overwrite=quality_unlabeled, default=quality
    )

    p_y_labeled = exp.calculate_value(
        overwrite=prevalence_labeled, default=1 / n_y
    )
    p_y_unlabeled = exp.calculate_value(
        overwrite=prevalence_unlabeled, default=1 / n_y
    )

    return DataSetting(
        scalar_p_y_labeled=p_y_labeled,
        scalar_p_y_unlabeled=p_y_unlabeled,
        quality_labeled=quality,
        quality_unlabeled=quality_unlabeled,
        n_y=n_y,
        n_c=n_c,
        n_labeled=n_labeled,
        n_unlabeled=n_unlabeled,
    )

ALPHA_SMALL = 0.1
ALPHA_MEDIUM = 1.0
ALPHA_LARGE = 10.0


MODELS = {
    str(ALPHA_SMALL): algo.DiscreteCategoricalMeanEstimator(params=algo.SamplingParams(chains=4), alpha=ALPHA_SMALL),
    str(ALPHA_MEDIUM): algo.DiscreteCategoricalMeanEstimator(params=algo.SamplingParams(chains=4), alpha=ALPHA_MEDIUM),
    str(ALPHA_LARGE): algo.DiscreteCategoricalMeanEstimator(params=algo.SamplingParams(chains=4), alpha=ALPHA_LARGE),
}
COLORS = {
    str(ALPHA_SMALL): "darkblue",
    str(ALPHA_MEDIUM): "purple",
    str(ALPHA_LARGE): "goldenrod",
}

N_SMALL = 50
N_MEDIUM = 500
N_LARGE = 5_000

DATA_SETTINGS = {
    str(N_SMALL): generate_data_setting(n_labeled=N_SMALL, n_unlabeled=N_SMALL, L=2, K=2),
    str(N_MEDIUM): generate_data_setting(n_labeled=N_MEDIUM, n_unlabeled=N_MEDIUM, L=2, K=2),
    str(N_LARGE): generate_data_setting(n_labeled=N_LARGE, n_unlabeled=N_LARGE, L=2, K=2),
}


def get_data_setting(data_setting: str) -> DataSetting:
    return DATA_SETTINGS[data_setting]


rule all:
    input: "prior_sensitivity.pdf", "convergence_stats.json"


rule plot:
    input:
        expand("posterior_samples/{data_setting}/model-{model}/1.joblib", data_setting=DATA_SETTINGS.keys(), model=MODELS.keys())
    output: "prior_sensitivity.pdf"
    run:
        data_sets = {}
        for path in input:
            samples = joblib.load(path)
            setting = samples["data_setting"]
            model = samples["model"]
            data_sets[(setting, model)] = samples[algo.DiscreteCategoricalMeanEstimator.P_TEST_Y][:, 0]

        fig, axs = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(6, 1.8), dpi=350)
        
        for ax in axs:
            ax.spines[["top", "left", "right"]].set_visible(False)
            ax.set_yticks([])
            ax.set_xlabel("$\\pi_1'$")
            ax.axvline(0.7, linestyle="--", linewidth=1, c="k", label="$\\pi_1^*$")

        bins = np.linspace(0, 1, 30)

        def plot_posterior(ax, samples, color, label=None):
            ax.hist(samples, bins=bins, histtype="step", color=color, alpha=0.8)
            ax.axvline(np.mean(samples), color=color, linewidth=1, label=label)

        for data_setting, ax in zip(DATA_SETTINGS.keys(), axs):
            ax.set_title(f"$N=N'={data_setting}$")

            for model in MODELS.keys():
                samples = data_sets[(data_setting, model)]
                plot_posterior(ax, samples, color=COLORS[model], label=f"$\\alpha={model}$")

        ax = axs.ravel()[-1]
        ax.legend(frameon=False, bbox_to_anchor=(1.05, 1.))

        fig.tight_layout()
        fig.savefig(str(output))

rule generate_data:
    output: "data/{data_setting}/{seed}.joblib"
    run:
        data_setting = get_data_setting(wildcards.data_setting)
        sampler = data_setting.create_sampler()
        
        summary_statistic = sampler.sample_summary_statistic(
            n_labeled=data_setting.n_labeled,
            n_unlabeled=data_setting.n_unlabeled,
            seed=int(wildcards.seed),
        )
        joblib.dump(summary_statistic, str(output))


rule apply_estimator:
    input: "data/{data_setting}/{seed}.joblib"
    output: 
        posterior_samples = "posterior_samples/{data_setting}/model-{model}/{seed}.joblib",
        convergence = "convergence/{data_setting}/model-{model}/{seed}.txt"
    run:
        data = joblib.load(str(input))
        estimator = MODELS[wildcards.model]

        posterior_samples = estimator.sample_posterior(data)
        posterior_samples["data_setting"] = wildcards.data_setting
        posterior_samples["model"] = wildcards.model

        joblib.dump(posterior_samples, filename=output.posterior_samples)

        with open(output.convergence, "w") as fh:
            with redirect_stdout(fh):
                estimator.get_mcmc().print_summary()


def parse_text_to_dataframe(file_path):
    # Read the entire file into a list of lines
    with open(file_path) as file:
        lines = file.readlines()

    # Find the start of the actual data (ignoring initial empty lines and headers)
    start_index = 0
    while not lines[start_index].strip():  # This finds the first non-empty line
        start_index += 1
    
    # We assume the table ends where non-table data starts again, typically after an empty line
    end_index = start_index
    while end_index < len(lines) and lines[end_index].strip():
        end_index += 1

    # Now extract only the relevant lines
    data_lines = lines[start_index:end_index]

    # Use pandas to read these lines, considering whitespace as a separator
    from io import StringIO
    data_str = '\n'.join(data_lines)
    dataframe = pd.read_csv(StringIO(data_str), sep=r'\s+', engine='python')

    return dataframe

rule parse_convergence_txt_to_csv:
    input: "convergence/{data_setting}/model-{model}/{seed}.txt"
    output: "convergence-csv/{data_setting}/model-{model}/{seed}.csv"
    run:
        parse_text_to_dataframe(str(input)).to_csv(str(output), index=False)

rule get_convergence_stats:
    input: expand("convergence-csv/{data_setting}/model-{model}/1.csv", data_setting=DATA_SETTINGS, model=MODELS)
    output: "convergence_stats.json"
    run:
        min_n_eff = 1e12
        max_r_hat = -100

        for pth in input:
            df = pd.read_csv(pth)
            min_n_eff = min(min_n_eff, df["n_eff"].values.min())
            max_r_hat = max(max_r_hat, df["r_hat"].values.max())

        with open(str(output), "w") as fp:
            json.dump(obj={"r_hat": max_r_hat, "n_eff": min_n_eff}, fp=fp)
