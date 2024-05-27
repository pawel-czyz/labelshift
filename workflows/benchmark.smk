# --------------------------------------------------------------------------------------------------
# ---  Benchmark of point quantification estimators employing black-box categorical classifiers  ---
# --------------------------------------------------------------------------------------------------
from dataclasses import dataclass
import joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
matplotlib.use("Agg")

import numpy as np


import labelshift.algorithms.api as algo
import labelshift.experiments.api as exp
import labelshift.datasets.discrete_categorical as dc

workdir: "generated/benchmark"

ESTIMATORS = {
    "BBS": algo.BlackBoxShiftEstimator(),
    "CC": algo.ClassifyAndCount(),
    "RIR": algo.InvariantRatioEstimator(restricted=True),
    "BAY": algo.DiscreteCategoricalMeanEstimator(),
}
ESTIMATOR_COLORS = {
    "BBS": "orangered",
    "CC": "goldenrod",
    "RIR": "limegreen",
    "BAY": "mediumblue",
}
ESTIMATOR_NAMES = {key: key for key in ESTIMATORS}

assert set(ESTIMATORS.keys()) == set(ESTIMATOR_COLORS.keys())
assert set(ESTIMATORS.keys()) == set(ESTIMATOR_NAMES.keys())


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


def _rmse(true, estimate):
    return np.sqrt(np.mean(np.square(true - estimate)))

def _mae(true, estimate):
    return np.abs(true - estimate).mean()

METRICS = {
    "RMSE": _rmse,
    "MAE": _mae,
}


N_SEEDS: int = 50

@dataclass
class BenchmarkSettings:
    param_name: str
    param_values: list

    settings: list[DataSetting]

_pi_unlabeled = [0.5, 0.6, 0.7, 0.8, 0.9]
_n_unlabeled = [10, 50, 100, 500, 1000, 10_000]
_k_vals = [2, 3, 5, 7, 9]
_quality = [0.55, 0.65, 0.75, 0.85, 0.95]
_quality_prime = [0.45, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95]

BENCHMARKS = {
    "change_prevalence": BenchmarkSettings(
        param_name="Prevalence $\\pi'_1$",
        param_values=_pi_unlabeled,
        settings=[generate_data_setting(prevalence_unlabeled=value) for value in _pi_unlabeled],
    ),
    "change_n_unlabeled": BenchmarkSettings(
        param_name="Unlabeled sample size $N'$",
        param_values=_n_unlabeled,
        settings=[generate_data_setting(n_unlabeled=value) for value in _n_unlabeled],
    ),
    "change_jointly_lk": BenchmarkSettings(
        param_name="Number of labels $L=K$",
        param_values=_k_vals,
        settings=[generate_data_setting(L=value, K=value) for value in _k_vals],
    ),
    "change_k": BenchmarkSettings(
        param_name="Classifier outputs $K$",
        param_values=_k_vals,
        settings=[generate_data_setting(K=value) for value in _k_vals],
    ),
    "change_quality": BenchmarkSettings(
        param_name="Classifier quality $q$",
        param_values=_quality,
        settings=[generate_data_setting(quality=value) for value in _quality],
    ),
    "change_misspecified": BenchmarkSettings(
        param_name="Misspecified quality $q'$",
        param_values=_quality_prime,
        settings=[generate_data_setting(quality_unlabeled=value) for value in _quality_prime],
    ),
}

def get_data_setting(benchmark: str, param: int | str) -> DataSetting:
    return BENCHMARKS[str(benchmark)].settings[int(param)]

rule all:
    input:
        individual_plots = expand("plots/benchmark-{benchmark}-metric-{metric}.pdf", benchmark=BENCHMARKS.keys(), metric=METRICS.keys()),
        figures = expand("plots/summary-{metric}.pdf", metric=METRICS.keys())


rule generate_data:
    output: "data/benchmark-{benchmark}/param-{param}/{seed}.joblib"
    run:
        data_setting = get_data_setting(
            benchmark=wildcards.benchmark,
            param=wildcards.param,
        )
        sampler = data_setting.create_sampler()
        
        summary_statistic = sampler.sample_summary_statistic(
            n_labeled=data_setting.n_labeled,
            n_unlabeled=data_setting.n_unlabeled,
            seed=int(wildcards.seed),
        )
        joblib.dump(summary_statistic, str(output))


@dataclass
class RunResult:
    p_y_unlabeled_true: np.ndarray
    p_y_unlabeled_estimate: np.ndarray
    param_value: float | int | str
    time: float
    algorithm: str
    run_ok: bool
    additional_info: dict


rule apply_estimator:
    input: "data/benchmark-{benchmark}/param-{param}/{seed}.joblib"
    output: "run_results/benchmark-{benchmark}/algorithm-{alg}/param-{param}/{seed}.joblib"
    run:
        data = joblib.load(str(input))
        estimator = ESTIMATORS[wildcards.alg]
        param_value = BENCHMARKS[wildcards.benchmark].param_values[int(wildcards.param)]

        p_y_unlabeled_true = get_data_setting(benchmark=wildcards.benchmark, param=wildcards.param).p_y_unlabeled

        try:
            timer = exp.Timer()
            estimate = estimator.estimate_from_summary_statistic(data)        
            if estimate.shape != p_y_unlabeled_true.shape:
                raise ValueError(f"For algorithm {wildcards.alg}, the estimate has shape {estimate.shape} but the true value has shape {p_y_unlabeled_true.shape}") 
            elapsed_time = timer.check()
            run_ok = True
            additional_info = {}
        except Exception as e:
            elapsed_time = float("nan")
            estimate = np.full_like(data.n_y_labeled, fill_value=float("nan"))
            run_ok = False
            additional_info = {"error": str(e)}
        
        result = RunResult(
            algorithm=str(wildcards.alg),
            time=elapsed_time,
            run_ok=run_ok,
            additional_info=additional_info,
            p_y_unlabeled_true=p_y_unlabeled_true,
            p_y_unlabeled_estimate=estimate,
            param_value=param_value,
        )

        joblib.dump(result, str(output))


def _get_paths_to_be_assembled(wildcards):
    benchmark = BENCHMARKS[wildcards.benchmark]
    params = list(range(len(benchmark.param_values)))
    return [
        f"run_results/benchmark-{wildcards.benchmark}/algorithm-{alg}/param-{param}/{seed}.joblib"
        for alg in ESTIMATORS
        for param in params
        for seed in range(N_SEEDS)
    ]


rule assemble_results:
    output:
        csv = "results/benchmark-{benchmark}-metric-{metric}.csv",
        err = "results/status/benchmark-{benchmark}-metric-{metric}.txt"
    input: _get_paths_to_be_assembled
    run:
        results = []
        for pth in input:
            res = joblib.load(pth)

            metric = METRICS[wildcards.metric](res.p_y_unlabeled_true, res.p_y_unlabeled_estimate)
            nice = {
                "param_value": res.param_value,
                "time": res.time,
                "algorithm": res.algorithm,
                "run_ok": res.run_ok,
                "additional_info": str(res.additional_info),
                "metric": metric,
            }
            results.append(nice)
        
        results = pd.DataFrame(results)

        df_ok = results[results["run_ok"]]

        with open(output.err, "w") as f:
            if len(results) != len(df_ok):
                f.write(f"Failed runs: {len(results) - len(df_ok)}\n")
            else:
                f.write("All runs successful\n")

        df_ok = df_ok.drop(columns=["run_ok", "additional_info"])
        df_ok.to_csv(str(output.csv), index=False)



def plot_results(ax, df, plot_std: bool = True, alpha: float = 0.5):
    for alg, df_alg in df.groupby("algorithm"):
        color = ESTIMATOR_COLORS[alg]
        
        data_mean = df_alg.groupby("param_value")["metric"].mean().reset_index()
        ax.plot(
            data_mean["param_value"],
            data_mean["metric"],
            color=color,
            label=ESTIMATOR_NAMES[alg],
        )
        ax.scatter(
            data_mean["param_value"],
            data_mean["metric"],
            color=color,
            s=5,
        )

        if plot_std:
            data_std = df_alg.groupby("param_value")["metric"].std().reset_index()
            ax.fill_between(
                data_std["param_value"],
                data_mean["metric"] - data_std["metric"],
                data_mean["metric"] + data_std["metric"],
                color=color,
                alpha=alpha,
                label=None,
                ec=None,
            )


rule plot_results_rule:
    output: "plots/benchmark-{benchmark}-metric-{metric}.pdf"
    input: "results/benchmark-{benchmark}-metric-{metric}.csv"
    run:
        fig, ax = plt.subplots(dpi=150, figsize=(4, 3))

        data = pd.read_csv(str(input), index_col=False)

        plot_results(ax, data)
        ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
        
        ax.set_xlabel(BENCHMARKS[wildcards.benchmark].param_name)
        ax.set_ylabel(str(wildcards.metric))

        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(str(output))

def label_ax(fig, ax, label):
    trans = mtransforms.ScaledTranslation(11/72, -1/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, f"{label}.", transform=ax.transAxes + trans, fontsize='medium', verticalalignment='top')


rule plot_large_plot:
    output: "plots/summary-{metric}.pdf"
    input:
        prevalence = "results/benchmark-change_prevalence-metric-{metric}.csv",
        n_unlabeled = "results/benchmark-change_n_unlabeled-metric-{metric}.csv",
        quality = "results/benchmark-change_quality-metric-{metric}.csv",
        k = "results/benchmark-change_k-metric-{metric}.csv",
        jointly_lk = "results/benchmark-change_jointly_lk-metric-{metric}.csv",
        misspecified = "results/benchmark-change_misspecified-metric-{metric}.csv"
    run:
        fig, axs = plt.subplots(2, 3, dpi=150, figsize=(7, 3), sharex=False, sharey=True)

        for ax in axs.ravel():
            ax.spines[["top", "right"]].set_visible(False)

        for i in range(2):
            axs[i, 0].set_ylabel(str(wildcards.metric))

        # Prevalence
        ax = axs[0, 0]

        data = pd.read_csv(input.prevalence, index_col=False)
        plot_results(ax, data)
        ax.set_xlabel("Prevalence $\\pi'_1$")
        label_ax(fig, ax, "a")

        # Unlabeled data set size
        ax = axs[0, 1]
        data = pd.read_csv(input.n_unlabeled, index_col=False)
        plot_results(ax, data)
        ax.set_xscale("log", base=10)
        ax.set_xlabel("Unlabeled sample size $N'$")
        label_ax(fig, ax, "b")


        # Classifier quality
        ax = axs[0, 2]
        data = pd.read_csv(input.quality, index_col=False)
        plot_results(ax, data)
        ax.set_xlabel("Classifier quality $q$")
        label_ax(fig, ax, "c")

        ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

        # Change K
        ax = axs[1, 0]
        data = pd.read_csv(input.k, index_col=False)
        plot_results(ax, data)
        ax.set_xlabel("Classifier outputs $K$")
        ax.set_xticks([3, 5, 7, 9])
        label_ax(fig, ax,  "d")

        # Change L = K
        ax = axs[1, 1]
        data = pd.read_csv(input.jointly_lk, index_col=False)
        plot_results(ax, data)
        ax.set_xlabel("Number of labels $L=K$")
        ax.set_xticks([3, 5, 7, 9])
        label_ax(fig, ax,  "e")

        # Change misspecification
        ax = axs[1, 2]
        data = pd.read_csv(input.misspecified, index_col=False)
        plot_results(ax, data)
        ax.set_xlabel("Misspecified quality $q'$")
        ax.axvline(0.85, color="black", linestyle="--")
        label_ax(fig, ax,  "f")

        fig.tight_layout()
        fig.savefig(str(output))
