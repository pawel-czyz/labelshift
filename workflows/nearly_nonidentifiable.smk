# ---------------------------------------------------
# - Experiment with a nearly non-identifiable model -
# ---------------------------------------------------
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

import matplotlib
matplotlib.use("agg")

import labelshift.datasets.discrete_categorical as dc
import labelshift.algorithms.api as algo


workdir: "generated/nearly_nonidentifiable"

N_BOOTSTRAPS: int = 100

N_SEEDS: int = 5
SEEDS: list[int] = list(range(1, N_SEEDS + 1))

# 1st class very different from 2 and 3, while 2 and 3 are almost exactly the same 
p_c_y = np.asarray([
    [0.96, 0.02, 0.02],
    [0.05, 0.5, 0.45],
    [0.05, 0.45, 0.5],
])
p_y_labeled = np.ones(3) / 3
p_y_unlabeled = np.asarray([0.5, 0.35, 0.15])


@dataclass
class Settings:
    n_labeled: int
    n_unlabeled: int

SETTINGS = {
    "small": Settings(n_labeled=100, n_unlabeled=100),
    "medium": Settings(n_labeled=1000, n_unlabeled=1000),
    "large": Settings(n_labeled=20_000, n_unlabeled=20_000),
}

BOOTSTRAP_ALGORITHMS = {
    "RIR": algo.InvariantRatioEstimator(restricted=True, enforce_square=True),
    "UIR": algo.InvariantRatioEstimator(restricted=False, enforce_square=True),
    "BBS": algo.BlackBoxShiftEstimator(enforce_square=True),
}

rule all:
    input: expand("figures/{setting}-{seed}.pdf", setting=SETTINGS.keys(), seed=SEEDS)


rule generate_data:
    output: "data/{setting}-{seed}.joblib"
    run:
        sampler = dc.discrete_sampler_factory(
            p_c_cond_y_labeled=p_c_y,
            p_y_labeled=p_y_labeled,
            p_y_unlabeled=p_y_unlabeled
        )
        seed = int(wildcards.seed)
        setting = SETTINGS[wildcards.setting]
        summary_statistic = sampler.sample_summary_statistic(
            n_labeled=setting.n_labeled,
            n_unlabeled=setting.n_unlabeled,
            seed=seed,
        )
        joblib.dump(summary_statistic, str(output))


rule run_mcmc:
    input: "data/{setting}-{seed}.joblib"
    output: "samples/MCMC/{setting}-{seed}.npy"
    run:
        data = joblib.load(str(input))
        estimator = algo.DiscreteCategoricalMeanEstimator()
        samples = np.asarray(estimator.sample_posterior(data)[estimator.P_TEST_Y])
        np.save(str(output), samples)


def _bootstrap(rng, stat: dc.SummaryStatistic) -> dc.SummaryStatistic:
    n_unlabeled = np.sum(stat.n_c_unlabeled)
    n_c_unlabeled = rng.multinomial(n=n_unlabeled, pvals=stat.n_c_unlabeled / n_unlabeled)

    p_c_y = stat.n_y_and_c_labeled / np.sum(stat.n_y_and_c_labeled, axis=1, keepdims=True)

    n_y_and_c_labeled = rng.multinomial(n=stat.n_y_labeled, pvals=p_c_y)

    return dc.SummaryStatistic(
        n_y_labeled=stat.n_y_labeled,
        n_y_and_c_labeled=n_y_and_c_labeled,
        n_c_unlabeled=n_c_unlabeled,
    )


rule run_bootstrap:
    input: "data/{setting}-{seed}.joblib"
    output: "samples/bootstrap-{algorithm}/{setting}-{seed}.npy"
    run:
        data = joblib.load(str(input))
        estimator = BOOTSTRAP_ALGORITHMS[wildcards.algorithm]
        rng = np.random.default_rng(int(wildcards.seed) + 10)

        samples = []

        while len(samples) < N_BOOTSTRAPS:
            boot = _bootstrap(rng, data)
            try:
                estimate = estimator.estimate_from_summary_statistic(boot)
                samples.append(estimate)
            except Exception:
                continue

        samples = np.asarray(samples)
        np.save(str(output), samples)


def plot(ax, samples, color="darkblue"):
    for sample in samples[-N_BOOTSTRAPS:]:
        ax.plot(sample, alpha=0.05, color=color, rasterized=True)
        ax.scatter(np.arange(3), sample, color=color, s=2, alpha=0.05, rasterized=True)

    ax.set_ylim(-0.001, 1.01)
    ax.set_xlim(-0.2, 2.2)
    ax.set_xticks(np.arange(3), np.arange(1, 4))
    ax.set_xlabel("Class")
    ax.spines[['right', 'top']].set_visible(False)
    ax.plot(p_y_unlabeled, color="black", linestyle="--")
    ax.scatter(np.arange(3), p_y_unlabeled, color='black', s=5)


rule plot:
    input:
        mcmc = "samples/MCMC/{setting}-{seed}.npy",
        rir = "samples/bootstrap-RIR/{setting}-{seed}.npy",
        uir = "samples/bootstrap-UIR/{setting}-{seed}.npy",
        bbs = "samples/bootstrap-BBS/{setting}-{seed}.npy"
    output: "figures/{setting}-{seed}.pdf"
    run:
        fig, axs = plt.subplots(1, 4, figsize=(8, 2), dpi=130, sharex=True, sharey=True)
        
        ax = axs[0]
        ax.set_ylabel("Prevalence")
        ax.set_title("Posterior samples")
        samples = np.load(str(input.mcmc))
        plot(ax, samples)

        ax = axs[1]
        ax.set_title("RIR (bootstrapped)")
        samples = np.load(str(input.rir))
        plot(ax, samples)

        ax = axs[2]
        ax.set_title("UIR (bootstrapped)")
        samples = np.load(str(input.uir))
        plot(ax, samples)

        ax = axs[3]
        ax.set_title("BBS (bootstrapped)")
        samples = np.load(str(input.bbs))
        plot(ax, samples)

        fig.tight_layout()
        fig.savefig(str(output))
