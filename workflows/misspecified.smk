# ----------------------------------------------------------------------------------
# - Experiment with fitting a misspecified Gaussian mixture to the Student mixture -
# ----------------------------------------------------------------------------------
from dataclasses import dataclass
import numpy as np
import json
import joblib

import matplotlib
import matplotlib.pyplot as plt
from subplots_from_axsize import subplots_from_axsize
matplotlib.use("agg")

import jax
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary

import labelshift.algorithms.api as algo
from labelshift.datasets.discrete_categorical import SummaryStatistic


workdir: "generated/misspecified"

N_SEEDS: int = 200

SEEDS: list[int] = list(range(1, N_SEEDS + 1))


@dataclass
class Data:
    x0: np.ndarray
    x1: np.ndarray

    @property
    def xs(self):
        return np.concatenate([self.x0, self.x1])


def sample(rng, pi: float, N: int):
    """Samples a data set from a Student mixture
    with P(Y=1) = pi and P(Y=0) = 1 - pi.
    """
    assert 0 <= pi <= 1

    n1 = rng.binomial(N, pi)
    n0 = N - n1

    x0 = 0.5 * rng.standard_t(3, size=n0)
    x1 = 1 + 0.5 * rng.standard_t(4, size=n1)

    assert len(x0) + len(x1) == N

    return Data(x0=x0, x1=x1)

N_POINTS = [100, 1000, 10_000]
PI_LABELED = 0.5
PI_UNLABELED = 0.2

N_MCMC_WARMUP = 1500
N_MCMC_SAMPLES = 2000
N_MCMC_CHAINS = 4


COVERAGES = np.arange(0.05, 0.96, 0.05)


rule all:
    input:
        plots = expand("plots/{n_points}.pdf", n_points=N_POINTS),
        convergence = "convergence_overall.json",


rule generate_data:
    output: "data/{n_points}/{seed}.npy"
    run:
        rng = np.random.default_rng(int(wildcards.seed))
        data_labeled = sample(rng, pi=PI_LABELED, N=int(wildcards.n_points))
        data_unlabeled = sample(rng, pi=PI_UNLABELED, N=int(wildcards.n_points))
        joblib.dump((data_labeled, data_unlabeled), str(output))


def gaussian_model(observed: Data, unobserved: np.ndarray):
    sigma = numpyro.sample('sigma', dist.HalfCauchy(np.ones(2)))
    mu = numpyro.sample('mu', dist.Normal(np.zeros(2), 3))

    pi = numpyro.sample(algo.DiscreteCategoricalMeanEstimator.P_TEST_Y, dist.Dirichlet(np.ones(2)))

    mixture = dist.MixtureSameFamily(dist.Categorical(probs=pi), dist.Normal(mu, sigma))

    with numpyro.plate('N0', len(observed.x0)):
        numpyro.sample('x0', dist.Normal(mu[0], sigma[0]), obs=observed.x0)        

    with numpyro.plate('N1', len(observed.x1)):
        numpyro.sample('x1', dist.Normal(mu[1], sigma[1]), obs=observed.x1)        

    with numpyro.plate('N', len(unobserved)):
        numpyro.sample('x', mixture, obs=unobserved)


def student_model(observed: Data, unobserved: np.ndarray):
    df = numpyro.sample('df', dist.Gamma(np.ones(2), np.ones(2)))
    sigma = numpyro.sample('sigma', dist.HalfCauchy(np.ones(2)))
    mu = numpyro.sample('mu', dist.Normal(np.zeros(2), 3))

    pi = numpyro.sample(algo.DiscreteCategoricalMeanEstimator.P_TEST_Y, dist.Dirichlet(np.ones(2)))

    mixture = dist.MixtureSameFamily(dist.Categorical(probs=pi), dist.StudentT(df, mu, sigma))

    with numpyro.plate('N0', len(observed.x0)):
        numpyro.sample('x0', dist.StudentT(df[0], mu[0], sigma[0]), obs=observed.x0)        

    with numpyro.plate('N1', len(observed.x1)):
        numpyro.sample('x1', dist.StudentT(df[1], mu[1], sigma[1]), obs=observed.x1)        

    with numpyro.plate('N', len(unobserved)):
        numpyro.sample('x', mixture, obs=unobserved)


def generate_summary(samples):
    summ = summary(samples)
    n_eff_list = [float(np.min(d["n_eff"])) for d in summ.values()]
    r_hat_list = [float(np.max(d["r_hat"])) for d in summ.values()]
    return {"min_n_eff": min(n_eff_list), "max_r_hat": max(r_hat_list)}

rule run_gaussian_mcmc:
    input: "data/{n_points}/{seed}.npy"
    output:
        samples = "samples/{n_points}/Gaussian/{seed}.npy",
        convergence = "convergence/{n_points}/Gaussian/{seed}.joblib",
    run:    
        data_labeled, data_unlabeled = joblib.load(str(input))
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(gaussian_model),
            num_warmup=N_MCMC_WARMUP,
            num_samples=N_MCMC_SAMPLES,
            num_chains=N_MCMC_CHAINS,
        )
        rng_key = jax.random.PRNGKey(int(wildcards.seed) + 101)
        mcmc.run(rng_key, observed=data_labeled, unobserved=data_unlabeled.xs)
        samples = mcmc.get_samples()
        joblib.dump(samples, output.samples)

        summ = generate_summary(mcmc.get_samples(group_by_chain=True))
        joblib.dump(summ, output.convergence)


rule run_student_mcmc:
    input: "data/{n_points}/{seed}.npy"
    output:
        samples = "samples/{n_points}/Student/{seed}.npy",
        convergence = "convergence/{n_points}/Student/{seed}.joblib",
    run:
        data_labeled, data_unlabeled = joblib.load(str(input))
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(student_model),
            num_warmup=N_MCMC_WARMUP,
            num_samples=N_MCMC_SAMPLES,
            num_chains=N_MCMC_CHAINS,
        )
        rng_key = jax.random.PRNGKey(int(wildcards.seed) + 101)
        mcmc.run(rng_key, observed=data_labeled, unobserved=data_unlabeled.xs)
        samples = mcmc.get_samples()
        joblib.dump(samples, output.samples)

        summ = generate_summary(mcmc.get_samples(group_by_chain=True))
        joblib.dump(summ, output.convergence)



def _calculate_bins(n: int):
    return np.concatenate(([-np.inf], np.linspace(-4, 4, n - 1) , [np.inf]))


def binning(xs, n):
    bins = _calculate_bins(n)
    return np.histogram(xs, bins=bins)[0]


def generate_summary_statistic(
    observed: Data, unobserved: np.ndarray, n_bins: int,
) -> SummaryStatistic:
    return SummaryStatistic(
        n_y_labeled=np.asarray([len(observed.x0), len(observed.x1)]),
        n_y_and_c_labeled=np.vstack([binning(observed.x0, n=n_bins), binning(observed.x1, n=n_bins)]),
        n_c_unlabeled=binning(unobserved, n=n_bins),
    )

rule run_discrete_mcmc:
    input: "data/{n_points}/{seed}.npy"
    output:
        samples = "samples/{n_points}/Discrete-{n_bins}/{seed}.npy",
        convergence = "convergence/{n_points}/Discrete-{n_bins}/{seed}.joblib",
    run:
        data_labeled, data_unlabeled = joblib.load(str(input))
        estimator = algo.DiscreteCategoricalMeanEstimator(
            seed=int(wildcards.seed) + 101,
            params=algo.SamplingParams(
                warmup=N_MCMC_WARMUP,
                samples=N_MCMC_SAMPLES,
                chains=N_MCMC_CHAINS,
            ),
        )
        samples = estimator.sample_posterior(generate_summary_statistic(data_labeled, data_unlabeled.xs, int(wildcards.n_bins)))
        joblib.dump(samples, output.samples)

        summ = generate_summary(estimator.get_mcmc().get_samples(group_by_chain=True))
        joblib.dump(summ, output.convergence)


def calculate_hdi(arr, prob: float) -> tuple[float, float]:
    if prob <= 0 or prob >= 1:
        raise ValueError("prob should be between 0 and 1")
    n = len(arr)
    arr = np.sort(arr)

    # Range which contains approximately `prob` fraction of samples
    prob_range = int(np.floor(prob * n))
    # Check interval widths starting at different positions
    widths = arr[prob_range:] - arr[:n-prob_range]

    min_idx = np.argmin(widths)
    return arr[min_idx], arr[min_idx + prob_range]


rule contains_ground_truth:
    input:
        samples = "samples/{n_points}/{algorithm}/{seed}.npy",
        convergence = "convergence/{n_points}/{algorithm}/{seed}.joblib",
    output: "contains/{n_points}/{algorithm}/{seed}.joblib"
    run:
        samples = joblib.load(input.samples)
        convergence = joblib.load(input.convergence)
        run_ok = True if convergence["max_r_hat"] < 1.02 else False

        pi_samples = samples[algo.DiscreteCategoricalMeanEstimator.P_TEST_Y][:, 1]

        results = []
        intervals = []
        for coverage in COVERAGES:
            lower, upper = calculate_hdi(pi_samples, coverage)
            intervals.append([lower, upper])

            results.append(1 if lower <= PI_UNLABELED <= upper else 0)

        results = np.asarray(results, dtype=float)
        intervals = np.asarray(intervals, dtype=float)
        joblib.dump((results, intervals, run_ok), str(output))


def _input_paths_calculate_coverages(wildcards):
    return [f"contains/{wildcards.n_points}/{wildcards.algorithm}/{seed}.joblib" for seed in SEEDS]


rule calculate_coverages:
    input: _input_paths_calculate_coverages
    output:
        coverages = "coverages/{n_points}/{algorithm}.npy",
        excluded_runs = "excluded/{n_points}-{algorithm}.json"
    run:
        results = []

        ok_runs = 0
        excluded_runs = 0
        for pth in input:
            res, _, run_ok = joblib.load(pth)
            if run_ok:
                results.append(res)
                ok_runs += 1
            else:
                excluded_runs += 1

        results = np.asarray(results)
        coverages = results.mean(axis=0)
        np.save(output.coverages, coverages)

        with open(output.excluded_runs, "w") as fh:
            json.dump({"excluded_runs": excluded_runs, "ok_runs": ok_runs}, fh)

def _input_paths_summarize_convergence(wildcards):
    return [f"convergence/{wildcards.n_points}/{wildcards.algorithm}/{seed}.joblib" for seed in SEEDS]


rule summarize_convergence:
    input: _input_paths_summarize_convergence
    output: "convergence/{n_points}/{algorithm}.json"
    run:
        min_n_effs = []
        max_r_hats = []
        for pth in input:
            res = joblib.load(pth)
            min_n_effs.append(res["min_n_eff"])
            max_r_hats.append(res["max_r_hat"])

        with open(str(output), "w") as fh:
            json.dump({"min_n_eff": min(min_n_effs), "max_r_hat": max(max_r_hats)}, fh)


rule summarize_convergence_overall:
    input: expand("convergence/{n_points}/{algorithm}.json", n_points=N_POINTS, algorithm=["Gaussian", "Student", "Discrete-5", "Discrete-10"])
    output: "convergence_overall.json"
    run:
        min_n_effs = []
        max_r_hats = []
        for pth in input:
            with open(pth) as fh:
                res = json.load(fh)
            min_n_effs.append(res["min_n_eff"])
            max_r_hats.append(res["max_r_hat"])

        with open(str(output), "w") as fh:
            json.dump({"min_n_eff": min(min_n_effs), "max_r_hat": max(max_r_hats)}, fh)

rule plot_coverage:
    input:
        gaussian = "coverages/{n_points}/Gaussian.npy",
        student = "coverages/{n_points}/Student.npy",
        discrete5 = "coverages/{n_points}/Discrete-5.npy",
        discrete10 = "coverages/{n_points}/Discrete-10.npy",
        sample_gaussian = "samples/{n_points}/Gaussian/1.npy",
        sample_student = "samples/{n_points}/Student/1.npy",
        sample_discrete5 = "samples/{n_points}/Discrete-5/1.npy",
        sample_discrete10 = "samples/{n_points}/Discrete-10/1.npy",
    output: "plots/{n_points}.pdf"
    run:
        fig, axs = subplots_from_axsize(axsize=(2, 1), wspace=[0.2, 0.3, 0.6],  dpi=400, left=0.2, top=0.3, right=1.8)
        axs = axs.ravel()

        # Conditional distributions P(X|Y)
        ax = axs[0]
        rng = np.random.default_rng(42)
        labeled = sample(rng, pi=PI_LABELED, N=100_000)
        bins = np.linspace(-4, 4, 81)
        ax.hist(labeled.x0, bins=bins, color="orangered", density=True, rasterized=True, alpha=0.5, label="$Y=0$")
        ax.hist(labeled.x1, bins=bins, color="darkblue", density=True, rasterized=True, alpha=0.5, label="$Y=1$")

        ax.set_title("$P(X\\mid Y)$")
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel("$x$")
        ax.set_xlim(-4, 4)
        # ax.set_ylim(0, 1)
        ax.legend(frameon=False, bbox_to_anchor=(-0.03, 1.01), loc="upper left")

        # Observed distributions P(X)
        ax = axs[1]
        unlabeled = sample(rng, pi=PI_UNLABELED, N=100_000)
        ax.hist(labeled.xs, bins=bins, color="grey", density=True, rasterized=True, alpha=0.5, label="Train")
        ax.hist(unlabeled.xs, bins=bins, color="maroon", density=True, rasterized=True, alpha=0.5, label="Test")
        ax.set_title("$P_\\text{train}(X)$ and $P_\\text{test}(X)$")
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.set_xlim(-4, 4)
        ax.set_yticks([])
        # ax.set_ylim(0, 1)
        ax.set_xlabel("$x$")
        ax.legend(frameon=False, bbox_to_anchor=(0, 0.98), loc="upper left")

        # One posterior sample
        ax = axs[2]
        sample_gaussian = joblib.load(input["sample_gaussian"])[algo.DiscreteCategoricalMeanEstimator.P_TEST_Y][:, 1]
        sample_student = joblib.load(input["sample_student"])[algo.DiscreteCategoricalMeanEstimator.P_TEST_Y][:, 1]
        sample_discrete5 = joblib.load(input["sample_discrete5"])[algo.DiscreteCategoricalMeanEstimator.P_TEST_Y][:, 1]
        sample_discrete10 = joblib.load(input["sample_discrete10"])[algo.DiscreteCategoricalMeanEstimator.P_TEST_Y][:, 1]

        bins = np.linspace(0, 1, 41)
        ax.hist(sample_gaussian, bins=bins, color="C1", density=True, rasterized=True, histtype="step")
        ax.hist(sample_student, bins=bins, color="C2", density=True, rasterized=True, histtype="step")
        ax.hist(sample_discrete5, bins=bins, color="C3", density=True, rasterized=True, histtype="step")
        ax.hist(sample_discrete10, bins=bins, color="C4", density=True, rasterized=True, histtype="step")
        
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(-0.01, max(0.4, sample_gaussian.max(), sample_student.max(), sample_discrete5.max(), sample_discrete10.max()) + 0.01)

        ax.axvline(PI_UNLABELED, linestyle="--", c="black")
        ax.set_title("Posterior samples")
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel("$P_\\text{test}(Y=1)$")


        # The coverage plot
        ax = axs[3]

        ps = np.linspace(1e-3, 1 - 1e-3, 51)
        ax.plot(ps, ps, linestyle="--", c="black")
        se = np.sqrt(ps * (1-ps) / N_SEEDS)
        # ax.fill_between(ps, ps - 2 * se, ps + 2 * se, alpha=0.1, color="k", ec=None)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

        ax.set_title("Coverage")
        ax.set_xlabel("Expected")
        ax.set_ylabel("Observed")

        def plot_data(data, label, color):
            covs = np.load(data)
            ax.plot(COVERAGES, covs, c=color, label=label)
            ax.scatter(COVERAGES, covs, c=color, s=5)

        plot_data(input["gaussian"], "Gaussian", "C1")
        plot_data(input["student"], "Student", "C2")
        plot_data(input["discrete5"], "Discrete (5 bins)", "C3")
        plot_data(input["discrete10"], "Discrete (10 bins)", "C4")

        ax.spines[['right', 'top']].set_visible(False)
        ax.legend(frameon=False, bbox_to_anchor=(0.99, 0.98), loc="upper left")
        
        fig.savefig(str(output))
