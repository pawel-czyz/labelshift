"""This experiment plots the posterior in the Gaussian mixture model as well
as a discretized version of that.
"""
import string
from typing import List

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns

import labelshift.partition as part
import labelshift.summary_statistic as summ
import labelshift.algorithms.bayesian_discrete as discrete


plt.rcParams.update({"font.size": 22})


def plot_distributions(
    ax: plt.Axes,
    X: np.ndarray,
    X1: np.ndarray,
    breakpoints: np.ndarray,
    height: float = 1.0,
) -> None:
    """

    Args:
        ax: axes where to draw the plot
        X: points from the labeled distribution, shape (n_labeled,)
        X1: points from the unlabeled distribution, shape (n_unlabeled,)
        breakpoints: breakpoints to be plotted, shape (n_breakpoints,)
    """
    sns.kdeplot(data=np.hstack(X), ax=ax)
    sns.kdeplot(data=np.hstack(X1), ax=ax)

    for bp in breakpoints:
        ax.axvline(bp, ymax=height, linestyle="--", c="k", alpha=0.5)


def gaussian_model(
    labeled_data: List[np.ndarray], unlabeled_data: np.ndarray
) -> pm.Model:
    """
    Args:
        labeled_data: list of samples attributed to each Y:
            [
              [a1, ..., a_n0],
              [b1, ..., b_n1]
            ]
        unlabeled_data: array of shape (n_unlabeled,)
    """
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=1, shape=2)

        for i in range(2):
            pm.Normal(
                f"X_labeled{i}", mu=mu[i], sigma=sigma[i], observed=labeled_data[i]
            )

        weights = pm.Dirichlet("P_unlabeled(Y)", np.ones(2))

        pm.NormalMixture(
            "X_unlabeled", w=weights, mu=mu, sigma=sigma, observed=unlabeled_data
        )

    return model


def main() -> None:
    """The main method."""
    mus = [0.0, 1.0]
    sigmas = [0.3, 0.4]
    ns = [500, 500]
    ns_ = [200, 800]
    K = 7
    L = 2

    partition = part.RealLinePartition(np.linspace(-0.5, 1.5, K - 1))
    print(partition.breakpoints)

    assert len(partition) == K

    rng = np.random.default_rng(42)

    X_stratified = [
        rng.normal(loc=mu, scale=sigma, size=n) for mu, sigma, n in zip(mus, sigmas, ns)
    ]
    X = np.hstack(X_stratified)
    Y = np.hstack([[i] * n for i, n in enumerate(ns)])

    C = partition.predict(X)

    X1_stratified = [
        rng.normal(loc=mu, scale=sigma, size=n_)
        for mu, sigma, n_ in zip(mus, sigmas, ns_)
    ]
    X1 = np.hstack(X1_stratified)
    C1 = partition.predict(X1)

    n_c_unlabeled = summ.count_values(K, C1)
    n_y_c_labeled = summ.count_values_joint(L, K, Y, C)

    print(n_c_unlabeled)
    print(n_y_c_labeled)

    fig, axs = plt.subplots(3, figsize=(6, 9))
    plot_distributions(ax=axs[0], X=X, X1=X1, breakpoints=partition.breakpoints)

    with gaussian_model(labeled_data=X_stratified, unlabeled_data=X1):
        gaussian_data = pm.sample()

    _, ax_trash = plt.subplots()

    az.plot_posterior(gaussian_data, ax=[axs[1], ax_trash], var_names="P_unlabeled(Y)")
    axs[1].set_title(r"$\pi'_1$ (Gaussian)")

    with discrete.build_model(
        n_y_and_c_labeled=n_y_c_labeled, n_c_unlabeled=n_c_unlabeled
    ):
        discrete_data = pm.sample()

    az.plot_posterior(discrete_data, ax=[axs[2], ax_trash], var_names=discrete.P_TEST_Y)
    axs[2].set_title(r"$\pi'_1$ (Discrete)")

    for n, ax in enumerate(axs):
        ax.text(
            -0.1,
            1.1,
            string.ascii_uppercase[n],
            transform=ax.transAxes,
            size=20,
            weight="bold",
        )

    fig.tight_layout()
    fig.savefig("plot.pdf")


if __name__ == "__main__":
    main()
