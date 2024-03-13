# -------------------------------------------------------------------------------------------------------
# -------------              Quantification of single-cell RNA-seq data                  ----------------
# -------------------------------------------------------------------------------------------------------
#
# Note: To run this workflow, you need to download the data of Darmanis et al. (2017):
# from the Curated Cancer Cell Atlas: https://www.weizmann.ac.il/sites/3CA/brain
# Then, unzip them making sure that the directory structure (relative to the project root) is:
#
# .
# ├── data/Darmanis
# |   ├── Cells.csv
# |   ├── Genes.txt
# |   ├── Meta-data.csv
# |   └── normalized_Exp_data_TPM.mtx
# └── workflows
#     └── single_cell.smk
#
# You can verify the integrity of the files by using the checksum:
# $ md5sum data/Darmanis/*
# ed58a504d3cffa5f8ebf0ed81db97467  data/Darmanis/Cells.csv
# 8a687737e472fa23519c6f3de99d8994  data/Darmanis/Genes.txt
# a92f45068fb53affa0d909643619c3ac  data/Darmanis/Meta-data.csv
# d356301b5e70305347c9978084620c65  data/Darmanis/normalized_Exp_data_TPM.mtx
#
# Also, you need to install the following packages:
# $ pip install scanpy h5py
#
# Then, you can run the workflow as usual:
# $ snakemake -s workflows/single_cell.smk --cores 4
#
#
# Data acknowledgment:
# We use the data downloaded from the Curated Cancer Cell Atlas, which employs the data set from:
#
#   S. Darmanis, S.A. Sloan, D. Croote, et al., Single-Cell RNA-Seq Analysis of Infiltrating
#   Neoplastic Cells at the Migrating Front of Human Glioblastoma,
#   Cell Reports, Volume 21, Issue 5, 2017, Pages 1399-1410
#   ISSN 2211-1247, https://doi.org/10.1016/j.celrep.2017.10.030.
#
# -------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from subplots_from_axsize import subplots_from_axsize

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import labelshift.algorithms.api as algo
import labelshift.summary_statistic as summ
import labelshift.algorithms.ratio_estimator as re
from labelshift.datasets.discrete_categorical import SummaryStatistic
from labelshift.algorithms.expectation_maximization import expectation_maximization

workdir: "generated/single_cell"

rule all:
    input: "plots/p_x_y.pdf", "plots/manuscript_plot.pdf"


rule create_adata:
    input:
        mtx = "../../data/Darmanis/normalized_Exp_data_TPM.mtx",
        genes = "../../data/Darmanis/Genes.txt",
        cells = "../../data/Darmanis/Cells.csv"
    output:
        data = "data/full_data.h5ad",
        summary = "data/summary.txt"
    run:
        adata = sc.read_mtx(input.mtx).T
        adata.var_names = pd.read_csv(input.genes, header=None)[0].values
        adata.obs = pd.read_csv(input.cells)
        adata.obs["cell_type"] = adata.obs["cell_type"].replace({
            "Oligodendrocyte": "Oligo."
        })
        adata.write(output.data)

        with open(output.summary, "a") as f:
            for name, c in zip(*np.unique(adata.obs["cell_type"], return_counts=True)):
                f.write(f"{name}: {c}\n")


rule create_aggregated_cell_types:
    input: "data/full_data.h5ad"
    output: "data/aggregated_data.h5ad"
    run:
        adata = sc.read_h5ad(input[0])

        adata.obs["cell_type"] = adata.obs["cell_type"].replace({
            "Neuron": "Other",
            "Vascular": "Other",
        })
        adata.write(output[0])


def construct_encoders(adata) -> tuple[LabelEncoder, LabelEncoder]:
    """Creates:
      - cell type encoder
      - sample encoder
    """
    cell_type_encoder = LabelEncoder()
    cell_type_encoder.fit(adata.obs["cell_type"])

    sample_encoder = LabelEncoder()
    sample_encoder.fit(adata.obs["sample"])
    return cell_type_encoder, sample_encoder


def get_features(adata) -> np.ndarray:
    return np.log1p(adata.X.toarray())


def plot_cells(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    types: list | np.ndarray,
    encoder: LabelEncoder,
    seed: int = 123,
    cmap: str = "Set1",
    # Legend settings
    include_legend: bool = False,
    bbox_to_anchor: tuple[float, float] = (1, 1),
    fontsize: int = 8,
    ncol: int = 1,
):
    cmap = plt.cm.get_cmap(cmap)
    colors = {name: cmap(i / len(encoder.classes_)) for i, name in enumerate(encoder.classes_)}
    color_vec = np.array([colors[name] for name in types])

    assert len(xs) == len(ys) == len(color_vec)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(xs))

    ax.scatter(
        xs[permutation], ys[permutation],
        c=color_vec[permutation],
        s=3, alpha=0.5, marker=".",
        rasterized=True,
    )

    if include_legend:
        patches = [mpatches.Patch(color=color, label=name) for name, color in colors.items()]
        ax.legend(handles=patches, loc="upper left", bbox_to_anchor=bbox_to_anchor, frameon=False, fontsize=fontsize, ncol=ncol)


rule plot_p_x_y:
    input: "data/full_data.h5ad"
    output: "plots/p_x_y.pdf"
    run:
        adata = sc.read_h5ad(input[0])
        pca = PCA(n_components=4)
        pca.fit(get_features(adata))

        cell_type_encoder, sample_encoder = construct_encoders(adata)
        n_samples = len(sample_encoder.classes_)

        fig, axs = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 2 * 1.2), dpi=300)

        for ax in axs.ravel():
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        # Set scales for the axes
        _reps = pca.transform(get_features(adata))
        _offset = 3
        for ax in axs[0, :]:
            ax.set_xlim(np.min(_reps[:, 0]) - _offset, np.max(_reps[:, 0]) + _offset)
            ax.set_ylim(np.min(_reps[:, 1]) - _offset, np.max(_reps[:, 1]) + _offset)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
        for ax in axs[1, :]:
            ax.set_xlim(np.min(_reps[:, 2]) - _offset, np.max(_reps[:, 2]) + _offset)
            ax.set_ylim(np.min(_reps[:, 3]) - _offset, np.max(_reps[:, 3]) + _offset)
            ax.set_xlabel("PC 3")
            ax.set_ylabel("PC 4")
        del _reps

        # Plot the data
        for sample_name, ax1, ax2 in zip(sample_encoder.classes_, axs[0, :], axs[1, :]):
            specific_adata = adata[adata.obs["sample"] == sample_name]
            repres = pca.transform(get_features(specific_adata))

            ax1.set_title(sample_name)
            plot_cells(ax1, repres[:, 0], repres[:, 1], types=specific_adata.obs["cell_type"], encoder=cell_type_encoder)
            plot_cells(ax2, repres[:, 2], repres[:, 3], types=specific_adata.obs["cell_type"], encoder=cell_type_encoder)

        fig.tight_layout()
        fig.savefig(output[0])


rule manuscript_plot:
    input:
        full_data = "data/full_data.h5ad",
        proportions_full = "proportions/full.npz",
        proportions_agg = "proportions/aggregated.npz"
    output: "plots/manuscript_plot.pdf"
    run:
        adata = sc.read_h5ad(input[0])
        pca = PCA(n_components=2)
        pca.fit(get_features(adata))
        reps = pca.transform(get_features(adata))

        fig, axs = subplots_from_axsize(axsize=([1, 1, 1.5, 1.5], 1), dpi=300, top=0.3, left=0.1, wspace=[0.3, 0.7, 0.7], bottom=0.8, right=0.8)
        axs = axs.ravel()

        cell_type_encoder, sample_encoder = construct_encoders(adata)

        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)

        for ax in axs[:2]:
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(np.min(reps[:, 0]) - 1, np.max(reps[:, 0]) + 1)
            ax.set_ylim(np.min(reps[:, 1]) - 1, np.max(reps[:, 1]) + 1)

        axs[0].set_title("Samples")
        plot_cells(axs[0], reps[:, 0], reps[:, 1], types=adata.obs["sample"], encoder=sample_encoder, cmap="Pastel1")

        axs[1].set_title("Cell types")
        plot_cells(axs[1], reps[:, 0], reps[:, 1], types=adata.obs["cell_type"], encoder=cell_type_encoder, include_legend=True, bbox_to_anchor=(-1.3, -0.05), fontsize=6, ncol=3)

        ESTIMATOR_NAMES = {
            "bbs": "BBS",
            "rir": "RIR",
            "cc": "CC",
            "em": "EM",
            "rir_soft": "RIR (soft)",
        }
        ESTIMATOR_COLORS = {
            "bbs": "maroon",
            "rir": "limegreen",
            "cc": "red",
            "em": "purple",
            "rir_soft": "orange",
        }

        def plot(ax, props):
            x_axis = np.arange(len(props["cell_types"]))
            ax.set_xticks(x_axis, labels=props["cell_types"], rotation=70)
            ax.set_ylabel("Proportion")

            # Ground-truth
            ax.scatter(x_axis, props["true"], s=5**2, c="black", marker=".", rasterized=True)
            ax.plot(x_axis, props["true"], c="black", label="Observed", linestyle="--", linewidth=2, rasterized=True)

            # Posterior samples
            for sample in props["posterior"][-200:]:
                ax.plot(x_axis, sample, c="darkblue", alpha=0.1, linewidth=0.1, rasterized=True)
                ax.scatter(x_axis, sample, s=1, c="darkblue", alpha=0.08, marker=".", rasterized=True)

            # Posterior mean
            posterior_mean = np.mean(props["posterior"], axis=0)
            ax.plot(x_axis, posterior_mean, c="darkblue", label="Posterior\nmean", linewidth=1, alpha=0.5, rasterized=True)
            ax.scatter(x_axis, posterior_mean, s=3**2, c="darkblue", alpha=0.5, marker=".", rasterized=True)

            # Point estimators
            for estimator, name in ESTIMATOR_NAMES.items():
                ax.plot(x_axis, props[estimator], c=ESTIMATOR_COLORS[estimator], label=name, alpha=0.5, linewidth=1, rasterized=True)
                ax.scatter(x_axis, props[estimator], s=3**2, c=ESTIMATOR_COLORS[estimator], alpha=0.5, marker=".", rasterized=True)

        # Plot predictions: full data
        ax = axs[2]
        ax.set_title("All cell types")
        plot(ax, np.load(input.proportions_full, allow_pickle=True))

        # Plot predictions: aggregated
        ax = axs[3]
        ax.set_title("Aggregated cell types")
        plot(ax, np.load(input.proportions_agg, allow_pickle=True))

        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1), fontsize=6)


        fig.savefig(output[0])


rule estimate_proportions:
    input: "data/{name}_data.h5ad"
    output:
        estimates = "proportions/{name}.npz",
        error_log = "proportions/{name}.log"
    run:
        # Load the data
        adata = sc.read_h5ad(input[0])
        cell_type_encoder, sample_encoder = construct_encoders(adata)

        L = len(cell_type_encoder.classes_)

        # Select training, validation, and test sets
        train_samples = ["BT_S1", "BT_S2"]
        valid_sample = "BT_S4"
        test_sample = "BT_S6"
        assert len(set(train_samples + [valid_sample, test_sample])) == 4

        train_data = adata[(adata.obs["sample"] == train_samples[0]) | (adata.obs["sample"] == train_samples[1])].copy()
        valid_data = adata[adata.obs["sample"] == valid_sample].copy()
        test_data = adata[adata.obs["sample"] == test_sample].copy()

        # Do PCA + random forest
        pca = PCA(n_components=50, random_state=0)
        pca.fit(get_features(train_data))

        forest = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42)
        forest.fit(get_features(train_data), cell_type_encoder.transform(train_data.obs["cell_type"]))

        valid_predicted = forest.predict(get_features(valid_data))
        valid_labels = cell_type_encoder.transform(valid_data.obs["cell_type"])

        test_predicted = forest.predict(get_features(test_data))

        # Create summary statistic
        n_y_and_c_labeled = summ.count_values_joint(L, L, valid_labels, valid_predicted)
        n_y_labeled = summ.count_values(L, valid_labels)
        n_c_unlabeled = summ.count_values(L, test_predicted)
        statistic = SummaryStatistic(
            n_y_labeled=n_y_labeled,
            n_y_and_c_labeled=n_y_and_c_labeled,
            n_c_unlabeled=n_c_unlabeled,
        )

        posterior = algo.DiscreteCategoricalMeanEstimator().sample_posterior(statistic)[algo.DiscreteCategoricalMeanEstimator.P_TEST_Y]

        failed_counter = 0

        try:
            bbs = algo.BlackBoxShiftEstimator(enforce_square=True).estimate_from_summary_statistic(statistic)
        except Exception as e:
            bbs = np.full(L, np.nan)
            failed_counter += 1
            with open(output.error_log, "a") as f:
                f.write("BBS failed\n")
                f.write(str(e))
                f.write("\n")

        try:
            rir = algo.InvariantRatioEstimator(restricted=True, enforce_square=True).estimate_from_summary_statistic(statistic)
        except Exception as e:
            rir = np.full(L, np.nan)
            failed_counter += 1
            with open(output.error_log, "a") as f:
                f.write("RIR (hard) failed\n")
                f.write(str(e))
                f.write("\n")

        try:
            cc = algo.ClassifyAndCount().estimate_from_summary_statistic(statistic)
        except Exception as e:
            cc = np.full(L, np.nan)
            failed_counter += 1
            with open(output.error_log, "a") as f:
                f.write("CC failed\n")
                f.write(str(e))
                f.write("\n")

        # Algorithms using soft labels
        soft_pred = forest.predict_proba(get_features(test_data))
        
        train_counts = summ.count_values(L, cell_type_encoder.transform(train_data.obs["cell_type"]))

        try:
            _jitter = 0
            em = expectation_maximization(
                predictions=soft_pred,
                training_prevalences=train_counts / np.sum(train_counts),
            )
        except Exception as e:
            em = np.full(L, np.nan)
            failed_counter += 1
            with open(output.error_log, "a") as f:
                f.write("EM failed\n")
                f.write(str(e))
                f.write("\n")

        try:
            rir_soft = re.calculate_vector_and_matrix_from_predictions(
                unlabeled_predictions=soft_pred,
                labeled_predictions=forest.predict_proba(get_features(valid_data)),
                labeled_ground_truth=valid_labels,
            )
        except Exception as e:
            rir_soft = np.full(L, np.nan)
            failed_counter += 1
            with open(output.error_log, "a") as f:
                f.write("RIR (soft) failed\n")
                f.write(str(e))
                f.write("\n")

        with open(output.error_log, "a") as f:
            f.write("\n\nSummary:\n")
            f.write(f"Failed runs: {failed_counter}\n")

        test_labels = cell_type_encoder.transform(test_data.obs["cell_type"])
        n_y_test = summ.count_values(L, test_labels)
        ordering = np.argsort(n_y_test)[::-1]

        np.savez(output.estimates, **{
            "posterior": posterior[:, ordering],
            "bbs": bbs[ordering],
            "rir": rir[ordering],
            "cc": cc[ordering],
            "em": em[ordering],
            "rir_soft": rir_soft[ordering],
            "true": n_y_test[ordering] / np.sum(n_y_test),
            "cell_types": np.asarray(cell_type_encoder.classes_)[ordering],
        })
