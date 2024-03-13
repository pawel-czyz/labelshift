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

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


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
        adata.write(output.data)

        with open(output.summary, "a") as f:
            for name, c in zip(*np.unique(adata.obs["cell_type"], return_counts=True)):
                f.write(f"{name}: {c}\n")


rule create_rough_cell_types:
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
    include_legend: bool = False,
    bbox_to_anchor: tuple[float, float] = (1, 1),
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
        ax.legend(handles=patches, loc="upper left", bbox_to_anchor=bbox_to_anchor, frameon=False)


rule plot_p_x_y:
    input: "data/full_data.h5ad"
    output: "plots/p_x_y.pdf"
    run:
        adata = sc.read_h5ad(input[0])
        pca = PCA(n_components=4)
        pca.fit(get_features(adata))

        cell_type_encoder, sample_encoder = construct_encoders(adata)
        n_samples = len(sample_encoder.classes_)

        fig, axs = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 2 * 1.2), dpi=150)

        for ax in axs.ravel():
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        # Set scales for the axes
        _reps = pca.transform(get_features(adata))
        for ax in axs[0, :]:
            ax.set_xlim(np.min(_reps[:, 0]) - 1, np.max(_reps[:, 0]) + 1)
            ax.set_ylim(np.min(_reps[:, 1]) - 1, np.max(_reps[:, 1]) + 1)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
        for ax in axs[1, :]:
            ax.set_xlim(np.min(_reps[:, 2]) - 1, np.max(_reps[:, 2]) + 1)
            ax.set_ylim(np.min(_reps[:, 3]) - 1, np.max(_reps[:, 3]) + 1)
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
        full_data = "data/full_data.h5ad"
    output: "plots/manuscript_plot.pdf"
    run:
        adata = sc.read_h5ad(input[0])
        pca = PCA(n_components=2)
        pca.fit(get_features(adata))
        reps = pca.transform(get_features(adata))

        fig, axs = plt.subplots(1, 4, figsize=(6, 1.5), dpi=150)

        cell_type_encoder, sample_encoder = construct_encoders(adata)

        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)

        for ax in axs[:2]:
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
    
        axs[0].set_title("Cell types")
        plot_cells(axs[0], reps[:, 0], reps[:, 1], types=adata.obs["cell_type"], encoder=cell_type_encoder, include_legend=True, bbox_to_anchor=(1, 1))

        axs[1].set_title("Samples")
        plot_cells(axs[1], reps[:, 0], reps[:, 1], types=adata.obs["sample"], encoder=sample_encoder, cmap="tab10")

        fig.tight_layout()
        fig.savefig(output[0])

# rule estimate_proportions:
#     input: "data/{name}.h5ad"
#     output: "proportions/{name}.joblib"
#     run:
        # Load the data, select training, validation, and test sets
