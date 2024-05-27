[![Project Status: Concept – Minimal or no implementation has been done yet, or the repository is only intended to be a limited example, demo, or proof-of-concept.](https://www.repostatus.org/badges/latest/concept.svg)](https://www.repostatus.org/#concept)
[![Venue](https://img.shields.io/badge/venue-TMLR_2024-darkblue)](https://openreview.net/forum?id=Ft4kHrOawZ)
[![build](https://github.com/pawel-czyz/labelshift/actions/workflows/build.yml/badge.svg)](https://github.com/pawel-czyz/labelshift/actions/workflows/build.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Bayesian quantification with black-box estimators

*Quantification* is the problem of estimating the label prevalence from an unlabeled data set. In this repository we provide the code associated with our manuscript, which can be used to reproduce the experiments.

## Installation

We recommend using [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) to set a new Python 3.11 environment.
Then, the package can be installed with:

```bash
$ pip install -e .
```

To reproduce the experiments, install [Snakemake](https://snakemake.readthedocs.io/en/stable/) using the instructions provided. Then, install additional dependencies:

```bash
$ pip install -r requirements.txt
```

The experiments can be reproduced by running:

```bash
$ snakemake -c4 -s workflows/WORKFLOW_NAME.smk
```

