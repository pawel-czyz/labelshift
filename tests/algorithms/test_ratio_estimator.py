"""Tests of the Invariant Ratio Estimator algorithm."""
import numpy as np
import pytest

import labelshift.algorithms.ratio_estimator as re
import labelshift.datasets.discrete_categorical as dc


@pytest.mark.parametrize("restricted", (True, False))
@pytest.mark.parametrize("p1_labeled", [0.4, 0.8])
@pytest.mark.parametrize("p1_unlabeled", [0.2, 0.5, 0.9])
def test_ratio_estimator_from_sufficient_statistic(
    restricted: bool,
    p1_labeled: float,
    p1_unlabeled: float,
    n_labeled: int = 20_000,
    n_unlabeled: int = 20_000,
) -> None:
    """Generates the data according to the P(C|Y) model."""
    p_y_labeled = np.asarray([1 - p1_labeled, p1_labeled])
    p_y_unlabeled = np.asarray([1 - p1_unlabeled, p1_unlabeled])

    p_c_cond_y = np.asarray(
        [
            [0.95, 0.04, 0.01],
            [0.04, 0.95, 0.01],
        ]
    )

    sampler = dc.discrete_sampler_factory(
        p_y_labeled=p_y_labeled,
        p_y_unlabeled=p_y_unlabeled,
        p_c_cond_y_labeled=p_c_cond_y,
    )
    statistic = sampler.sample_summary_statistic(
        n_labeled=n_labeled, n_unlabeled=n_unlabeled, seed=111
    )

    p_y_estimated = re.prevalence_from_summary_statistics(
        n_y_and_c_labeled=statistic.n_y_and_c_labeled,
        n_c_unlabeled=statistic.n_c_unlabeled,
        enforce_square=False,
        restricted=restricted,
    )

    assert p_y_estimated == pytest.approx(p_y_unlabeled, abs=0.01)
