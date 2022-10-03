"""Tests of the Invariant Ratio Estimator algorithm."""
import numpy as np
import pytest

import labelshift.algorithms.ratio_estimator as re
import labelshift.datasets.discrete_categorical as dc


@pytest.mark.parametrize("restricted", (True, False))
def test_ratio_estimator_from_sufficient_statistic(
    restricted: bool, n_labeled: int = 20_000, n_unlabeled: int = 20_000
) -> None:
    """Generates the data according to the P(C|Y) model."""
    p_y_labeled = np.asarray([0.4, 0.6])
    p_y_unlabeled = np.asarray([0.7, 0.3])

    p_c_cond_y = np.asarray(
        [
            [0.95, 0.04, 0.01],
            [0.04, 0.95, 0.01],
        ]
    )

    sampler = dc.DiscreteSampler(
        p_y_labeled=p_y_labeled, p_y_unlabeled=p_y_unlabeled, p_c_cond_y=p_c_cond_y
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
