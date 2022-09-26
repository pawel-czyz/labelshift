"""Tests for the labelshift/algorithms/bayesian_discrete.py"""
import numpy as np
import pytest

import labelshift.algorithms.bayesian_discrete as bd
import labelshift.datasets.discrete_categorical as dc


def test_right_values(n_labeled: int = 10_000, n_unlabeled: int = 10_000) -> None:
    """Test whether the mean over the samples gives the right answer
    in the limit of large data and little noise."""
    p_y_labeled = np.asarray([0.4, 0.6])
    p_y_unlabeled = np.asarray([0.7, 0.3])

    p_c_cond_y = np.asarray(
        [
            [0.95, 0.04, 0.01],
            [0.04, 0.95, 0.01],
        ]
    )

    sampler = dc.DiscreteSampler(
        p_y_labeled=[0.4, 0.6], p_y_unlabeled=p_y_unlabeled, p_c_cond_y=p_c_cond_y
    )
    statistic = sampler.sample_summary_statistic(
        n_labeled=n_labeled, n_unlabeled=n_unlabeled, seed=111
    )

    params = bd.SamplingParams(chains=1, draws=100)

    inference_result = bd.sample_from_bayesian_discrete_model_posterior(
        n_y_labeled=statistic.n_y_labeled,
        n_y_and_c_labeled=statistic.n_y_and_c_labeled,
        n_c_unlabeled=statistic.n_c_unlabeled,
        sampling_params=params,
    )

    def get_mean(key) -> np.ndarray:
        """Returns the mean over all samples for variable `key`."""
        return np.asarray(inference_result.posterior.data_vars[key].mean(axis=(0, 1)))

    assert get_mean(bd.P_TEST_Y) == pytest.approx(p_y_unlabeled, abs=0.02)
    assert get_mean(bd.P_TRAIN_Y) == pytest.approx(p_y_labeled, abs=0.02)
    assert get_mean(bd.P_C_COND_Y) == pytest.approx(p_c_cond_y, abs=0.02)
