import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from brainweb_kpp.pat16_prediction import (
    interval_metrics, mean_mc_interval, mixture_interval, predictive_interval,
)


def test_gaussian_mixture_matches_single_normal_and_unequal_components():
    means = np.broadcast_to([.1, .4], (5, 2))
    exact = mixture_interval(means, .015)
    np.testing.assert_allclose(exact, means[0]+.015*ndtri(np.array([.025, .975]))[:, None], atol=1e-14)
    asymmetric = np.array([[-.2, 0], [.1, .3], [.8, .35]])
    interval = mixture_interval(asymmetric, .06)
    probabilities = ndtr((interval[:, None]-asymmetric[None])/.06).mean(axis=1)
    np.testing.assert_allclose(probabilities, np.broadcast_to([[.025], [.975]], (2, 2)), atol=2e-9)


def test_more_predictive_draws_converge_to_mixture_and_repeat_exactly():
    means = np.array([[0, .01], [.02, .03], [.04, .01]])
    exact = mixture_interval(means, .015)
    mc = predictive_interval(means, .015, 20000, 21)
    np.testing.assert_array_equal(mc, predictive_interval(means, .015, 20000, 21))
    np.testing.assert_allclose(mc, exact, atol=.001)
    with pytest.raises(ValueError):
        predictive_interval(means, .015, 1, 21)
    with pytest.raises(ValueError):
        mixture_interval(means, 0)


def test_coverage_width_proper_interval_score_and_trial_uncertainty():
    bounds = np.array([[[[0., 0.]]], [[[1., 1.]]]])
    obs = np.array([[[-.1, .5]]])
    metrics = interval_metrics(bounds, obs, np.zeros_like(obs), .2, np.ones((1, 2), bool))
    np.testing.assert_allclose(metrics[0, :3], [.5, 1, 3])
    np.testing.assert_allclose(metrics[0, 3], ndtr(5)-.5)
    values = np.array([.9, .95, 1.])
    ci = mean_mc_interval(values)
    assert ci[0] < .95 < ci[1]
    # Replication is across independent datasets, never pooled voxels.
    np.testing.assert_array_equal(mean_mc_interval(np.tile(values[:, None], (1, 100)))[0], ci)
