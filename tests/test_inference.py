from dataclasses import replace

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from brainweb_kpp.inference import Phantom, SyntheticExperiment, fit_parameters


@pytest.fixture
def experiment():
    old_threads = get_num_threads()
    set_num_threads(1)
    yield SyntheticExperiment(Phantom(width_mm=24, voxel_mm=4, dt_days=0.25,
                                     seed_sigma_mm=(2.5, 3, 2.5), seed_center_mm=(0, 0, 0)))
    set_num_threads(old_threads)


def test_reaction_matches_exact_logistic(experiment):
    times = np.array([0, 1.3, 5.7])
    actual = experiment.states(0, 0.07, times)
    u0 = experiment.initial
    expected = np.array([u0 / (np.exp(-0.07 * t) + u0 * (1 - np.exp(-0.07 * t)))
                         for t in times])
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-15)


def test_diffusion_and_imaging_conserve_mass(experiment):
    states = experiment.states(0.4, 0, [0, 7.3])
    np.testing.assert_allclose(states.sum(axis=(1, 2, 3)), experiment.initial.sum(), rtol=1e-13)
    observed = experiment.observe(states)
    np.testing.assert_allclose(observed.sum(axis=(1, 2, 3)) * experiment.phantom.voxel_mm**3,
                               states.sum(axis=(1, 2, 3)) * experiment.phantom.dx_mm**3,
                               rtol=1e-13)
    assert states.min() >= 0


def test_observation_preserves_constants_and_does_not_clip(experiment):
    # Noise belongs after imaging. Clipping negative measurements biases inference.
    fields = np.stack([np.full(experiment.shape, value) for value in (-0.1, 0.5, 1.1)])
    observed = experiment.observe(fields)
    np.testing.assert_allclose(observed[:, 0, 0, 0], [-0.1, 0.5, 1.1])
    assert observed.shape == (3, 6, 6, 6)


def test_noiseless_parameter_recovery_and_withheld_forecast(experiment):
    truth = np.array([0.27, 0.047])
    times = [10, 20, 30]
    observed = experiment.predict(*truth, times)
    fit = fit_parameters(experiment, times, observed, 0.02, start=(0.6, 0.07))
    np.testing.assert_allclose(fit.parameters, truth, rtol=1e-5)
    np.testing.assert_allclose(experiment.predict(*fit.parameters, [45]),
                               experiment.predict(*truth, [45]), rtol=3e-5, atol=1e-9)
    assert np.linalg.eigvalsh(fit.log_covariance).min() > 0
    assert np.all(fit.intervals[:, 0] < truth)
    assert np.all(fit.intervals[:, 1] > truth)
    assert not fit.at_bound


def test_known_noise_scales_information_intervals(experiment):
    times = [10, 20, 30]
    observed = experiment.predict(0.27, 0.047, times)
    fit1 = fit_parameters(experiment, times, observed, 0.01)
    fit2 = fit_parameters(experiment, times, observed, 0.02)
    np.testing.assert_allclose(fit2.log_covariance, 4 * fit1.log_covariance, rtol=1e-5)


def test_uninformative_initial_scan_rejected(experiment):
    observed = experiment.predict(0.3, 0.04, [0])
    with pytest.raises(ValueError, match="not locally identifiable"):
        fit_parameters(experiment, [0], observed, 0.03)


def test_grid_alignment_and_unstable_steps_rejected(experiment):
    with pytest.raises(ValueError, match="align"):
        SyntheticExperiment(replace(experiment.phantom, voxel_mm=5))
    with pytest.raises(ValueError, match="stability"):
        experiment.states(4, 0.04, [1])
    with pytest.raises(ValueError, match="strictly increasing"):
        experiment.states(0.3, 0.04, [2, 1])


def test_invalid_observations_and_noise_rejected(experiment):
    with pytest.raises(ValueError, match="positive noise"):
        fit_parameters(experiment, [1], np.zeros((1, 6, 6, 6)), 0)
    with pytest.raises(ValueError, match="observations"):
        fit_parameters(experiment, [1], np.zeros((1, 3, 3, 3)), 0.03)
