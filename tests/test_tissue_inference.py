from dataclasses import replace

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from brainweb_kpp.data import Brain
from brainweb_kpp.model import face_coefficients
from brainweb_kpp.tissue_inference import (
    BrainWebExperiment, TissueRecoveryDesign, fit_tissue_parameters,
)


@pytest.fixture
def tissue_experiment():
    old = get_num_threads()
    set_num_threads(1)
    labels = np.full((17, 19, 17), 2, np.uint8)
    labels[:8] = 3
    labels[[0, -1], :, :] = 0
    labels[:, [0, -1], :] = 1
    labels[:, :, [0, -1]] = 0
    labels[11, 12, 11] = 8
    brain = Brain(np.ones(labels.shape), labels, np.eye(4), 1.)
    design = TissueRecoveryDesign(dt_days=.1, seed_center_mm=(8, 9, 8),
                                  seed_sigma_mm=(2, 2, 2), observation_center_mm=(8, 9, 8),
                                  observation_voxels=4, voxel_mm=3, blur_sigma_mm=1)
    yield BrainWebExperiment(brain, design)
    set_num_threads(old)


def test_tissue_face_lookup_matches_existing_harmonic_stencil(tissue_experiment):
    e = tissue_experiment
    diffusion = np.where(e.brain.labels == 3, .5, np.where(e.mask, .1, 0.))
    expected, _ = face_coefficients(diffusion, e.mask, e.brain.spacing)
    np.testing.assert_allclose(e.faces(.5, .1), expected, rtol=1e-14)


def test_tissue_transport_conserves_cells_and_excludes_csf(tissue_experiment):
    e = tissue_experiment
    states = e.states(.5, .1, 0, [0, 2.3, 5.7])
    np.testing.assert_allclose(states.sum(axis=(1, 2, 3)), e.initial.sum(), rtol=2e-14)
    assert np.all(states[:, ~e.mask] == 0)
    assert states.min() >= 0


def test_physical_voxel_operator_is_grid_aligned(tissue_experiment):
    e = tissue_experiment
    p = replace(e.design, blur_sigma_mm=0.)
    one = BrainWebExperiment(e.brain, p)
    two = BrainWebExperiment(Brain(e.brain.t1, e.brain.labels, np.diag([2., 2., 2., 1.]), 2.), p)
    # A linear world-coordinate signal has the same voxel average on both grids.
    for experiment in (one, two):
        h = experiment.brain.spacing
        x, y, z = np.indices(experiment.shape) * h
        actual = experiment.observe(x + 2 * y + 3 * z)
        a, b, c = np.meshgrid(*experiment.observation_axes, indexing="ij")
        np.testing.assert_allclose(actual, a + 2*b + 3*c, atol=1e-12)


def test_three_rate_recovery_and_withheld_forecast(tissue_experiment):
    e = tissue_experiment
    known = np.array([.45, .12, .045])
    scans = [8, 16, 24]
    observed = e.predict(*known, scans)
    fit = fit_tissue_parameters(e, scans, observed, .02, start=(.3, .2, .03))
    np.testing.assert_allclose(fit.parameters, known, rtol=1e-5)
    np.testing.assert_allclose(e.predict(*fit.parameters, [30]), e.predict(*known, [30]),
                               rtol=3e-5, atol=1e-9)
    np.testing.assert_allclose(np.diag(fit.correlation), 1)
    assert np.linalg.eigvalsh(fit.log_covariance).min() > 0
    assert not fit.at_bound


def test_missing_tissue_rejects_unidentifiable_diffusion(tissue_experiment):
    e = tissue_experiment
    brain = Brain(e.brain.t1, np.where(e.mask, 3, 0).astype(np.uint8), e.brain.affine, 1.)
    single = BrainWebExperiment(brain, e.design)
    with pytest.raises(ValueError, match="not locally identifiable"):
        fit_tissue_parameters(single, [10, 20], single.predict(.5, .1, .04, [10, 20]), .02)


def test_tissue_input_and_stability_checks(tissue_experiment):
    e = tissue_experiment
    with pytest.raises(ValueError, match="stability"):
        e.predict(2, .1, .04, [1])
    with pytest.raises(ValueError, match="nonnegative"):
        e.predict(-.2, .1, .04, [1])
    with pytest.raises(ValueError, match="observation"):
        fit_tissue_parameters(e, [1], np.zeros((1, 2, 2, 2)), .02)
    with pytest.raises(ValueError, match="positive noise"):
        fit_tissue_parameters(e, [1], np.zeros((1, 4, 4, 4)), 0)


def test_roi_outside_image_rejected(tissue_experiment):
    e = tissue_experiment
    with pytest.raises(ValueError, match="inside the image"):
        BrainWebExperiment(e.brain, replace(e.design, observation_center_mm=(0, 0, 0)))
