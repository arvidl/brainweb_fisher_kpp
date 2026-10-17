from dataclasses import replace

import numpy as np
import pytest

from brainweb_kpp.anisotropic import simulate
from brainweb_kpp.pat16 import PilotData
from brainweb_kpp.pat16_inference import (
    PAT16Experiment, PAT16RecoveryDesign, fit_pat16_parameters,
)


@pytest.fixture
def experiment():
    shape = (17, 19, 17)
    tissue = np.full(shape, 2, np.uint8)
    tissue[:8] = 3
    mask = np.ones(shape, bool)
    mask[[0, -1]] = False
    mask[:, [0, -1]] = False
    mask[:, :, [0, -1]] = False
    mask[4, 5, 5] = False  # check no-flux holes too
    tissue[~mask] = 0
    x, y, z = np.indices(shape)
    lesion = ((x-8)**2 + (y-9)**2 + (z-8)**2 < 5).astype(float)
    directions = np.broadcast_to([1., 2., -.5], shape+(3,)).copy()
    pilot = PilotData(np.ones(shape), tissue, lesion, mask, directions,
                      np.ones(shape), mask & (tissue == 3), np.zeros(shape),
                      np.eye(4), 1., {})
    design = PAT16RecoveryDesign(dt_days=.1, observation_center_mm=(8, 9, 8),
                                 observation_voxels=4, voxel_mm=3, blur_sigma_mm=1)
    return PAT16Experiment(pilot, design)


def test_double_precision_matches_pilot_and_conserves_mass(experiment):
    e = experiment
    matrix = e.transport(.5, .1, 2.5)
    times = [0, 1.3, 4.8]
    states = e.states(.5, .1, 0, 2.5, times)
    reference, _ = simulate(matrix, e.initial, times, rho=0, dt=e.design.dt_days)
    assert states.dtype == np.float64
    np.testing.assert_allclose(states, reference, rtol=6e-8, atol=1e-15)
    np.testing.assert_allclose(states.sum(axis=1), e.initial.sum(), rtol=2e-14)
    assert states.min() >= 0 and states.max() <= 1


def test_nested_prediction_trial_refits_and_resumes(experiment, tmp_path):
    from dataclasses import asdict
    from brainweb_kpp.pat16_prediction import _initialize, _trial
    e = experiment
    days = np.array([8., 16., 24., 30., 36.])
    b = dict(days=days, clean=e.predict(.5, .1, .04, 2.5, days), noise_sd=.001,
             training_scans=3, near=np.ones(e.observation_shape, bool), sentinel=[2, 2, 2])
    _initialize(e.pilot, asdict(e.design))
    settings = dict(trials=2, bootstrap=2, predictive_draws=200, seed=53)
    index, metadata = _trial(0, str(tmp_path), b, settings, 'test')
    assert index == 0 and metadata['usable'] and metadata['bootstrap_successful'] == 2
    z = np.load(tmp_path/'trial_000.npz')
    assert z['intervals'].shape == (3, 2, 2, 4, 4, 4)
    assert z['metrics'].shape == (3, 2, 2, 4)
    assert np.all(np.isfinite(z['intervals']))
    _, replay = _trial(0, str(tmp_path), b, settings, 'test')
    assert replay == metadata
    import pytest
    with pytest.raises(ValueError, match='Checkpoint differs'):
        _trial(0, str(tmp_path), b, settings, 'changed')


def test_failed_prediction_trial_is_retained_as_coverage_miss(experiment, tmp_path, monkeypatch):
    from dataclasses import asdict
    from brainweb_kpp import pat16_prediction as prediction
    prediction._initialize(experiment.pilot, asdict(experiment.design))
    def failed_fit(*args, **kwargs):
        raise RuntimeError('deliberate failure')
    monkeypatch.setattr(prediction, 'fit_pat16_parameters', failed_fit)
    b = dict(days=np.array([8., 16., 24., 30., 36.]), clean=np.zeros((5, 4, 4, 4)),
             noise_sd=.015, training_scans=3, near=np.ones((4, 4, 4), bool), sentinel=[2, 2, 2])
    _, metadata = prediction._trial(0, str(tmp_path), b,
                                   dict(trials=2, bootstrap=2, predictive_draws=200, seed=53), 'test')
    assert not metadata['usable'] and metadata['primary_failed']
    assert metadata['failures'] == [{'primary': 'deliberate failure'}]
    with np.load(tmp_path/'trial_000.npz') as z:
        assert not z['sentinel_hits'].any()
        assert np.all(z['metrics'][..., 0] == 0)
        assert np.all(np.isnan(z['metrics'][..., 1:3]))


def test_four_parameter_recovery_and_withheld_forecast(experiment):
    e = experiment
    truth = [.45, .12, .045, 2.6]
    days = [8, 16, 24]
    fit = fit_pat16_parameters(e, days, e.predict(*truth, days), .01,
                               start=(.3, .2, .03, 1.7))
    np.testing.assert_allclose(fit.parameters, truth, rtol=2e-5)
    np.testing.assert_allclose(e.predict(*fit.parameters, [30]), e.predict(*truth, [30]),
                               rtol=3e-5, atol=1e-9)
    assert np.linalg.eigvalsh(fit.log_covariance).min() > 0
    np.testing.assert_allclose(np.diag(fit.correlation), 1.)
    assert not fit.at_bound


@pytest.mark.parametrize('control', ['time_zero', 'no_directions', 'single_tissue'])
def test_unidentifiable_designs_are_rejected(experiment, control):
    e = experiment
    days = [0] if control == 'time_zero' else [8, 16]
    if control == 'no_directions':
        e = PAT16Experiment(replace(e.pilot, anisotropic=np.zeros(e.shape, bool)), e.design)
    if control == 'single_tissue':
        e = PAT16Experiment(replace(e.pilot, tissue=np.where(e.pilot.mask, 3, 0)), e.design)
    y = e.predict(.5, .1, .04, 2.5, days)
    with pytest.raises(ValueError, match='not locally identifiable'):
        fit_pat16_parameters(e, days, y, .01)


def test_voxel_operator_preserves_physical_linear_signal(experiment):
    e = PAT16Experiment(experiment.pilot, replace(experiment.design, blur_sigma_mm=0.))
    x, y, z = np.indices(e.shape)
    # Use a full mask for this imaging-operator check.
    p = replace(e.pilot, mask=np.ones(e.shape, bool), tissue=np.where(x < 8, 3, 2))
    e = PAT16Experiment(p, e.design)
    actual = e.observe((x + 2*y + 3*z).ravel()[e.active][None])[0]
    a, b, c = np.meshgrid(*e.observation_axes, indexing='ij')
    np.testing.assert_allclose(actual, a+2*b+3*c, atol=1e-12)


def test_invalid_data_and_bounds(experiment):
    e = experiment
    y = e.predict(.5, .1, .04, 2.5, [10])
    with pytest.raises(ValueError, match='positive noise'):
        fit_pat16_parameters(e, [10], y, 0)
    with pytest.raises(ValueError, match='observation'):
        fit_pat16_parameters(e, [10], y[:, :2], .01)
    with pytest.raises(ValueError, match='ratio bounds'):
        fit_pat16_parameters(e, [10], y, .01, bounds=((.05, .02, .01, 1), (1, .6, .09, 4)))
    with pytest.raises(ValueError, match='ratio'):
        e.predict(.5, .1, .04, 4, [10])
    with pytest.raises(ValueError, match='inside the image'):
        PAT16Experiment(e.pilot, replace(e.design, observation_center_mm=(0, 0, 0)))
    with pytest.raises(ValueError, match='strictly increasing'):
        e.predict(.5, .1, .04, 2.5, [10, 5])


def test_isotropic_limit_has_no_ratio_effect_without_eligible_wm(experiment):
    e = PAT16Experiment(replace(experiment.pilot, anisotropic=np.zeros(experiment.shape, bool)), experiment.design)
    np.testing.assert_array_equal(e.predict(.5, .1, .04, 1., [10]),
                                  e.predict(.5, .1, .04, 3.5, [10]))


def test_complete_recovery_audit_records_uncertainty_and_provenance(experiment, tmp_path):
    import json
    from brainweb_kpp.pat16 import sha256
    from brainweb_kpp.pat16_recovery import PAT16RecoverySettings, run_pat16_recovery

    settings = PAT16RecoverySettings(scan_days=(8., 16., 24., 30.),
                                      n_bootstrap=2, n_coverage=2,
                                      starts=((.3, .2, .03, 1.7),))
    report, arrays = run_pat16_recovery(experiment.pilot, tmp_path, design=experiment.design,
                                        settings=settings, progress=None)
    assert report['bootstrap_successful'] == 2
    assert report['coverage_successful'] == 2
    assert report['numerical_validation']['matched_model_max_relative_error'] < 1e-4
    assert 'not locally identifiable' in report['numerical_validation']['time_zero_negative_control']
    assert report['forecast_full_region'][0]['day'] == 30.
    assert report['forecast_near_center'][0]['voxels'] > 0
    assert arrays['bootstrap_parameters'].shape == (2, 4)
    assert arrays['noise_trial_intervals'].shape == (2, 4, 2)
    assert arrays['signal_interval'].shape == (2, 4, 4, 4, 4)
    assert arrays['prediction_interval'].shape == arrays['signal_interval'].shape
    for row in report['parameters']:
        assert 0 <= row['local_coverage_hits'] <= 2
    saved = json.loads((tmp_path/'summary.json').read_text())
    assert saved['settings']['seed'] == settings.seed
    provenance = json.loads((tmp_path/'provenance.json').read_text())
    for name, expected in provenance['outputs_sha256'].items():
        assert sha256(tmp_path/name) == expected
