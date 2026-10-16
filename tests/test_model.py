"""Analytical and conservation checks, independent of the BrainWeb anatomy."""
from dataclasses import replace
import numpy as np
import pytest
from brainweb_kpp.data import Brain, load_brainweb, verify_data
from brainweb_kpp.model import Parameters, simulate, initial_condition, face_coefficients


def brain(shape=(17, 15, 13), spacing=1):
    affine = np.diag([spacing, spacing, spacing, 1.])
    return Brain(np.ones(shape, dtype=np.float32), np.full(shape, 3, dtype=np.uint8), affine, spacing)


def parameters(**kwargs):
    return replace(Parameters(origin_mm=(8., 7., 6.), radii_mm=(2., 2., 2.),
                              duration_days=2, frame_days=.5, dt_days=.1), **kwargs)


def test_exact_logistic_without_diffusion():
    b, p = brain(), parameters(d_white=0, d_gray=0, rho=.2)
    initial = initial_condition(b, p)
    expected = initial / (np.exp(-p.rho * p.duration_days) + initial / p.carrying_capacity * (1 - np.exp(-p.rho * p.duration_days)))
    r = simulate(b, p)
    np.testing.assert_allclose(r.final_density, expected, rtol=4e-6, atol=.005)


def test_heterogeneous_diffusion_conserves_cells_and_is_positive():
    b, p = brain(), parameters(rho=0, dt_days=.8)
    rng = np.random.default_rng(24)
    d = rng.uniform(.01, 2, b.labels.shape).astype(np.float32)
    r = simulate(b, p, diffusion=d)
    np.testing.assert_allclose(r.metrics['cells'], r.metrics['cells'][0], rtol=3e-7)
    assert r.final_density.min() >= 0
    assert r.final_density.max() <= initial_condition(b, p).max()
    assert r.metadata['actual_max_dt_days'] < p.dt_days


def test_csf_and_external_barriers_are_impermeable():
    b, p = brain((19, 15, 13)), parameters(origin_mm=(4., 7., 6.), radii_mm=(1., 1., 1.), rho=0)
    b.labels[8:10] = 1
    b.labels[:, 0] = 7
    r = simulate(b, p)
    assert np.max(r.final_density[8:]) == 0
    assert np.max(r.final_density[:, 0]) == 0
    np.testing.assert_allclose(r.metrics['cells'], r.metrics['cells'][0], rtol=3e-7)


def test_no_periodic_wraparound():
    b, p = brain((19, 15, 13)), parameters(origin_mm=(0., 7., 6.), radii_mm=(.2, .2, .2),
                                          duration_days=.01, frame_days=.01, dt_days=.01, rho=0)
    r = simulate(b, p)
    assert np.max(r.final_density[-1]) == 0
    np.testing.assert_allclose(r.metrics['cells'], r.metrics['cells'][0], rtol=3e-7)


def test_unstable_dt_rejected_in_strict_mode():
    with pytest.raises(ValueError, match='safe diffusion step'):
        simulate(brain(), parameters(dt_days=4, auto_substep=False))


def test_uniform_capacity_is_an_invariant_upper_bound():
    b, p = brain(), parameters(initial_density=100000, rho=.3, dt_days=2)
    r = simulate(b, p)
    assert np.max(r.final_density) <= p.carrying_capacity * (1 + 1e-6)
    assert np.min(r.final_density) >= 0


def test_diffusion_manufactured_neumann_eigenmode():
    # Cell-centered cosine is an exact eigenvector of the reflecting discrete operator.
    from brainweb_kpp.model import advance
    b = brain((20, 5, 5))
    x = np.arange(20) + .5
    n = np.broadcast_to((10000 + 1000 * np.cos(np.pi * x / 20))[:, None, None], b.labels.shape).copy().astype(np.float32)
    f, _ = face_coefficients(np.ones_like(n), b.mask, 1)
    active = np.arange(n.size, dtype=np.int32)
    dt, steps = .05, 20
    a, out = n.ravel().copy(), np.zeros(n.size, dtype=np.float32)
    for _ in range(steps):
        advance(a, out, f, active, np.array([25, 5, 1]), np.ones(n.size, np.float32),
                np.full(n.size, 100000, np.float32), dt)
        a, out = out, a
    amplitude = (1 - 4 * dt * np.sin(np.pi / 40) ** 2) ** steps
    np.testing.assert_allclose(a.reshape(n.shape), 10000 + (n - 10000) * amplitude, rtol=3e-6)


def test_temporal_refinement_reduces_splitting_error():
    b = brain()
    p = parameters(rho=.2, duration_days=2, frame_days=2)
    ref = simulate(b, replace(p, dt_days=.0125)).final_density
    coarse = simulate(b, replace(p, dt_days=.2)).final_density
    fine = simulate(b, replace(p, dt_days=.1)).final_density
    assert np.linalg.norm(fine - ref) < .65 * np.linalg.norm(coarse - ref)


def test_frame_times_include_zero_and_exact_final_time():
    r = simulate(brain(), parameters(duration_days=2.1, frame_days=.8))
    np.testing.assert_allclose(r.times, [0, .8, 1.6, 2.1])
    assert len(r.planes['Axial']) == len(r.times)


def test_bundled_data_alignment_and_tissue_counts():
    verify_data()
    b = load_brainweb(1)
    assert b.labels.shape == (181, 217, 181)
    assert np.count_nonzero(b.labels == 2) == 902912
    assert np.count_nonzero(b.labels == 3) == 674777
    np.testing.assert_allclose(b.world((0, 0, 0)), [-90, -126, -72])


@pytest.mark.parametrize('kwargs', [{'rho': -1}, {'dt_days': 0}, {'radii_mm': (0, 1, 1)}, {'initial_density': 200000}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError): parameters(**kwargs).validate()
