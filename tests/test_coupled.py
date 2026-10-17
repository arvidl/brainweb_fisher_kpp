"""Physical balances, analytical limits and saved-field/replay compatibility."""
from dataclasses import replace

import numpy as np
import pytest

from brainweb_kpp.data import Brain
from brainweb_kpp.model import Parameters, OxygenSolver, coupled_reaction, simulate, get_planes
from brainweb_kpp.render import (save_result, load_result, SlideRenderer, ComparisonRenderer,
                                export_media, field_data, overlay_rgba, VIEW_STYLES)


def cube(shape=(9, 7, 5)):
    return Brain(np.ones(shape, np.float32), np.full(shape, 3, np.uint8), np.eye(4), 1.)


def params(**kwargs):
    return replace(Parameters(model="coupled", origin_mm=(4., 3., 2.), radii_mm=(100.,) * 3,
                              seed_profile="ellipsoid", initial_density=80000., duration_days=2.,
                              frame_days=.5, dt_days=.1), **kwargs)


def test_uniform_oxygen_matches_supply_uptake_balance():
    b, p = cube(), params()
    mask = b.mask
    solver = OxygenSolver(mask, 1., p)
    c, z, k = np.full(mask.shape, .6), np.full(mask.shape, .2), np.ones(mask.shape)
    oxygen = solver.solve(c, z, k)
    supply = 1 / (1 + p.perfusion_suppression * .8)
    np.testing.assert_allclose(oxygen, supply / (supply + p.oxygen_uptake_ratio * .6), rtol=2e-7)
    np.testing.assert_array_equal(solver.solve(c * 0, z * 0, k), 1.)


def test_oxygen_no_flux_barrier_and_no_wrap():
    shape = (9, 4, 3)
    mask = np.ones(shape, bool)
    mask[4] = False
    c, z, k = np.zeros(shape), np.zeros(shape), np.ones(shape)
    c[:4] = 1
    solver = OxygenSolver(mask, 1., params(oxygen_length_mm=2.))
    oxygen = solver.solve(c, z, k)
    np.testing.assert_allclose(oxygen[:4], 1 / 21, rtol=2e-6)
    np.testing.assert_allclose(oxygen[5:], 1., rtol=2e-6)
    assert np.all(oxygen[4] == 0)


def test_oxygen_recovers_nonuniform_neumann_cosine_solution():
    b = cube((20, 5, 3))
    p = params(perfusion_suppression=0., oxygen_length_mm=.8, oxygen_rtol=1e-11)
    mode = np.broadcast_to(np.cos(np.pi * (np.arange(20) + .5) / 20)[:, None, None], b.labels.shape)
    exact = .6 + .05 * mode
    laplace = -4 * np.sin(np.pi / 40) ** 2 * .05 * mode
    c = (p.oxygen_length_mm ** 2 * laplace + 1 - exact) / (p.oxygen_uptake_ratio * exact)
    actual = OxygenSolver(b.mask, 1., p).solve(c, c * 0, np.ones(c.shape))
    np.testing.assert_allclose(actual, exact, rtol=1e-7)


def test_coupled_time_refinement_against_independent_uniform_rk4():
    # No spatial gradients: the oxygen equation has an algebraic solution.
    p = params(duration_days=3., frame_days=3., initial_density=80000.)
    def rhs(state):
        u, z = state
        supply = 1 / (1 + p.perfusion_suppression * (u + z))
        oxygen = supply / (supply + p.oxygen_uptake_ratio * u)
        births = p.rho * min(oxygen / p.oxygen_prolif_level, 1) * u * max(1 - u - z, 0)
        death = p.necrosis_rate * max(1 - oxygen / p.oxygen_death_level, 0) * u
        return np.array([births - death, death - p.necrosis_clearance * z])
    expected = np.array([.8, 0.])
    dt = .002
    for _ in range(round(p.duration_days / dt)):
        a = rhs(expected)
        b = rhs(expected + dt * a / 2)
        c = rhs(expected + dt * b / 2)
        d = rhs(expected + dt * c)
        expected += dt * (a + 2*b + 2*c + d) / 6
    errors = []
    for step in (.2, .025):
        r = simulate(cube(), replace(p, dt_days=step))
        observed = np.array([r.final_density.mean(), r.final_necrosis.mean()]) / p.carrying_capacity
        errors.append(np.linalg.norm(observed - expected))
    assert errors[1] < .35 * errors[0]
    assert errors[1] < .001


def test_anoxic_death_transfers_exactly_and_clearance_is_exponential():
    c, z = np.array([30000.]), np.array([10000.])
    coupled_reaction(c, z, np.array([0.]), np.array([0.]), np.array([100000.]),
                     np.array([0]), 3., .3, .1, .5, 0.)
    np.testing.assert_allclose(c, 30000 * np.exp(-1.5), rtol=1e-12)
    np.testing.assert_allclose(c + z, 40000., rtol=1e-12)
    before = z.copy()
    coupled_reaction(c, z, np.array([1.]), np.array([0.]), np.array([100000.]),
                     np.array([0]), 2., .3, .1, .5, .2)
    np.testing.assert_allclose(z, before * np.exp(-.4), rtol=1e-12)


def test_necrotic_material_occupies_growth_capacity():
    c, z = np.array([20000., 20000.]), np.array([40000., 120000.])
    coupled_reaction(c, z, np.ones(2), np.full(2, .2), np.full(2, 100000.),
                     np.arange(2), 10., .3, .1, 0., 0.)
    expected = 60000 / (1 + (60000 / 20000 - 1) * np.exp(-.2 * .6 * 10))
    np.testing.assert_allclose(c, [expected, 20000.], rtol=1e-12)


def test_zero_uptake_recovers_fisher_kpp_and_matching_reference():
    b = cube()
    p = params(oxygen_uptake_ratio=0., initial_density=20000., radii_mm=(2.,) * 3,
               d_csf=.5)
    progress = []
    coupled = simulate(b, p, compare=True, store_volumes=True,
                       progress=lambda done, total: progress.append(done / total))
    original = simulate(b, replace(p, model="fisher_kpp", d_csf=0.))
    np.testing.assert_allclose(coupled.final_density, original.final_density, rtol=3e-6)
    np.testing.assert_array_equal(coupled.final_necrosis, 0.)
    np.testing.assert_allclose(coupled.final_oxygen, 1., atol=1e-6)
    np.testing.assert_array_equal(coupled.comparison.final_density, original.final_density)
    assert progress[-1] == 1 and np.all(np.diff(progress) >= 0)
    assert len(coupled.oxygen_volumes) == len(coupled.times)


def test_hypoxic_simulation_conserves_live_plus_dead_without_births():
    r = simulate(cube(), params(rho=0.))
    assert r.metrics['necrotic_cells'][-1] > 0
    assert r.metrics['cells'][-1] < r.metrics['cells'][0]
    np.testing.assert_allclose(r.metrics['cells'] + r.metrics['necrotic_cells'],
                               r.metrics['cells'][0], rtol=2e-6)
    assert r.final_density.min() >= 0 and r.final_necrosis.min() >= 0
    assert r.final_oxygen.min() > 0 and r.final_oxygen.max() <= 1
    assert r.metadata['oxygen_solver']['solves'] == r.metadata['internal_steps'] + 1


def test_coupled_excludes_csf_even_with_custom_diffusion():
    b = cube()
    b.labels[5] = 1
    r = simulate(b, params(d_csf=2.), diffusion=np.ones(b.labels.shape))
    assert np.all(r.final_density[5] == 0)
    assert np.all(r.final_oxygen[5] == 0)
    assert np.all(r.final_necrosis[5] == 0)


def test_coupled_replay_roundtrip_and_all_views(tmp_path):
    r = simulate(cube(), params(duration_days=.5), compare=True)
    save_result(r, tmp_path)
    restored = load_result(tmp_path)
    for name in ('final_density', 'final_oxygen', 'final_necrosis', 'tissue_mask'):
        np.testing.assert_array_equal(getattr(restored, name), getattr(r, name))
    for field in ('oxygen_planes', 'necrosis_planes'):
        for plane in ('Axial', 'Coronal', 'Sagittal'):
            np.testing.assert_array_equal(getattr(restored, field)[plane], getattr(r, field)[plane])
    np.testing.assert_array_equal(restored.comparison.final_density, r.comparison.final_density)
    for view in ('viable', 'oxygen', 'necrosis', 'reference'):
        assert SlideRenderer(restored, width=640, view=view).frame(1).shape == (360, 640, 3)


def test_original_replay_without_new_parameters(tmp_path):
    r = simulate(cube(), replace(params(), model='fisher_kpp'))
    # Legacy replays lack the model selector and every coupled parameter.
    r.parameters = {name: value for name, value in r.parameters.items()
                    if name not in ('model', 'perfusion_suppression') and not name.startswith(('oxygen_', 'necrosis_'))}
    save_result(r, tmp_path)
    restored = load_result(tmp_path)
    np.testing.assert_array_equal(restored.final_density, r.final_density)
    assert restored.final_oxygen is None and restored.comparison is None
    assert SlideRenderer(restored, width=640).frame(0).shape == (360, 640, 3)


def test_comparison_renderer_synchronizes_fields_and_custom_slices():
    b = cube()
    r = simulate(b, params(radii_mm=(2.,) * 3, duration_days=.5), compare=True, store_volumes=True)
    indices = (3, 2, 1)
    renderer = ComparisonRenderer(r, width=640, indices=indices, brain=b)
    for i in (0, len(r.times) - 1):
        assert renderer.frame(i).shape == (480, 640, 3)
        assert renderer.day.get_text() == f'DAY {r.times[i]:05.1f}'
        for view in renderer.fields:
            _, _, volumes, scale = field_data(r, view)
            planes = get_planes(volumes[i], indices)
            masks = get_planes(r.tissue_mask, indices) if view == 'oxygen' else None
            for name, image in renderer.images[view].items():
                expected = overlay_rgba(planes[name], scale, cmap=VIEW_STYLES[view][2],
                                        threshold=0 if view == 'oxygen' else .02,
                                        mask=masks[name] if masks is not None else None)
                np.testing.assert_array_equal(image.get_array(), expected)
            assert renderer.colorbars[view].mappable.get_clim() == (0, 1)
    assert ComparisonRenderer(r, width=640, plane='Axial').frame(0).shape == (360, 640, 3)
    with pytest.raises(ValueError, match='identical saved times'):
        ComparisonRenderer(replace(r, comparison=replace(r.comparison, times=r.times + .01)))
    with pytest.raises(ValueError, match='requires coupled fields'):
        ComparisonRenderer(replace(r, comparison=None))
    with pytest.raises(ValueError, match='saved 3D frames'):
        ComparisonRenderer(replace(r, volumes=None), indices=indices, brain=b)


def test_comparison_mp4_gif_decode_and_timeline_metadata(tmp_path):
    import json
    import imageio.v2 as imageio
    from PIL import Image
    r = simulate(cube(), params(duration_days=.5), compare=True)
    export_media(r, tmp_path, view='comparison', width=640, fps=5)
    metadata = json.loads((tmp_path / 'comparison_media_metadata.json').read_text())
    assert metadata['fields'] == ['viable', 'oxygen', 'necrosis', 'reference']
    assert metadata['saved_times_days'] == r.times.tolist()
    assert metadata['synchronized'] and metadata['planes'] == ['Axial', 'Coronal', 'Sagittal']
    assert metadata['rendered_times_days'] == [r.times[i] for i in metadata['rendered_frame_indices']]
    reader = imageio.get_reader(tmp_path / 'comparison.mp4')
    try:
        frames = [frame for frame in reader]
    finally:
        reader.close()
    assert len(frames) == len(metadata['rendered_frame_indices'])
    assert all(frame.shape == (480, 640, 3) for frame in frames)
    with Image.open(tmp_path / 'comparison.gif') as gif:
        assert gif.size == (640, 480) and gif.info['loop'] == 0
        duration = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            gif.load()
            duration += gif.info['duration']
    assert duration / 1000 == pytest.approx(metadata['video_seconds'])


@pytest.mark.parametrize('kwargs', [dict(oxygen_death_level=.4), dict(necrosis_rate=-1),
                                  dict(oxygen_length_mm=-.1), dict(oxygen_rtol=0),
                                  dict(oxygen_maxiter=0), dict(model='unknown')])
def test_coupled_parameter_validation(kwargs):
    with pytest.raises(ValueError): params(**kwargs).validate()


def test_unconverged_oxygen_fails_explicitly():
    b = cube()
    rng = np.random.default_rng(7)
    solver = OxygenSolver(b.mask, 1., params(oxygen_length_mm=3., oxygen_maxiter=1, oxygen_rtol=1e-12))
    with pytest.raises(RuntimeError, match='did not converge'):
        solver.solve(rng.random(b.labels.shape), np.zeros(b.labels.shape), np.ones(b.labels.shape))


def test_inactive_coupled_thresholds_do_not_block_original_model():
    p = params(model='fisher_kpp', oxygen_death_level=.8, oxygen_prolif_level=.3)
    result = simulate(cube(), p)
    assert result.final_oxygen is None
