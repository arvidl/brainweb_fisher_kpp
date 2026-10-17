"""Paired, prior-driven PAT16 growth scenarios and synchronized plots."""
from pathlib import Path
from importlib.metadata import version
import json
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import norm, qmc
from .anisotropic import migration_tensor, transport_matrix, simulate
from .pat16 import initial_density, sha256


PARAMETER_NAMES = ['D_WM_mm2_day', 'D_GM_mm2_day', 'rho_per_day', 'parallel_perpendicular',
                   'initial_amplitude', 'lesion_threshold', 'direction_angle_degrees']


def parameter_samples(n=32, seed=16026):
    """Latin-hypercube sensitivity design, NOT a posterior or patient estimate."""
    if n < 8:
        raise ValueError('use at least eight paired scenarios')
    u = qmc.LatinHypercube(7, seed=seed).random(n)
    samples = np.column_stack([.5*np.exp(.25*norm.ppf(u[:, 0])),
                               .1*np.exp(.25*norm.ppf(u[:, 1])),
                               .04*np.exp(.20*norm.ppf(u[:, 2])),
                               1.5+2*u[:, 3], .4+.4*u[:, 4], .4+.2*u[:, 5],
                               3*norm.ppf(u[:, 6])])
    return samples


def run_ensemble(pilot, n=32, seed=16026, times=(0, 15, 30, 45, 60, 75, 90), progress=None):
    samples = parameter_samples(n, seed)
    rng = np.random.default_rng(seed+1)
    times = np.asarray(times, float)
    active = np.flatnonzero(pilot.mask)
    states = np.empty((n, 2, len(times), len(active)), np.float32)
    actual_dt = np.empty((n, 2))
    for sample, values in enumerate(samples):
        dw, dg, rho, ratio, amplitude, threshold, angle = values
        diffusion = np.where(pilot.tissue == 3, dw, dg) * pilot.mask
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()
        directions = pilot.directions @ rotation.T
        initial = initial_density(pilot, threshold, amplitude).ravel()[active]
        for model, r in enumerate((1., ratio)):
            tensor = migration_tensor(diffusion, directions, r, pilot.anisotropic)
            matrix, check_active = transport_matrix(tensor, pilot.mask, pilot.spacing)
            if not np.array_equal(active, check_active):
                raise RuntimeError('paired domains differ')
            states[sample, model], actual_dt[sample, model] = simulate(matrix, initial, times, rho, .5)
        if progress:
            progress(sample+1, n)
    return {'states': states, 'active': active, 'samples': samples, 'times': times,
            'dt': actual_dt, 'seed': seed}


def summarize(pilot, network, result, threshold=.16):
    states = result['states']
    voxel_ml = pilot.spacing**3 / 1000
    burden = states.sum(axis=-1, dtype=np.float64) * voxel_ml
    volume = (states >= threshold).sum(axis=-1) * voxel_ml
    atlas = pilot.atlas.ravel()[result['active']]
    exposure = np.array([states[..., atlas == roi].mean(axis=-1)
                         if np.any(atlas == roi) else np.full(states.shape[:3], np.nan)
                         for roi in network['ids']])
    # [scenario, model, time, ROI] = mean simulated normalized density in each ROI.
    exposure = np.moveaxis(exposure, 0, -1)
    return {'burden_ml': burden, 'volume_ml': volume, 'roi_mean_density': exposure,
            'threshold': threshold}


def save_results(pilot, network, result, summary, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'ensemble_cache.npz', **{k:v for k,v in result.items()},
                        **summary, fc=network['fc'], roi_ids=network['ids'])
    provenance = dict(pilot.provenance)
    provenance['dti_history'] = [s.split()[0] for s in provenance['dti_history']]
    provenance['fMRI_QC'] = network['qc']
    provenance['software_sha256'] = {name: sha256(Path(__file__).with_name(name))
                                     for name in ['pat16.py', 'anisotropic.py', 'pat16_experiment.py']}
    provenance['numpy_version'] = np.__version__
    provenance['software_versions'] = {name: version(name) for name in
                                      ['numpy', 'scipy', 'nibabel', 'numba', 'matplotlib', 'ipywidgets', 'nbclient']}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    def interval(values):
        return np.quantile(values, [.05, .5, .95], axis=0).tolist()
    report = {
        'scope': 'Conditional untreated 90-day simulations from a pre-operative lesion; no growth-rate fitting or clinical validation',
        'uncertainty': '5th/50th/95th empirical prior-sensitivity quantiles; not confidence or posterior credible intervals',
        'scenario_count': len(result['samples']), 'seed': result['seed'], 'times_days': result['times'].tolist(),
        'models': ['isotropic', 'anisotropic'], 'density_threshold': summary['threshold'],
        'parameter_names': PARAMETER_NAMES, 'parameter_samples': result['samples'].tolist(),
        'burden_ml_q05_q50_q95': interval(summary['burden_ml']),
        'threshold_volume_ml_q05_q50_q95': interval(summary['volume_ml']),
        'paired_final_burden_difference_ml_q05_q50_q95': interval(summary['burden_ml'][:, 1, -1]-summary['burden_ml'][:, 0, -1]),
        'paired_final_volume_difference_ml_q05_q50_q95': interval(summary['volume_ml'][:, 1, -1]-summary['volume_ml'][:, 0, -1]),
        'min_density': float(result['states'].min()), 'max_density': float(result['states'].max()),
        'actual_dt_range_days': [float(result['dt'].min()), float(result['dt'].max())],
        'fMRI_QC': network['qc'],
    }
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    import csv
    with (output/'regional_exposure.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['roi_id', 'roi_name', 'model', 'day', 'mean_density_q05', 'mean_density_q50', 'mean_density_q95'])
        for roi, (roi_id, name) in enumerate(zip(network['ids'], network['names'])):
            for model, label in enumerate(report['models']):
                for time, day in enumerate(result['times']):
                    values = summary['roi_mean_density'][:, model, time, roi]
                    writer.writerow([roi_id, name, label, day, *np.quantile(values, [.05, .5, .95])])
    return report


def grid_field(pilot, active, values):
    field = np.zeros(pilot.mask.shape, np.float32)
    field.ravel()[active] = values
    return field


def slice_image(pilot, field, axis, index):
    axes = [i for i in range(3) if i != axis]
    image = np.take(field, index, axis=axis).T
    origin = pilot.affine[:3, 3]
    extent = []
    for i in axes:
        extent += [origin[i]-.5*pilot.spacing,
                   origin[i]+(pilot.mask.shape[i]-.5)*pilot.spacing]
    return image, extent, axes


def network_overlay(ax, pilot, network, axis, index, slab_mm=12., max_edges=35):
    """Static measured FC projected through a labeled slab, never evolved FC."""
    axes = [i for i in range(3) if i != axis]
    centers = []
    for roi in network['ids']:
        coordinates = np.argwhere(pilot.atlas == roi)
        centers.append(nib_world(pilot, coordinates.mean(axis=0)) if len(coordinates) else [np.nan]*3)
    centers = np.asarray(centers)
    position = pilot.affine[axis, 3] + index*pilot.spacing
    visible = np.abs(centers[:, axis]-position) <= slab_mm
    fc = network['fc']
    i, j = np.triu_indices(len(fc), 1)
    use = visible[i] & visible[j] & (np.abs(fc[i, j]) >= .35)
    i, j = i[use], j[use]
    order = np.argsort(np.abs(fc[i, j]))[-max_edges:]
    for a, b in zip(i[order], j[order]):
        ax.plot(centers[[a, b], axes[0]], centers[[a, b], axes[1]],
                color='#30d5c8' if fc[a, b] > 0 else '#b5a0ff', alpha=.5,
                linewidth=.4+1.2*abs(fc[a, b]), zorder=5)
    ax.scatter(centers[visible, axes[0]], centers[visible, axes[1]], s=9,
               facecolors='none', edgecolors='#6ee7e0', linewidths=.6, zorder=6)


def nib_world(pilot, voxel):
    return pilot.affine[:3, :3] @ voxel + pilot.affine[:3, 3]


def comparison_figure(pilot, network, result, frame=-1, axis=2, index=None,
                      quantile=.5, overlay=True, zoom=True, columns=4):
    import matplotlib.pyplot as plt
    if index is None:
        index = int(np.round(np.argwhere(pilot.lesion > .5).mean(axis=0)[axis]))
    states = result['states'][:, :, frame]
    iso, ani = np.quantile(states, quantile, axis=0)
    difference = np.median(states[:, 1]-states[:, 0], axis=0)
    probability = np.mean(states[:, 1] >= .16, axis=0)
    values = [iso, ani, difference, probability]
    titles = [f'Isotropic • q={quantile:g}', f'Anisotropic • q={quantile:g}',
              'Paired median: anisotropic − isotropic', 'Prior ensemble: P(u ≥ 0.16)\nanisotropic']
    cmaps, lows, highs = ['inferno', 'inferno', 'RdBu_r', 'viridis'], [0,0,-.15,0], [1,1,.15,1]
    if columns not in (2, 4):
        raise ValueError('use two or four panel columns')
    fig, axes = plt.subplots(4//columns, columns,
                             figsize=(19, 5.4) if columns == 4 else (12, 10), layout='constrained')
    axes = np.asarray(axes).ravel()
    background, extent, plane_axes = slice_image(pilot, pilot.t1, axis, index)
    lesion, _, _ = slice_image(pilot, pilot.lesion, axis, index)
    for k, (ax, value) in enumerate(zip(axes, values)):
        ax.imshow(background, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=1)
        image, _, _ = slice_image(pilot, grid_field(pilot, result['active'], value), axis, index)
        masked = np.ma.masked_where(abs(image) < (.005 if k == 2 else .02), image)
        artist = ax.imshow(masked, origin='lower', extent=extent, cmap=cmaps[k],
                           vmin=lows[k], vmax=highs[k], alpha=.85)
        if lesion.max() > .5:
            ax.contour(lesion, levels=[.5], colors=['#52ff8f'], linewidths=.7,
                       origin='lower', extent=extent)
        if overlay and k < 2:
            network_overlay(ax, pilot, network, axis, index)
        ax.set_title(titles[k], fontsize=10)
        ax.set_xlabel(['R (mm)', 'A (mm)', 'S (mm)'][plane_axes[0]])
        ax.set_ylabel(['R (mm)', 'A (mm)', 'S (mm)'][plane_axes[1]])
        if zoom:
            center = nib_world(pilot, np.argwhere(pilot.lesion > .5).mean(axis=0))
            ax.set_xlim(center[plane_axes[0]]-52, center[plane_axes[0]]+52)
            ax.set_ylim(center[plane_axes[1]]-55, center[plane_axes[1]]+55)
        fig.colorbar(artist, ax=ax, shrink=.65, label='normalized density' if k < 2 else ('Δu' if k == 2 else 'fraction of scenarios'))
    position = pilot.affine[axis, 3]+index*pilot.spacing
    fig.suptitle(f'PAT16 pre-op anatomy • conditional day {result["times"][frame]:g} • '
                 f'{["sagittal", "coronal", "axial"][axis]} {position:g} mm\n'
                 'Green: initial lesion contour. Cyan/purple: measured positive/negative FC in a ±12 mm slab. '
                 'No functional-loss prediction.', fontsize=10)
    return fig


def uncertainty_figure(result, summary, network):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout='constrained')
    for key, ax, title in [('burden_ml', axes[0], 'Integral of normalized density (mL)'),
                           ('volume_ml', axes[1], 'Simulated volume u ≥ 0.16 (mL)')]:
        for model, (label, color) in enumerate([('isotropic', '#2979b8'), ('anisotropic', '#c65d1c')]):
            low, med, high = np.quantile(summary[key][:, model], [.05, .5, .95], axis=0)
            ax.fill_between(result['times'], low, high, color=color, alpha=.16)
            ax.plot(result['times'], med, color=color, label=label)
        ax.set(xlabel='Conditional time (days)', ylabel=title); ax.legend()
    exposure = summary['roi_mean_density'][:, :, -1]
    order = np.argsort(np.nanmedian(exposure, axis=0).mean(axis=0))[-9:]
    for model, (label, color, offset) in enumerate([('isotropic', '#2979b8', -.12), ('anisotropic', '#c65d1c', .12)]):
        low, med, high = np.quantile(exposure[:, model, order], [.05,.5,.95], axis=0)
        axes[2].errorbar(med, np.arange(len(order))+offset, xerr=[med-low, high-med],
                         fmt='o', ms=4, color=color, label=label)
    axes[2].set(yticks=np.arange(len(order)), yticklabels=np.asarray(network['names'])[order],
                xlabel='Mean simulated normalized density in ROI', title='Regions with greatest simulated overlap at day 90')
    fig.suptitle(f'{len(result["samples"])} paired prior-sensitivity scenarios • median and 5th–95th percentiles\n'
                 'Assumed parameter distributions; no patient calibration or functional-risk estimates.', fontsize=11)
    return fig
