"""Build PAT16 recovery manuscript assets from verified saved inference records.

No MRI access or parameter refitting is required. Recompute every tabulated
sampling/forecast quantity from the arrays before generating tables and figures.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, norm

from make_recovery_figures import rows_file, scientific

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORDS = ROOT/'outputs/pat16_recovery'
FIG, GEN = HERE/'figures', HERE/'generated'
LABELS = [r'$D_{\rm WM}$', r'$D_{\rm GM}$', r'$\rho$', r'$r$']
NAMES = [r'$D_{\rm WM}$ (mm$^2$/day)', r'$D_{\rm GM}$ (mm$^2$/day)',
         r'$\rho$ (day$^{-1}$)', r'$r$ (dimensionless)']


def load_records():
    provenance = json.loads((RECORDS/'provenance.json').read_text())
    for name, expected in provenance['software_sha256'].items():
        actual = hashlib.sha256((ROOT/'src/brainweb_kpp'/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Stale PAT16 recovery code: {name}; rerun the audit')
    for name, expected in provenance['outputs_sha256'].items():
        if hashlib.sha256((RECORDS/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Stale PAT16 recovery record: {name}')
    summary = json.loads((RECORDS/'summary.json').read_text())
    with np.load(RECORDS/'recovery_data.npz', allow_pickle=False) as archive:
        z = {name: archive[name] for name in archive.files}
    s, rows = summary['settings'], summary['parameters']
    true, estimate = z['known_parameters'], z['estimate']
    # These settings are described explicitly in the manuscript prose.
    np.testing.assert_array_equal(true, [.5, .1, .04, 2.5])
    np.testing.assert_array_equal(z['scan_days'], [20, 40, 60, 75, 90])
    assert int(z['training_scans']) == s['training_scans'] == 3
    assert s['confidence'] == .95 and s['noise_sd'] == .015
    assert s['n_bootstrap'] == 100 and s['n_coverage'] == 60 and s['seed'] == 161026
    np.testing.assert_array_equal(summary['bounds'], [[.05, .02, .01, 1.], [1., .6, .09, 3.5]])
    np.testing.assert_array_equal(s['starts'], [[.25, .20, .025, 1.5], [.40, .15, .035, 2.], [.80, .05, .065, 3.2]])
    assert provenance['anatomy']['spacing_mm'] == 3
    assert provenance['anatomy']['active_voxels'] == 44667
    np.testing.assert_array_equal(provenance['anatomy']['grid_shape'], [51, 62, 50])
    assert summary['design']['observation_voxels'] == 12
    assert summary['design']['voxel_mm'] == 6 and summary['design']['blur_sigma_mm'] == 2
    assert summary['design']['quadrature_points'] == 3
    assert summary['design']['lesion_threshold'] == .5 and summary['design']['initial_amplitude'] == .6
    assert summary['generator_dt_days'] == .25 and summary['inverse_dt_days'] == .5
    assert s['near_center_radius_mm'] == 24
    np.testing.assert_array_equal(true, s['known_parameters'])
    np.testing.assert_allclose(estimate, [row['estimate'] for row in rows])
    local, boot = z['local_intervals'], z['bootstrap_intervals']
    np.testing.assert_allclose(local, [row['local_interval'] for row in rows])
    np.testing.assert_allclose(boot, [row['bootstrap_interval'] for row in rows])
    sd = np.sqrt(np.diag(z['log_covariance']))
    np.testing.assert_allclose(local, np.exp(np.log(estimate)[:, None]+norm.ppf(.975)*sd[:, None]*[-1, 1]))
    np.testing.assert_allclose(z['correlation'], z['log_covariance']/np.outer(sd, sd))
    np.testing.assert_allclose(z['correlation'], summary['log_parameter_correlation'])
    np.testing.assert_allclose(z['standardized_residuals'],
                               ((z['predicted'][:3]-z['observed'][:3])/s['noise_sd']).ravel())
    np.testing.assert_allclose(np.sqrt(np.mean(z['standardized_residuals']**2)),
                               summary['training_standardized_residual_rms'])
    assert z['clean'].shape == z['observed'].shape == z['predicted'].shape == (5, 12, 12, 12)
    starts = summary['multistart']
    assert len(starts) == 3 and not summary['at_bound'] and not any(row['at_bound'] for row in starts)
    np.testing.assert_array_equal([row['start'] for row in starts], s['starts'])
    best = min(starts, key=lambda row: row['sum_squared_standardized_residuals'])
    np.testing.assert_array_equal(estimate, best['estimate'])
    np.testing.assert_allclose(np.sum(z['standardized_residuals']**2), best['sum_squared_standardized_residuals'])
    np.testing.assert_allclose([row['estimate'] for row in starts], np.broadcast_to(estimate, (3, 4)), rtol=1e-5)
    # Reconstruct both noise streams independently of the saved interval arrays.
    streams = np.random.SeedSequence(s['seed']).spawn(4)
    noise_rng, forecast_rng = (np.random.default_rng(streams[i]) for i in [0, 3])
    np.testing.assert_array_equal(z['observed'], z['clean']+s['noise_sd']*noise_rng.standard_normal(z['clean'].shape))
    valid_boot = np.all(np.isfinite(z['bootstrap_parameters']), axis=1)
    valid = np.all(np.isfinite(z['noise_trial_parameters']), axis=1)
    np.testing.assert_array_equal(valid_boot, z['bootstrap_valid'])
    np.testing.assert_array_equal(valid, z['noise_trial_valid'])
    assert len(valid_boot) == s['n_bootstrap'] and len(valid) == s['n_coverage']
    assert valid_boot.sum() == summary['bootstrap_successful']
    assert valid.sum() == summary['coverage_successful']
    assert z['bootstrap_boundary'].sum() == summary['bootstrap_boundary_fits']
    assert z['noise_trial_boundary'].sum() == summary['coverage_boundary_fits']
    assert len(summary['failures']['bootstrap']) == (~valid_boot).sum()
    assert len(summary['failures']['coverage']) == (~valid).sum()
    assert valid_boot.all() and valid.all()
    assert not z['bootstrap_boundary'].any() and not z['noise_trial_boundary'].any()
    np.testing.assert_allclose(boot, np.quantile(z['bootstrap_parameters'][valid_boot], [.025, .975], axis=0).T)
    np.testing.assert_allclose(z['signal_interval'], np.quantile(z['bootstrap_predictions'], [.025, .975], axis=0))
    alpha = (1-s['confidence'])/2
    noisy_predictions = z['bootstrap_predictions']+s['noise_sd']*forecast_rng.standard_normal(z['bootstrap_predictions'].shape)
    np.testing.assert_allclose(z['prediction_interval'], np.quantile(noisy_predictions, [alpha, 1-alpha], axis=0),
                               rtol=1e-13, atol=2e-15)
    usable = valid & ~z['noise_trial_boundary']
    ci = z['noise_trial_intervals']
    hits = ((ci[:, :, 0] <= true) & (true <= ci[:, :, 1]) & usable[:, None]).sum(axis=0)
    np.testing.assert_array_equal(hits, z['coverage_hits'])
    assert usable.sum() == summary['coverage_usable_local_intervals']
    for k, row in enumerate(rows):
        recovered = z['noise_trial_parameters'][valid, k]
        assert row['known'] == true[k]
        assert row['local_contains_known'] == bool(local[k, 0] <= true[k] <= local[k, 1])
        assert row['bootstrap_contains_known'] == bool(boot[k, 0] <= true[k] <= boot[k, 1])
        np.testing.assert_allclose(row['relative_error_percent'], 100*(estimate[k]/true[k]-1))
        np.testing.assert_allclose(row['noise_trial_mean'], recovered.mean())
        np.testing.assert_allclose(row['noise_trial_bias_percent'], 100*(recovered.mean()/true[k]-1))
        np.testing.assert_allclose(row['noise_trial_rmse'], np.sqrt(np.mean((recovered-true[k])**2)))
        assert hits[k] == row['local_coverage_hits'] and row['local_coverage_trials'] == len(valid)
        np.testing.assert_allclose(row['local_coverage_fraction'], hits[k]/len(valid))
        binomial_ci = binomtest(int(hits[k]), len(valid)).proportion_ci(method='exact')
        np.testing.assert_allclose(row['local_coverage_binomial_interval'], binomial_ci)
        np.testing.assert_allclose(z['coverage_binomial_intervals'][k], binomial_ci)
    coordinates = np.stack(np.meshgrid(*z['observation_axes_mm'], indexing='ij'), axis=-1)
    near = np.linalg.norm(coordinates-z['observation_center_mm'], axis=-1) <= 24
    np.testing.assert_array_equal(near, z['near_center'])
    assert z['near_center'].sum() == 280
    for key, mask in [('forecast_full_region', np.ones(near.shape, bool)), ('forecast_near_center', near)]:
        assert len(summary[key]) == 2
        for index, row in zip(range(3, 5), summary[key]):
            assert row['day'] == z['scan_days'][index] and row['voxels'] == mask.sum()
            truth, obs, pred = (z[name][index][mask] for name in ['clean', 'observed', 'predicted'])
            signal, prediction = (z[name][:, index][:, mask] for name in ['signal_interval', 'prediction_interval'])
            checks = {
                'truth_rmse': np.sqrt(np.mean((pred-truth)**2)),
                'observation_rmse_over_noise_sd': np.sqrt(np.mean((pred-obs)**2))/s['noise_sd'],
                'signal_interval_truth_fraction': np.mean((signal[0] <= truth) & (truth <= signal[1])),
                'prediction_interval_observation_fraction': np.mean((prediction[0] <= obs) & (obs <= prediction[1])),
                'mean_signal_interval_width': np.mean(signal[1]-signal[0]),
                'mean_prediction_interval_width': np.mean(prediction[1]-prediction[0]),
            }
            for name, value in checks.items():
                np.testing.assert_allclose(row[name], value)
    numerical = summary['numerical_validation']
    np.testing.assert_allclose(np.max(np.abs(np.array(numerical['matched_model_noiseless_estimate'])/true-1)),
                               numerical['matched_model_max_relative_error'])
    assert numerical['matched_model_max_relative_error'] < 1e-4
    np.testing.assert_allclose(100*(np.array(numerical['finer_time_noiseless_estimate'])/true-1),
                               numerical['finer_time_relative_parameter_shift_percent'])
    np.testing.assert_allclose(np.sqrt(np.mean((z['clean']-z['matched_model_at_truth'])**2, axis=(1, 2, 3))),
                               numerical['generator_minus_inverse_rmse_by_scan'])
    assert 'not locally identifiable' in numerical['time_zero_negative_control']
    singular = np.array(summary['jacobian_singular_values'])
    np.testing.assert_allclose(singular[0]/singular[-1], summary['jacobian_condition_number'])
    return summary, z


def tables(summary, z):
    rows, settings = summary['parameters'], summary['settings']
    macros = {'PRBootstraps': str(settings['n_bootstrap']), 'PRTrials': str(settings['n_coverage']),
              'PRCondition': f"{summary['jacobian_condition_number']:.2f}",
              'PRWMRatioCorrelation': f"{z['correlation'][0, 3]:+.3f}",
              'PRMatchedError': scientific(summary['numerical_validation']['matched_model_max_relative_error']),
              'PRTemporalMaxShift': f"{max(abs(v) for v in summary['numerical_validation']['finer_time_relative_parameter_shift_percent']):.3f}",
              'PRTrainingRMS': f"{summary['training_standardized_residual_rms']:.3f}",
              'PRSingularValues': ', '.join(f'{v:.2f}' for v in summary['jacobian_singular_values']),
              'PRProfileY': f"{z['observation_axes_mm'][1, 6]:.2f}",
              'PRProfileZ': f"{z['observation_axes_mm'][2, 6]:.2f}"}
    for key, row, shift in zip(['White', 'Gray', 'Rho', 'Ratio'], rows,
                               summary['numerical_validation']['finer_time_relative_parameter_shift_percent']):
        macros[f'PR{key}Estimate'] = f"{row['estimate']:.6f}"
        macros[f'PR{key}Coverage'] = str(row['local_coverage_hits'])
        macros[f'PR{key}Error'] = f"{row['relative_error_percent']:+.2f}"
        macros[f'PR{key}Bias'] = f"{row['noise_trial_bias_percent']:+.3f}"
        macros[f'PR{key}TemporalShift'] = f'{shift:+.3f}'
    for key, full, near in zip(['SeventyFive', 'Ninety'], summary['forecast_full_region'], summary['forecast_near_center']):
        macros[f'PRDay{key}RMSE'] = scientific(full['truth_rmse'])
        macros[f'PRDay{key}SignalContainment'] = f"{100*full['signal_interval_truth_fraction']:.1f}"
        macros[f'PRDay{key}PredictionContainment'] = f"{100*full['prediction_interval_observation_fraction']:.1f}"
        macros[f'PRNearDay{key}PredictionContainment'] = f"{100*near['prediction_interval_observation_fraction']:.1f}"
        macros[f'PRDay{key}SignalWidth'] = scientific(full['mean_signal_interval_width'])
        macros[f'PRDay{key}PredictionWidth'] = f"{full['mean_prediction_interval_width']:.5f}"
    (GEN/'pat16_recovery_numbers.tex').write_text('% Generated from verified PAT16 recovery records.\n'+
        '\n'.join(r'\newcommand{\%s}{%s}' % (key, value) for key, value in macros.items())+'\n')
    rows_file('pat16_recovery_rows.tex', 'PRRecoveryRows', [
        f"{name} & {r['known']:.6f} & {r['estimate']:.6f} & "
        f"[{r['local_interval'][0]:.6f}, {r['local_interval'][1]:.6f}] & "
        f"[{r['bootstrap_interval'][0]:.6f}, {r['bootstrap_interval'][1]:.6f}] "+r'\\'
        for name, r in zip(NAMES, rows)])
    rows_file('pat16_recovery_coverage_rows.tex', 'PRCoverageRows', [
        f"{name} & {r['noise_trial_bias_percent']:+.3f}\\% & {scientific(r['noise_trial_rmse'])} & "
        f"{r['local_coverage_hits']}/{r['local_coverage_trials']} ({100*r['local_coverage_fraction']:.1f}\\%) & "
        f"[{100*r['local_coverage_binomial_interval'][0]:.2f}, {100*r['local_coverage_binomial_interval'][1]:.2f}]\\% "+r'\\'
        for name, r in zip(NAMES, rows)])
    rows_file('pat16_recovery_forecast_rows.tex', 'PRForecastRows', [
        f"{row['day']:g} & {label} & {scientific(row['truth_rmse'])} & "
        f"{row['observation_rmse_over_noise_sd']:.3f} & {100*row['signal_interval_truth_fraction']:.1f}\\% & "
        f"{100*row['prediction_interval_observation_fraction']:.1f}\\% "+r'\\'
        for time in range(2) for key, label in [('forecast_full_region', 'Full FOV'), ('forecast_near_center', 'Near center')]
        for row in [summary[key][time]]])


def claims_audit(summary):
    """Retain a compact review record without modifying the inference results."""
    provenance = json.loads((RECORDS/'provenance.json').read_text())
    audit = {
        'scope': 'Manuscript claims checked against saved PAT16 arrays and the recorded inference implementation; no refitting.',
        'source_sha256': provenance['software_sha256'],
        'record_sha256': {name: provenance['outputs_sha256'][name] for name in ['summary.json', 'recovery_data.npz', 'parameters.csv']},
        'verified': ['fixed design, units and bounds', 'primary-noise stream and multistart selection',
                     'known-noise log-Wald covariance/intervals and correlations', 'bootstrap marginal/signal quantiles',
                     'predictive-noise stream and prediction quantiles', 'all fit failures and boundary flags',
                     'marginal coverage denominators and exact binomial intervals', 'trial bias and RMSE',
                     'physical evaluation regions', 'both regional forecast errors, containment and widths',
                     'matched-model recovery and temporal-mismatch shifts'],
        'parameters': summary['parameters'],
        'forecast_full_region': summary['forecast_full_region'],
        'forecast_near_center': summary['forecast_near_center'],
        'interpretation': {
            'parameters': 'Separate tissue diffusivities with joint uncertainty; independence is not imposed or established.',
            'intervals': 'Nominal marginal/pointwise 95% approximations; neither joint 95% parameter coverage nor simultaneous image coverage was assessed.',
            'coverage': '60 noise trials audit local intervals only; broad binomial intervals do not establish calibration. Marginal bootstrap parameter-interval coverage is unaudited.',
            'forecasts': 'Final prediction intervals integrate Gaussian noise over bootstrap forecast means. A separate 40-dataset nested audit assesses pointwise predictive coverage; neither spatial fractions nor this fixed-design experiment establish simultaneous or clinical coverage.',
            'density_scale': 'Recovery uses u=c/K. Uniform K cancels from the normalized dynamics; neither absolute capacity nor cell counts are identified.',
            'validation': 'Known-value recovery on fixed patient-derived anatomy; no patient growth-rate or biological validation.',
        },
    }
    if 'prediction_audit' in summary:
        audit['prediction_summary_sha256'] = hashlib.sha256((ROOT/'outputs/pat16_prediction/summary.json').read_bytes()).hexdigest()
    (HERE/'data/pat16_recovery_claims_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')


def save_figure(fig, name):
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.png')
    plt.close(fig)


def uncertainty_figure(summary, z):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.7), layout='constrained')
    true, estimate = z['known_parameters'], z['estimate']
    ax = axes[0, 0]
    for intervals, offset, label, color in [
            (z['local_intervals'], -.12, 'Local 95% CI', '#2978a0'),
            (z['bootstrap_intervals'], .12, 'Bootstrap 95% CI', '#bf5b21')]:
        relative = intervals/true[:, None]
        ax.hlines(np.arange(4)+offset, relative[:, 0], relative[:, 1], color=color, lw=1.4, label=label)
        ax.plot(estimate/true, np.arange(4)+offset, 'o', color=color, ms=3)
    ax.axvline(1, color='black', ls='--', lw=.8)
    ax.set(yticks=np.arange(4), yticklabels=LABELS, xlabel='Estimate / known value', title='a. Four-parameter recovery')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.24), ncol=2, fontsize=6)
    ax = axes[0, 1]
    artist = ax.imshow(z['correlation'], vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set(xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS,
           title='b. Local log-parameter correlation')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{z['correlation'][i, j]:.2f}", ha='center', va='center', fontsize=7,
                    color='white' if abs(z['correlation'][i, j]) > .6 else 'black')
    fig.colorbar(artist, ax=ax, shrink=.8, pad=.025)
    ax = axes[1, 0]
    boot = z['bootstrap_parameters'][z['bootstrap_valid']]
    ax.scatter(boot[:, 0], boot[:, 3], s=9, alpha=.6, color='#2978a0', label='Bootstrap fits')
    ax.scatter(true[0], true[3], marker='*', s=55, color='black', label='Known')
    ax.scatter(estimate[0], estimate[3], marker='+', s=45, color='#bf5b21', label='Estimate')
    ax.set(xlabel=r'$D_{\rm WM}$ (mm$^2$/day)', ylabel=r'$r=D_\parallel/D_\perp$',
           title='c. WM migration-anisotropy dependence')
    ax.legend(fontsize=6)
    ax = axes[1, 1]
    line = (slice(None), 6, 6)
    x = z['observation_axes_mm'][0]
    ax.fill_between(x, z['prediction_interval'][0, -1][line], z['prediction_interval'][1, -1][line],
                    color='#8dbbd3', alpha=.4, label='New-scan 95% PI')
    ax.fill_between(x, z['signal_interval'][0, -1][line], z['signal_interval'][1, -1][line],
                    color='#2978a0', alpha=.7, label='Signal 95% CI')
    ax.plot(x, z['clean'][-1][line], color='black', lw=.9, label='Known signal')
    ax.plot(x, z['predicted'][-1][line], '--', color='#bf5b21', lw=.9, label='Forecast')
    ax.scatter(x, z['observed'][-1][line], color='black', s=8, label='Withheld data')
    ax.set(xlabel='RAS x (mm)', ylabel='Normalized signal', title='d. Withheld day-90 profile')
    ax.legend(fontsize=5.4, loc='lower left')
    save_figure(fig, 'pat16_recovery')


def validation_figure(summary, z):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8), layout='constrained')
    for k, ax in enumerate(axes.flat):
        valid = z['noise_trial_valid']
        recovered, ci = z['noise_trial_parameters'][valid], z['noise_trial_intervals'][valid]
        bound = z['noise_trial_boundary'][valid]
        for row, index in enumerate(np.argsort(recovered[:, k])):
            inside = ci[index, k, 0] <= z['known_parameters'][k] <= ci[index, k, 1] and not bound[index]
            ax.plot(ci[index, k], [row, row], color='#2978a0' if inside else '#c63d32', lw=.7)
        ax.axvline(z['known_parameters'][k], color='black', ls='--', lw=.9)
        ax.set(title=f"{'abcd'[k]}. Local CI coverage: {int(z['coverage_hits'][k])}/{summary['settings']['n_coverage']}",
               xlabel=NAMES[k], ylabel='Independent trial (sorted)')
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        ax.locator_params(axis='x', nbins=3)
    save_figure(fig, 'pat16_recovery_validation')


def forecast_figure(summary, z):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.5), layout='constrained')
    coordinates, plane = z['observation_axes_mm'], 6
    extent = [coordinates[0, 0]-3, coordinates[0, -1]+3, coordinates[1, 0]-3, coordinates[1, -1]+3]
    difference = z['predicted'][3:]-z['clean'][3:]
    limit = np.max(np.abs(difference[:, :, :, plane]))
    for row, index in enumerate((3, 4)):
        for col, title in enumerate(['Known signal', 'Fitted forecast', 'Forecast - known']):
            values = [z['clean'][index], z['predicted'][index], difference[row]][col][:, :, plane].T
            artist = axes[row, col].imshow(values, origin='lower', extent=extent, interpolation='nearest',
                                          cmap='RdBu_r' if col == 2 else 'inferno',
                                          vmin=-limit if col == 2 else 0, vmax=limit if col == 2 else 1)
            axes[row, col].set(title=f"{'abcdef'[row*3+col]}. Day {z['scan_days'][index]:g}\n{title}",
                               xlabel='RAS x (mm)', ylabel='RAS y (mm)')
            fig.colorbar(artist, ax=axes[row, col], shrink=.8, pad=.025)
    save_figure(fig, 'pat16_recovery_forecast')


def main():
    summary, arrays = load_records()
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
                         'legend.fontsize': 6, 'pdf.fonttype': 42, 'savefig.dpi': 240,
                         'axes.spines.top': False, 'axes.spines.right': False})
    from make_pat16_prediction_figures import verify_records, overlay_primary, generate_assets
    audit, primary = verify_records(summary, arrays)
    overlay_primary(summary, arrays, audit, primary)
    generate_assets(audit, primary)
    tables(summary, arrays)
    claims_audit(summary)
    uncertainty_figure(summary, arrays)
    validation_figure(summary, arrays)
    forecast_figure(summary, arrays)
    print('Verified PAT16 recovery and predictive audits; generated four vector figures and four tables.')


if __name__ == '__main__':
    main()
