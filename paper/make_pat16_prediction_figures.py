"""Verify nested PAT16 predictive records and generate manuscript assets."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from make_recovery_figures import rows_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'src'))
from brainweb_kpp.pat16_prediction import (interval_metrics, mixture_interval,
                                          predictive_interval, summarize_trials)

RECORDS = ROOT/'outputs/pat16_prediction'
METHODS = ['100 noisy draws', '20,000 noisy draws', 'Gaussian mixture']
COLORS = ['#a85432', '#387b9b', '#27664b']


def verify_records(baseline, z):
    """Check provenance, independent noise streams, interval CDFs and every metric."""
    s = json.loads((RECORDS/'summary.json').read_text())
    assert s['settings'] == dict(trials=40, bootstrap=100, predictive_draws=20000, seed=161027)
    assert s['noise_sd'] == .015 and s['withheld_days'] == [75., 90.]
    assert s['design'] == baseline['design']
    anatomy_file = RECORDS/'anatomy_validation.json'
    assert hashlib.sha256(anatomy_file.read_bytes()).hexdigest() == s['anatomy_validation_sha256']
    anatomy = json.loads(anatomy_file.read_text())
    baseline_provenance = ROOT/'outputs/pat16_recovery/provenance.json'
    assert anatomy['baseline_provenance_sha256'] == hashlib.sha256(baseline_provenance.read_bytes()).hexdigest()
    assert anatomy['anatomy'] == json.loads(baseline_provenance.read_text())['anatomy']
    assert anatomy['runner_sha256'] == hashlib.sha256((ROOT/'scripts/run_pat16_prediction.py').read_bytes()).hexdigest()
    assert anatomy['known_signal_max_absolute_difference'] == 0.
    for name, expected in s['source_sha256'].items():
        assert hashlib.sha256((ROOT/'src/brainweb_kpp'/name).read_bytes()).hexdigest() == expected, name
    for name, expected in s['baseline_sha256'].items():
        assert hashlib.sha256((ROOT/'outputs/pat16_recovery'/name).read_bytes()).hexdigest() == expected, name
    for name, expected in s['output_sha256'].items():
        assert hashlib.sha256((RECORDS/name).read_bytes()).hexdigest() == expected, name
    fingerprint = {key: s[key] for key in ['settings', 'design', 'source_sha256', 'baseline_sha256']}
    assert hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest() == s['fingerprint']
    with np.load(RECORDS/'primary.npz') as archive:
        primary = {name: archive[name] for name in archive.files}
    means, sd = z['bootstrap_predictions'][:, 3:], s['noise_sd']
    seeds = np.random.SeedSequence([s['settings']['seed'], 0]).spawn(2)
    np.testing.assert_array_equal(primary['intervals'][0], z['prediction_interval'][:, 3:])
    np.testing.assert_array_equal(primary['intervals'][1], predictive_interval(means, sd, 1000, seeds[0]))
    np.testing.assert_array_equal(primary['intervals'][2], predictive_interval(means, sd, 20000, seeds[1]))
    np.testing.assert_array_equal(primary['intervals'][3], mixture_interval(means, sd))
    rmse = np.sqrt(np.mean((primary['intervals']-primary['intervals'][3])**2, axis=(1, 2, 3, 4, 5)))
    np.testing.assert_array_equal(rmse, s['primary_endpoint_rmse'])
    masks = [np.ones(z['near_center'].shape, bool), z['near_center']]
    for m, pi in enumerate(primary['intervals']):
        for r, mask in enumerate(masks):
            metrics = interval_metrics(pi, z['observed'][3:], z['clean'][3:], sd, mask)
            np.testing.assert_array_equal(primary['metrics'][m, r], metrics)
    np.testing.assert_array_equal(primary['metrics'], s['primary_metrics'])
    for i, record in enumerate(s['records']):
        assert record == json.loads((RECORDS/f'trials/trial_{i:03d}.json').read_text())
        assert record['fingerprint'] == s['fingerprint']
        path = RECORDS/f'trials/trial_{i:03d}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with np.load(path) as trial:
            streams = np.random.SeedSequence([s['settings']['seed'], i, 1]).spawn(5)
            train_rng, test_rng = [np.random.default_rng(streams[j]) for j in [0, 2]]
            np.testing.assert_array_equal(trial['training_observed'], z['clean'][:3]+sd*train_rng.standard_normal(z['clean'][:3].shape))
            np.testing.assert_array_equal(trial['future_observed'], z['clean'][3:]+sd*test_rng.standard_normal(z['clean'][3:].shape))
            assert trial['bootstrap_valid'].sum() == record['bootstrap_successful']
            assert trial['bootstrap_boundary'].sum() == record['bootstrap_boundary']
            assert record['bootstrap_successful'] == 100 and not record['failures']
            assert record['usable'] and not record['primary_boundary'] and not record['primary_failed']
            forecasts, intervals = trial['bootstrap_predictions'], trial['intervals']
            old_rng = np.random.default_rng(streams[3])
            np.testing.assert_array_equal(intervals[0], np.quantile(forecasts+sd*old_rng.standard_normal(forecasts.shape), [.025, .975], axis=0))
            # Verify exact-mixture endpoints by their CDF rather than refitting MRI.
            probabilities = ndtr((intervals[2][:, None]-forecasts[None])/sd).mean(axis=1)
            np.testing.assert_allclose(probabilities[0], .025, atol=2e-9)
            np.testing.assert_allclose(probabilities[1], .975, atol=2e-9)
            for method in range(3):
                for region, mask in enumerate(masks):
                    expected = interval_metrics(intervals[method], trial['future_observed'], z['clean'][3:], sd, mask)
                    np.testing.assert_array_equal(trial['metrics'][method, region], expected)
                point = (slice(None), 6, 6, 6)
                low, high = intervals[method, 0][point], intervals[method, 1][point]
                np.testing.assert_array_equal(trial['sentinel_hits'][method],
                                              (low <= trial['future_observed'][point]) & (trial['future_observed'][point] <= high))
    coverage, sentinel = summarize_trials(s['records'], RECORDS/'trials', s['withheld_days'])
    assert coverage == s['coverage'] and sentinel == s['sentinel']
    assert s['usable_trials'] == 40 and not s['primary_boundary_trials']
    assert not s['bootstrap_failures'] and not s['bootstrap_boundary_fits']
    return s, primary


def overlay_primary(summary, arrays, audit, primary):
    """Use improved intervals for final forecasts after validating legacy records."""
    arrays['prediction_interval'] = arrays['prediction_interval'].copy()
    arrays['prediction_interval'][:, 3:] = primary['intervals'][3]
    for region, key in enumerate(['forecast_full_region', 'forecast_near_center']):
        for time, row in enumerate(summary[key]):
            row['prediction_interval_observation_fraction'] = float(primary['metrics'][3, region, time, 0])
            row['mean_prediction_interval_width'] = float(primary['metrics'][3, region, time, 1])
    summary['prediction_audit'] = audit


def generate_assets(s, primary):
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
                         'legend.fontsize': 6, 'pdf.fonttype': 42, 'savefig.dpi': 240,
                         'axes.spines.top': False, 'axes.spines.right': False})
    macros = {'PPTrials': str(s['settings']['trials']), 'PPDraws': '20{,}000',
              'PPNestedFits': str(s['settings']['trials']*s['settings']['bootstrap']),
              'PPEndpointLegacy': f"{s['primary_endpoint_rmse'][0]:.5f}",
              'PPEndpointThousand': f"{s['primary_endpoint_rmse'][1]:.5f}",
              'PPEndpointMany': f"{s['primary_endpoint_rmse'][2]:.5f}"}
    words = {75.: 'SeventyFive', 90.: 'Ninety'}
    for row in s['coverage']:
        prefix = {'legacy_100': 'Old', 'mc_20000': 'Many', 'gaussian_mixture': 'Mix'}[row['method']]
        region = {'full': 'Full', 'near': 'Near'}[row['region']]
        key = prefix+region+words[row['day']]
        macros['PP'+key] = f"{100*row['mean_coverage']:.2f}"
        macros['PP'+key+'CI'] = '['+', '.join(f'{100*x:.2f}' for x in row['coverage_mc_interval'])+']'
        if row['method'] == 'gaussian_mixture' and row['region'] == 'full':
            macros['PPGain'+words[row['day']]] = f"{100*row['paired_coverage_change_from_legacy']:.2f}"
            macros['PPGain'+words[row['day']]+'CI'] = '['+', '.join(f'{100*x:.2f}' for x in row['paired_change_mc_interval'])+']'
    for point in s['sentinel']:
        if point['method'] == 'gaussian_mixture':
            key = words[point['day']]
            macros['PPSentinel'+key] = str(point['hits'])
            macros['PPSentinel'+key+'CI'] = '['+', '.join(f'{100*x:.1f}' for x in point['exact_binomial_interval'])+']'
    (HERE/'generated/pat16_prediction_numbers.tex').write_text('% Generated from verified nested prediction audit.\n'+
        '\n'.join(r'\newcommand{\%s}{%s}' % (key, value) for key, value in macros.items())+'\n')
    rows_file('pat16_prediction_rows.tex', 'PPValidationRows', [
        f"{label} & {row['day']:g} & {'Full FOV' if row['region'] == 'full' else 'Near center'} & "
        f"{100*row['mean_coverage']:.2f} [{100*row['coverage_mc_interval'][0]:.2f}, {100*row['coverage_mc_interval'][1]:.2f}] & "
        f"{row['mean_width']:.5f} & {row['mean_interval_score']:.5f} "+r'\\'
        for method, label in zip(s['methods'], ['100 draws', '20,000 draws', 'Mixture'])
        for row in s['coverage'] if row['method'] == method])
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.1), layout='constrained')
    ax = axes[0, 0]
    ax.loglog([100, 1000, 20000], s['primary_endpoint_rmse'][:3], 'o-', color=COLORS[1], lw=1)
    ax.set(xticks=[100, 1000, 20000], xticklabels=['100', '1,000', '20,000'],
           xlabel='Predictive draws', ylabel='Endpoint RMSE (signal units)',
           title='a. Primary interval convergence')
    ax.minorticks_off()
    for ax, region, letter in [(axes[0, 1], 'full', 'b'), (axes[1, 0], 'near', 'c')]:
        for m, method in enumerate(s['methods']):
            rows = [r for r in s['coverage'] if r['method'] == method and r['region'] == region]
            mean = 100*np.array([r['mean_coverage'] for r in rows])
            ci = 100*np.array([r['coverage_mc_interval'] for r in rows])
            ax.errorbar(np.arange(2)+(m-1)*.15, mean, yerr=[mean-ci[:, 0], ci[:, 1]-mean],
                        fmt=['o', 's', 'D'][m], ms=3, capsize=3, color=COLORS[m], label=METHODS[m])
        ax.axhline(95, color='black', ls='--', lw=.8)
        ax.set(xticks=[0, 1], xticklabels=['75', '90'], xlabel='Withheld day',
               ylabel='Mean spatial coverage (%)', title=f'{letter}. '+('Full observation cube' if region == 'full' else 'Near-center sphere'))
        ax.legend(fontsize=5.7, loc='center', bbox_to_anchor=(.5, .5))
    ax = axes[1, 1]
    for m, method in enumerate(s['methods']):
        rows = [r for r in s['coverage'] if r['method'] == method and r['region'] == 'full']
        ax.plot([75, 90], [r['mean_interval_score'] for r in rows], ['o-', 's--', 'D:'][m], ms=3,
                lw=1, color=COLORS[m], label=METHODS[m])
    ax.set(xticks=[75, 90], xlabel='Withheld day', ylabel='Mean 95% interval score', title='d. Width and miss penalties')
    ax.legend(fontsize=5.7)
    for suffix in ['pdf', 'png']:
        fig.savefig(HERE/f'figures/pat16_prediction_validation.{suffix}', dpi=240)
    plt.close(fig)
    (HERE/'data/pat16_prediction_claims_audit.json').write_text(json.dumps(dict(
        summary_sha256=hashlib.sha256((RECORDS/'summary.json').read_bytes()).hexdigest(),
        source_sha256=s['source_sha256'], output_sha256=s['output_sha256'],
        settings=s['settings'], verified=['record hashes', 'primary predictive draws and exact quantiles',
        'all independent training/test noise streams', 'all legacy quantiles and Gaussian-mixture CDF endpoints',
        'all trial containment/width/proper scores and expected coverage', 'trial-level Monte Carlo intervals',
        'paired changes and fixed-voxel exact binomial coverage'],
        coverage=s['coverage'], sentinel=s['sentinel'], interpretation=s['interpretation']), indent=2)+'\n')


if __name__ == '__main__':
    from make_pat16_recovery_figures import load_records
    baseline, z = load_records()
    audit, primary = verify_records(baseline, z)
    generate_assets(audit, primary)
    print('Verified 40 nested PAT16 prediction experiments; generated coverage table and validation figure.')
