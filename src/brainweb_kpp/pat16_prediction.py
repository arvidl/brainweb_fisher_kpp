"""Prediction intervals and independent, nested PAT16 coverage experiments.

The empirical bootstrap distribution of forecast means is convolved with the
known Gaussian observation noise. No withheld observations calibrate the tails.
All Monte Carlo error bars use independently refitted datasets as replicates.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import multiprocessing
import time

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import binomtest, t

from .pat16 import sha256
from .pat16_inference import PAT16Experiment, PAT16RecoveryDesign, fit_pat16_parameters


@dataclass(frozen=True)
class PredictionSettings:
    trials: int = 40
    bootstrap: int = 100
    predictive_draws: int = 20000
    seed: int = 161027


def mixture_interval(means, noise_sd, confidence=.95):
    """Pointwise quantiles of an equally weighted Gaussian location mixture.

    Bisection integrates observation noise analytically, eliminating predictive
    noise Monte Carlo error while retaining finite-bootstrap uncertainty.
    """
    means = np.asarray(means, float)
    if (means.ndim < 2 or len(means) < 2 or not np.all(np.isfinite(means))
            or not np.isfinite(noise_sd) or noise_sd <= 0 or not 0 < confidence < 1):
        raise ValueError('Need finite bootstrap means, positive noise and confidence in (0,1)')
    alpha = (1-confidence)/2
    result = []
    for probability in (alpha, 1-alpha):
        # Every component quantile lies inside this bracket.
        low = means.min(axis=0) + noise_sd*ndtri(probability)
        high = means.max(axis=0) + noise_sd*ndtri(probability)
        for _ in range(32):
            mid = (low+high)/2
            below = ndtr((mid[None]-means)/noise_sd).mean(axis=0) < probability
            low, high = np.where(below, mid, low), np.where(below, high, mid)
        result.append((low+high)/2)
    return np.array(result)


def predictive_interval(means, noise_sd, draws, seed, confidence=.95, chunk=128):
    """Resample whole bootstrap trajectories, then add independent image noise.

    Chunking bounds memory. Fixed chunks and seeds make saved runs reproducible.
    The same component index is used across voxels for each predictive draw.
    """
    means = np.asarray(means, float)
    if (means.ndim < 2 or len(means) < 2 or not np.all(np.isfinite(means))
            or draws < 2 or int(draws) != draws or noise_sd <= 0
            or not np.isfinite(noise_sd) or not 0 < confidence < 1 or chunk < 1):
        raise ValueError('Invalid predictive draw settings')
    rng = np.random.default_rng(seed)
    index = rng.integers(len(means), size=draws)
    flat = means.reshape(len(means), -1)
    interval = np.empty((2, flat.shape[1]))
    alpha = (1-confidence)/2
    for start in range(0, flat.shape[1], chunk):
        stop = min(start+chunk, flat.shape[1])
        samples = flat[index, start:stop] + noise_sd*rng.standard_normal((draws, stop-start))
        interval[:, start:stop] = np.quantile(samples, [alpha, 1-alpha], axis=0)
    return interval.reshape((2,)+means.shape[1:])


def interval_metrics(interval, observed, truth, noise_sd, mask, confidence=.95):
    """Region summaries; these voxels are not independent fit replicates."""
    low, high = interval[:, :, mask]
    y, mu = observed[:, mask], truth[:, mask]
    score = high-low + 2/(1-confidence)*(np.maximum(low-y, 0)+np.maximum(y-high, 0))
    return np.stack((np.mean((low <= y) & (y <= high), axis=1),
                     np.mean(high-low, axis=1), np.mean(score, axis=1),
                     np.mean(ndtr((high-mu)/noise_sd)-ndtr((low-mu)/noise_sd), axis=1)), axis=-1)


def mean_mc_interval(values):
    """Student-t Monte Carlo interval across independent complete experiments."""
    values = np.asarray(values, float)
    mean = values.mean(axis=0)
    half = t.ppf(.975, len(values)-1)*values.std(axis=0, ddof=1)/np.sqrt(len(values))
    return np.stack((mean-half, mean+half), axis=-1)


_EXPERIMENT = None


def _initialize(pilot, design):
    global _EXPERIMENT
    _EXPERIMENT = PAT16Experiment(pilot, PAT16RecoveryDesign(**design))


def _trial(index, directory, baseline, settings, fingerprint):
    e = _EXPERIMENT
    output = Path(directory)/f'trial_{index:03d}.npz'
    record = output.with_suffix('.json')
    if output.exists() and record.exists():
        metadata = json.loads(record.read_text())
        if metadata['fingerprint'] == fingerprint and metadata['sha256'] == sha256(output):
            return index, metadata
        raise ValueError(f'Checkpoint differs from requested experiment: {record}')
    s = PredictionSettings(**settings)
    b = baseline
    train_rng, boot_rng, test_rng, old_rng, mc_seed = np.random.SeedSequence(
        [s.seed, index, 1]).spawn(5)
    train_rng, boot_rng, test_rng, old_rng = map(np.random.default_rng,
                                               (train_rng, boot_rng, test_rng, old_rng))
    sd, days, n = b['noise_sd'], b['days'], b['training_scans']
    clean = b['clean']
    train = clean[:n] + sd*train_rng.standard_normal(clean[:n].shape)
    future = clean[n:] + sd*test_rng.standard_normal(clean[n:].shape)
    parameters = np.full((s.bootstrap, 4), np.nan)
    bounds = np.zeros(s.bootstrap, bool)
    failures = []
    primary_failed, primary_bound = False, False
    estimate = np.full(4, np.nan)
    forecasts = []
    try:
        fit = fit_pat16_parameters(e, days[:n], train, sd)
        estimate, primary_bound = fit.parameters, fit.at_bound
        fitted_training = e.predict(*estimate, days[:n])
        for j in range(s.bootstrap):
            y = fitted_training + sd*boot_rng.standard_normal(fitted_training.shape)
            try:
                bf = fit_pat16_parameters(e, days[:n], y, sd, start=estimate)
                parameters[j], bounds[j] = bf.parameters, bf.at_bound
                forecasts.append(e.predict(*bf.parameters, days[n:]))
            except (ValueError, RuntimeError) as error:
                failures.append({'bootstrap_index': j, 'reason': str(error)})
    except (ValueError, RuntimeError) as error:
        primary_failed = True
        failures.append({'primary': str(error)})
    valid = np.isfinite(parameters).all(axis=1)
    shape = (2,)+clean[n:].shape
    intervals = np.full((3,)+shape, np.nan)
    usable = not primary_failed and not primary_bound and len(forecasts) >= 2
    if usable:
        forecasts = np.asarray(forecasts)
        intervals[0] = np.quantile(forecasts + sd*old_rng.standard_normal(forecasts.shape), [.025, .975], axis=0)
        intervals[1] = predictive_interval(forecasts, sd, s.predictive_draws, mc_seed)
        intervals[2] = mixture_interval(forecasts, sd)
    else:
        forecasts = np.empty((0,)+clean[n:].shape)
    masks = [np.ones(clean.shape[1:], bool), b['near']]
    # Unusable procedures count as misses, with no fabricated width/score.
    metrics = np.full((3, 2, len(days)-n, 4), np.nan)
    metrics[..., 0] = 0
    metrics[..., 3] = 0
    sentinel = np.zeros((3, len(days)-n), bool)
    if usable:
        for method in range(3):
            for region, mask in enumerate(masks):
                metrics[method, region] = interval_metrics(intervals[method], future, clean[n:], sd, mask)
            i = (slice(None),)+tuple(b['sentinel'])
            lo, hi = intervals[method][0][i], intervals[method][1][i]
            sentinel[method] = (lo <= future[i]) & (future[i] <= hi)
    np.savez_compressed(output, estimate=estimate, training_observed=train, future_observed=future,
                        bootstrap_parameters=parameters, bootstrap_boundary=bounds, bootstrap_valid=valid,
                        bootstrap_predictions=forecasts, intervals=intervals, metrics=metrics,
                        sentinel_hits=sentinel)
    metadata = dict(index=index, fingerprint=fingerprint, sha256=sha256(output),
                    primary_failed=primary_failed, primary_boundary=bool(primary_bound), usable=usable,
                    bootstrap_successful=int(valid.sum()), bootstrap_boundary=int(bounds.sum()), failures=failures)
    record.write_text(json.dumps(metadata, indent=2)+'\n')
    return index, metadata


def summarize_trials(records, directory, days):
    arrays = [np.load(Path(directory)/f'trial_{i:03d}.npz') for i in range(len(records))]
    metrics = np.array([a['metrics'] for a in arrays])
    hits = np.array([a['sentinel_hits'] for a in arrays]).sum(axis=0)
    rows = []
    methods = ['legacy_100', 'mc_20000', 'gaussian_mixture']
    for m, name in enumerate(methods):
        for r, region in enumerate(['full', 'near']):
            for d, day in enumerate(days):
                values = metrics[:, m, r, d]
                coverage_ci = mean_mc_interval(values[:, 0])
                rows.append(dict(method=name, region=region, day=float(day),
                                 mean_coverage=float(values[:, 0].mean()),
                                 coverage_mc_interval=coverage_ci.tolist(),
                                 mean_width=float(np.nanmean(values[:, 1])),
                                 mean_interval_score=float(np.nanmean(values[:, 2])),
                                 mean_expected_coverage=float(values[:, 3].mean()),
                                 paired_coverage_change_from_legacy=float((values[:, 0]-metrics[:, 0, r, d, 0]).mean()),
                                 paired_change_mc_interval=mean_mc_interval(values[:, 0]-metrics[:, 0, r, d, 0]).tolist()))
    sentinel = []
    for m, name in enumerate(methods):
        for d, day in enumerate(days):
            h = int(hits[m, d])
            ci = binomtest(h, len(records)).proportion_ci()
            sentinel.append(dict(method=name, day=float(day), hits=h, trials=len(records),
                                 fraction=h/len(records), exact_binomial_interval=[ci.low, ci.high]))
    for a in arrays:
        a.close()
    return rows, sentinel


def run_prediction_audit(pilot, recovery, output, settings=PredictionSettings(), workers=8, progress=print):
    """Run/resume independent nested experiments, then save an audited manifest."""
    started = time.perf_counter()
    s = settings
    if min(s.trials, s.bootstrap, s.predictive_draws) < 2 or workers < 1:
        raise ValueError('Need at least two trials, bootstrap fits and predictive draws')
    recovery, output = Path(recovery), Path(output)
    trials = output/'trials'
    trials.mkdir(parents=True, exist_ok=True)
    summary = json.loads((recovery/'summary.json').read_text())
    with np.load(recovery/'recovery_data.npz') as z:
        b = dict(clean=z['clean'], observed=z['observed'], days=z['scan_days'],
                 training_scans=int(z['training_scans']), near=z['near_center'],
                 noise_sd=summary['settings']['noise_sd'], sentinel=[6, 6, 6])
        original = z['prediction_interval'][:, b['training_scans']:]
        means = z['bootstrap_predictions'][:, b['training_scans']:]
    sources = {name: sha256(Path(__file__).with_name(name)) for name in
               ['pat16.py', 'anisotropic.py', 'pat16_inference.py', 'pat16_prediction.py']}
    config = dict(settings=asdict(s), design=summary['design'], source_sha256=sources,
                  baseline_sha256={name: sha256(recovery/name) for name in ['summary.json', 'recovery_data.npz']})
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    primary_file = output/'primary.npz'
    progress('Computing 1,000 and 20,000 predictive draws and exact Gaussian-mixture quantiles...')
    primary_seeds = np.random.SeedSequence([s.seed, 0]).spawn(2)
    mc1000 = predictive_interval(means, b['noise_sd'], 1000, primary_seeds[0])
    mc = predictive_interval(means, b['noise_sd'], s.predictive_draws, primary_seeds[1])
    exact = mixture_interval(means, b['noise_sd'])
    primary = np.array([original, mc1000, mc, exact])
    primary_metrics = np.array([[interval_metrics(pi, b['observed'][b['training_scans']:],
                                                 b['clean'][b['training_scans']:], b['noise_sd'], mask)
                                for mask in [np.ones(b['near'].shape, bool), b['near']]] for pi in primary])
    np.savez_compressed(primary_file, intervals=primary, metrics=primary_metrics,
                        endpoint_rmse=np.sqrt(np.mean((primary-exact)**2, axis=(1, 2, 3, 4, 5))))
    progress(f'Refitting {s.trials} independent datasets, each with {s.bootstrap} bootstrap fits ({workers} workers)...')
    records = [None]*s.trials
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize, initargs=(pilot, summary['design'])) as pool:
        futures = [pool.submit(_trial, i, str(trials), b, asdict(s), fingerprint) for i in range(s.trials)]
        for future in as_completed(futures):
            i, metadata = future.result()
            records[i] = metadata
            progress(f'Coverage trial {i+1}: {metadata["bootstrap_successful"]}/{s.bootstrap} bootstrap fits; '
                     f'{sum(r is not None for r in records)}/{s.trials} datasets complete')
    rows, sentinel = summarize_trials(records, trials, b['days'][b['training_scans']:])
    result = dict(experiment='Nested, independent synthetic PAT16 pointwise prediction-interval validation',
                  **config, fingerprint=fingerprint, scan_days=b['days'].tolist(),
                  withheld_days=b['days'][b['training_scans']:].tolist(), noise_sd=b['noise_sd'],
                  methods=['legacy_100', 'mc_20000', 'gaussian_mixture'],
                  primary_method_order=['legacy_100', 'mc_1000', 'mc_20000', 'gaussian_mixture'],
                  metric_order=['coverage', 'width', 'interval_score', 'expected_coverage'],
                  regions=['full', 'near'], region_voxels=[int(b['near'].size), int(b['near'].sum())],
                  sentinel_index=b['sentinel'], primary_metrics=primary_metrics.tolist(),
                  primary_endpoint_rmse=np.sqrt(np.mean((primary-exact)**2, axis=(1, 2, 3, 4, 5))).tolist(),
                  coverage=rows, sentinel=sentinel, records=records,
                  usable_trials=sum(r['usable'] for r in records),
                  primary_boundary_trials=sum(r['primary_boundary'] for r in records),
                  bootstrap_failures=sum(s.bootstrap-r['bootstrap_successful'] for r in records),
                  bootstrap_boundary_fits=sum(r['bootstrap_boundary'] for r in records),
                  interpretation='Pointwise intervals conditional on the specified anatomy, initialization, operator and known noise. '
                  'Independent complete datasets are the coverage replicates. Monte Carlo intervals for mean spatial coverage use '
                  'Student t across datasets, not a pooled voxel binomial. Fixed-voxel coverage has exact binomial intervals. '
                  'Gaussian convolution removes predictive noise sampling error, not finite-bootstrap error or model discrepancy. '
                  'No quantiles or width corrections were selected using withheld observations.',
                  elapsed_seconds=time.perf_counter()-started)
    files = ['primary.npz']+[f'trials/trial_{i:03d}{suffix}' for i in range(s.trials) for suffix in ['.npz', '.json']]
    result['output_sha256'] = {name: sha256(output/name) for name in files}
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    progress(f'Saved prediction audit to {output}')
    return result
