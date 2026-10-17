"""Reproducible uncertainty and known-value audit for PAT16 synthetic recovery."""
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
import csv
import json
import time

import numpy as np
from scipy.stats import binomtest

from .pat16 import sha256
from .pat16_inference import (
    DEFAULT_BOUNDS, PARAMETER_NAMES, PAT16Experiment, PAT16RecoveryDesign,
    fit_pat16_parameters,
)


@dataclass(frozen=True)
class PAT16RecoverySettings:
    known_parameters: tuple = (.5, .1, .04, 2.5)
    scan_days: tuple = (20., 40., 60., 75., 90.)
    training_scans: int = 3
    noise_sd: float = .015
    n_bootstrap: int = 100
    n_coverage: int = 60
    seed: int = 161026
    confidence: float = .95
    near_center_radius_mm: float = 24.
    starts: tuple = ((.25, .2, .025, 1.5), (.4, .15, .035, 2.),
                     (.8, .05, .065, 3.2))


def _binomial(hits, count, confidence):
    low, high = binomtest(int(hits), int(count)).proportion_ci(confidence_level=confidence)
    return [float(low), float(high)]


def run_pat16_recovery(pilot, output, *, design=PAT16RecoveryDesign(),
                       settings=PAT16RecoverySettings(), progress=print):
    """Fit early scans; audit uncertainty without using withheld data in fitting.

    The generator uses half the inference time step on the same spatial anatomy.
    Bootstrap resamples come from the fitted model, independent noise trials
    from the known finer-time generator. Failed/boundary fits are retained in
    diagnostics. Coverage counts an unusable local interval as a miss.
    """
    started = time.perf_counter()
    s = settings
    known, days = np.asarray(s.known_parameters, float), np.asarray(s.scan_days, float)
    bounds = np.asarray(DEFAULT_BOUNDS)
    if (known.shape != (4,) or not np.all(np.isfinite(known))
            or np.any(known <= bounds[0]) or np.any(known >= bounds[1])):
        raise ValueError("Known parameters must lie strictly inside the search bounds")
    if (days.ndim != 1 or len(days) < 2 or not np.all(np.isfinite(days))
            or days[0] < 0 or np.any(np.diff(days) <= 0)):
        raise ValueError("Need strictly increasing, nonnegative scan days")
    if (isinstance(s.training_scans, bool) or not isinstance(s.training_scans, int)
            or not 1 <= s.training_scans < len(days)):
        raise ValueError("Need training scans and at least one withheld scan")
    for name in ("n_bootstrap", "n_coverage"):
        count = getattr(s, name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError(f"{name} must be an integer of at least two")
    if (not np.isfinite(s.noise_sd) or s.noise_sd <= 0 or not 0 < s.confidence < 1
            or not np.isfinite(s.near_center_radius_mm) or s.near_center_radius_mm <= 0):
        raise ValueError("Need positive noise/radius and confidence in (0,1)")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def report(message):
        if progress is not None:
            progress(message)

    # Separate, reproducible streams prevent changed replication counts from
    # changing the primary noise realization or the coverage datasets.
    noise_rng, boot_rng, coverage_rng, forecast_rng = [
        np.random.default_rng(child) for child in np.random.SeedSequence(s.seed).spawn(4)]
    inverse = PAT16Experiment(pilot, design)
    # Both experiments use exactly the same prespecified physical observation region.
    design = replace(design, observation_center_mm=tuple(inverse.observation_center_mm))
    generator = PAT16Experiment(pilot, replace(design, dt_days=design.dt_days/2))
    train_days, n_train = days[:s.training_scans], s.training_scans
    clean = generator.predict(*known, days)
    observed = clean + s.noise_sd*noise_rng.standard_normal(clean.shape)
    options = {"confidence": s.confidence}
    report("Fitting four parameters from early synthetic PAT16 scans...")
    fits = [fit_pat16_parameters(inverse, train_days, observed[:n_train], s.noise_sd,
                                 start=start, **options) for start in s.starts]
    if not fits:
        raise ValueError("Provide at least one optimization start")
    fit = min(fits, key=lambda f: np.sum(f.standardized_residuals**2))
    predicted = inverse.predict(*fit.parameters, days)
    report(f"Estimate: {fit.parameters.tolist()}; boundary optimum: {fit.at_bound}")

    boot_parameters = np.full((s.n_bootstrap, 4), np.nan)
    boot_bound = np.zeros(s.n_bootstrap, bool)
    failures = {"bootstrap": [], "coverage": []}
    for i in range(s.n_bootstrap):
        y = predicted[:n_train] + s.noise_sd*boot_rng.standard_normal(predicted[:n_train].shape)
        try:
            bfit = fit_pat16_parameters(inverse, train_days, y, s.noise_sd,
                                        start=fit.parameters, **options)
            boot_parameters[i], boot_bound[i] = bfit.parameters, bfit.at_bound
        except (ValueError, RuntimeError) as error:
            failures['bootstrap'].append({"index": i, "reason": str(error)})
        if (i+1) % 10 == 0 or i+1 == s.n_bootstrap:
            report(f"Bootstrap {i+1}/{s.n_bootstrap}")
    valid_boot = np.all(np.isfinite(boot_parameters), axis=1)
    if valid_boot.sum() < 2:
        raise RuntimeError("Fewer than two successful bootstrap fits; uncertainty cannot be estimated")
    alpha = (1-s.confidence)/2
    boot_intervals = np.quantile(boot_parameters[valid_boot], [alpha, 1-alpha], axis=0).T
    boot_predictions = np.array([inverse.predict(*p, days) for p in boot_parameters[valid_boot]])
    signal_interval = np.quantile(boot_predictions, [alpha, 1-alpha], axis=0)
    noisy_forecasts = boot_predictions + s.noise_sd*forecast_rng.standard_normal(boot_predictions.shape)
    prediction_interval = np.quantile(noisy_forecasts, [alpha, 1-alpha], axis=0)

    recovered = np.full((s.n_coverage, 4), np.nan)
    coverage_intervals = np.full((s.n_coverage, 4, 2), np.nan)
    coverage_bound = np.zeros(s.n_coverage, bool)
    for i in range(s.n_coverage):
        y = clean[:n_train] + s.noise_sd*coverage_rng.standard_normal(clean[:n_train].shape)
        try:
            cfit = fit_pat16_parameters(inverse, train_days, y, s.noise_sd,
                                        start=fit.parameters, **options)
            recovered[i], coverage_intervals[i], coverage_bound[i] = cfit.parameters, cfit.intervals, cfit.at_bound
        except (ValueError, RuntimeError) as error:
            failures['coverage'].append({"index": i, "reason": str(error)})
        if (i+1) % 10 == 0 or i+1 == s.n_coverage:
            report(f"Independent noise trials {i+1}/{s.n_coverage}")
    valid_coverage = np.all(np.isfinite(recovered), axis=1)
    if not valid_coverage.any():
        raise RuntimeError("All independent noise fits failed")
    usable = valid_coverage & ~coverage_bound
    coverage_hits = ((coverage_intervals[:, :, 0] <= known)
                     & (known <= coverage_intervals[:, :, 1]) & usable[:, None]).sum(axis=0)

    report("Checking noise-free recovery, temporal discrepancy and identifiability...")
    matched = inverse.predict(*known, days)
    noiseless_fit = fit_pat16_parameters(inverse, train_days, matched[:n_train], s.noise_sd,
                                         start=s.starts[0], **options)
    finer_fit = fit_pat16_parameters(inverse, train_days, clean[:n_train], s.noise_sd,
                                     start=fit.parameters, **options)
    matched_error = np.max(np.abs(noiseless_fit.parameters/known-1))
    if matched_error > 1e-4:
        raise RuntimeError(f"Noise-free matched-model recovery failed: relative error {matched_error:g}")
    negative_control = None
    try:
        fit_pat16_parameters(inverse, [0.], inverse.predict(*known, [0.]), s.noise_sd,
                              start=s.starts[0], **options)
    except ValueError as error:
        if "not locally identifiable" not in str(error):
            raise
        negative_control = str(error)
    if negative_control is None:
        raise RuntimeError("Time-zero-only inference should be unidentifiable")

    coordinates = np.stack(np.meshgrid(*inverse.observation_axes, indexing='ij'), axis=-1)
    near = np.linalg.norm(coordinates-inverse.observation_center_mm, axis=-1) <= s.near_center_radius_mm
    if not near.any():
        raise ValueError("Near-center region contains no observation voxels")
    def forecast_metrics(region):
        rows = []
        for i in range(n_train, len(days)):
            truth, obs, pred = clean[i][region], observed[i][region], predicted[i][region]
            sig, pi = signal_interval[:, i][:, region], prediction_interval[:, i][:, region]
            rows.append({"day": float(days[i]), "voxels": int(truth.size),
                         "truth_rmse": float(np.sqrt(np.mean((pred-truth)**2))),
                         "observation_rmse_over_noise_sd": float(np.sqrt(np.mean((pred-obs)**2))/s.noise_sd),
                         "signal_interval_truth_fraction": float(np.mean((sig[0] <= truth) & (truth <= sig[1]))),
                         "prediction_interval_observation_fraction": float(np.mean((pi[0] <= obs) & (obs <= pi[1]))),
                         "mean_signal_interval_width": float(np.mean(sig[1]-sig[0])),
                         "mean_prediction_interval_width": float(np.mean(pi[1]-pi[0]))})
        return rows

    local_contains = (fit.intervals[:, 0] <= known) & (known <= fit.intervals[:, 1])
    boot_contains = (boot_intervals[:, 0] <= known) & (known <= boot_intervals[:, 1])
    coverage_ci = np.array([_binomial(h, s.n_coverage, s.confidence) for h in coverage_hits])
    parameter_rows = []
    for i, name in enumerate(PARAMETER_NAMES):
        estimates = recovered[valid_coverage, i]
        parameter_rows.append({
            "parameter": name, "known": float(known[i]), "estimate": float(fit.parameters[i]),
            "relative_error_percent": float(100*(fit.parameters[i]/known[i]-1)),
            "local_interval": fit.intervals[i].tolist(), "bootstrap_interval": boot_intervals[i].tolist(),
            "local_contains_known": bool(local_contains[i]), "bootstrap_contains_known": bool(boot_contains[i]),
            "noise_trial_mean": float(estimates.mean()),
            "noise_trial_bias_percent": float(100*(estimates.mean()/known[i]-1)),
            "noise_trial_rmse": float(np.sqrt(np.mean((estimates-known[i])**2))),
            "local_coverage_hits": int(coverage_hits[i]), "local_coverage_trials": s.n_coverage,
            "local_coverage_fraction": float(coverage_hits[i]/s.n_coverage),
            "local_coverage_binomial_interval": coverage_ci[i].tolist(),
        })
    summary = {
        "experiment": "Synthetic four-parameter recovery on real PAT16 pre-operative anatomy",
        "scope": "Known-value synthetic validation, not patient rate estimation or clinical validation",
        "uncertainty": "Known-noise local Wald and parametric percentile bootstrap confidence intervals; no prior",
        "conditional_on": ["tissue segmentation and geometry", "dMRI directions/eligibility", "initial density",
                           "carrying capacity and imaging calibration", "independent Gaussian noise and its SD"],
        "settings": asdict(s), "design": asdict(design), "bounds": bounds.tolist(),
        "generator_dt_days": generator.design.dt_days, "inverse_dt_days": design.dt_days,
        "parameter_names": list(PARAMETER_NAMES), "parameters": parameter_rows,
        "at_bound": fit.at_bound, "log_parameter_correlation": fit.correlation.tolist(),
        "jacobian_singular_values": fit.singular_values.tolist(),
        "jacobian_condition_number": float(fit.singular_values[0]/fit.singular_values[-1]),
        "training_standardized_residual_rms": float(np.sqrt(np.mean(fit.standardized_residuals**2))),
        "multistart": [{"start": list(start), "estimate": f.parameters.tolist(),
                        "sum_squared_standardized_residuals": float(np.sum(f.standardized_residuals**2)),
                        "at_bound": f.at_bound} for start, f in zip(s.starts, fits)],
        "bootstrap_successful": int(valid_boot.sum()), "bootstrap_boundary_fits": int(boot_bound.sum()),
        "coverage_successful": int(valid_coverage.sum()), "coverage_boundary_fits": int(coverage_bound.sum()),
        "coverage_usable_local_intervals": int(usable.sum()), "failures": failures,
        "coverage_method": "Local intervals only; denominator includes every attempted trial; failed/boundary intervals count as misses. Bootstrap coverage is not audited.",
        "forecast_full_region": forecast_metrics(np.ones(inverse.observation_shape, bool)),
        "forecast_near_center": forecast_metrics(near),
        "forecast_interval_method": "Pointwise percentile intervals; signal interval propagates fitted parameter uncertainty; prediction interval adds fresh observation noise. Spatial fractions are descriptive, not independent coverage trials.",
        "numerical_validation": {
            "matched_model_noiseless_estimate": noiseless_fit.parameters.tolist(),
            "matched_model_max_relative_error": float(matched_error),
            "finer_time_noiseless_estimate": finer_fit.parameters.tolist(),
            "finer_time_relative_parameter_shift_percent": (100*(finer_fit.parameters/known-1)).tolist(),
            "generator_minus_inverse_rmse_by_scan": np.sqrt(np.mean((clean-matched)**2, axis=(1, 2, 3))).tolist(),
            "time_zero_negative_control": negative_control,
            "limitation": "Generator and inverse share spatial anatomy, stencil and initialization; half time step is a temporal sensitivity check, not spatial convergence or independent biological validation.",
        },
        "elapsed_seconds": float(time.perf_counter()-started),
    }
    arrays = dict(known_parameters=known, scan_days=days, training_scans=np.array(n_train),
                  clean=clean, observed=observed, predicted=predicted,
                  estimate=fit.parameters, log_covariance=fit.log_covariance,
                  local_intervals=fit.intervals, bootstrap_intervals=boot_intervals,
                  correlation=fit.correlation, standardized_residuals=fit.standardized_residuals,
                  bootstrap_parameters=boot_parameters, bootstrap_boundary=boot_bound,
                  bootstrap_valid=valid_boot, bootstrap_predictions=boot_predictions,
                  signal_interval=signal_interval, prediction_interval=prediction_interval,
                  noise_trial_parameters=recovered, noise_trial_intervals=coverage_intervals,
                  noise_trial_boundary=coverage_bound, noise_trial_valid=valid_coverage,
                  coverage_hits=coverage_hits, coverage_binomial_intervals=coverage_ci,
                  near_center=near, observation_center_mm=inverse.observation_center_mm,
                  observation_axes_mm=np.array(inverse.observation_axes),
                  matched_model_at_truth=matched)
    np.savez_compressed(output/'recovery_data.npz', **arrays)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (output/'parameters.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['parameter', 'known', 'estimate', 'relative_error_percent',
                         'local_low', 'local_high', 'bootstrap_low', 'bootstrap_high',
                         'local_contains_known', 'bootstrap_contains_known', 'coverage_hits', 'coverage_trials'])
        for row in parameter_rows:
            writer.writerow([row['parameter'], row['known'], row['estimate'], row['relative_error_percent'],
                             *row['local_interval'], *row['bootstrap_interval'], row['local_contains_known'],
                             row['bootstrap_contains_known'], row['local_coverage_hits'], s.n_coverage])
    plot_pat16_recovery(summary, arrays, output)
    # No raw anatomy or source paths are copied to this output directory.
    provenance = {
        "anatomy": pilot.provenance,
        "software_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                            ['pat16.py', 'anisotropic.py', 'pat16_inference.py', 'pat16_recovery.py']},
        "software_versions": {name: version(name) for name in ['numpy', 'scipy', 'numba', 'nibabel', 'matplotlib']},
        "outputs_sha256": {name: sha256(output/name) for name in
                           ['recovery_data.npz', 'summary.json', 'parameters.csv',
                            'parameter_recovery.png', 'validation.png', 'withheld_forecast.png']},
    }
    # Match pilot exports: retain the executable name but remove source-local paths.
    if 'dti_history' in provenance['anatomy']:
        provenance['anatomy'] = dict(provenance['anatomy'])
        provenance['anatomy']['dti_history'] = [line.split()[0] for line in pilot.provenance['dti_history']]
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    report(f"Saved PAT16 recovery results to {output}")
    return summary, arrays


def plot_pat16_recovery(summary, arrays, output):
    """Generate standalone figures directly from recorded recovery arrays."""
    import matplotlib.pyplot as plt
    z, s = arrays, summary['settings']
    output = Path(output)
    known, estimate = z['known_parameters'], z['estimate']
    labels = [r'$D_{WM}$', r'$D_{GM}$', r'$\rho$', r'$r_{\parallel/\perp}$']
    confidence = f"{100*s['confidence']:g}%"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for intervals, offset, label, color in [
            (z['local_intervals'], -.10, 'Local '+confidence, '#2979b8'),
            (z['bootstrap_intervals'], .10, 'Bootstrap '+confidence, '#bf5b21')]:
        # Draw endpoints directly: percentile intervals need not contain the MLE.
        values = intervals/known[:, None]
        axes[0].hlines(np.arange(4)+offset, values[:, 0], values[:, 1], color=color, label=label, lw=2)
        axes[0].plot(estimate/known, np.arange(4)+offset, 'o', color=color, ms=4)
    axes[0].axvline(1, color='black', ls='--', lw=1, label='Known value')
    axes[0].set(yticks=np.arange(4), yticklabels=labels, xlabel='Estimate / known generating value',
                title='Independent WM/GM diffusion, growth and anisotropy')
    axes[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.17), ncol=3)
    artist = axes[1].imshow(z['correlation'], vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1].set(xticks=np.arange(4), yticks=np.arange(4), xticklabels=labels, yticklabels=labels,
                title='Local log-parameter correlation')
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{z['correlation'][i, j]:.2f}", ha='center', va='center',
                         color='white' if abs(z['correlation'][i, j]) > .6 else 'black')
    fig.colorbar(artist, ax=axes[1], shrink=.8)
    fig.suptitle('PAT16 anatomy • synthetic cellularity observations • conditional confidence intervals')
    fig.savefig(output/'parameter_recovery.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    recovered = z['noise_trial_parameters'][z['noise_trial_valid']]/known
    for i in range(4):
        axes[0].scatter(np.full(len(recovered), i), recovered[:, i], alpha=.35, s=12)
    axes[0].plot(np.arange(4), recovered.mean(axis=0), 'k_', ms=20, label='Trial mean')
    axes[0].axhline(1, color='black', ls='--', lw=1)
    axes[0].set(xticks=np.arange(4), xticklabels=labels, ylabel='Recovered / known value',
                title=f"{s['n_coverage']} independent noise trials")
    axes[0].legend()
    fraction = z['coverage_hits']/s['n_coverage']
    ci = z['coverage_binomial_intervals']
    axes[1].errorbar(np.arange(4), fraction, yerr=[fraction-ci[:, 0], ci[:, 1]-fraction], fmt='o', capsize=4)
    axes[1].axhline(s['confidence'], color='black', ls='--', label='Nominal '+confidence)
    axes[1].set(xticks=np.arange(4), xticklabels=labels, ylabel='Local interval coverage', ylim=(0, 1.03),
                title=confidence+' exact binomial intervals')
    axes[1].legend()
    fig.suptitle('Known-value audit • failed/boundary local intervals count as misses')
    fig.savefig(output/'validation.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), layout='constrained')
    middle = z['clean'].shape[-1]//2
    for row, index in enumerate(range(s['training_scans'], min(s['training_scans']+2, len(z['scan_days'])))):
        for col, (key, title) in enumerate([('clean', 'Known synthetic signal'), ('predicted', 'Fitted forecast'),
                                            ('error', 'Forecast − known signal')]):
            values = z['predicted'][index]-z['clean'][index] if key == 'error' else z[key][index]
            image = values[:, :, middle].T
            artist = axes[row, col].imshow(image, origin='lower', cmap='RdBu_r' if col == 2 else 'inferno',
                                           vmin=-.02 if col == 2 else 0, vmax=.02 if col == 2 else 1)
            axes[row, col].set(title=f"Day {z['scan_days'][index]:g}: {title}", xlabel='Observation voxel x', ylabel='Observation voxel y')
            fig.colorbar(artist, ax=axes[row, col], shrink=.8, label='Normalized signal')
    if len(z['scan_days'])-s['training_scans'] == 1:
        for ax in axes[1]:
            ax.set_visible(False)
    fig.suptitle('Withheld scans • fixed physical observation cube • no later data used for fitting')
    fig.savefig(output/'withheld_forecast.png', dpi=160)
    plt.close(fig)
