# Synthetic parameter recovery on PAT16 anatomy

The [notebook](../pat16_synthetic_parameter_recovery.ipynb) and headless runner
jointly estimate four separate parameters on the **full real PAT16 pre-operative
tissue domain**, using the pilot's dMRI-directed migration model. The signals
being fitted are generated synthetically. No patient tumor measurement is
used as a cellularity target.

| Parameter | Known generating value | Search bounds |
|---|---:|---:|
| WM mean migration diffusivity D_WM | 0.500 mm²/day | [0.05, 1.00] |
| GM migration diffusivity D_GM | 0.100 mm²/day | [0.02, 0.60] |
| Shared proliferation ρ | 0.040 day⁻¹ | [0.01, 0.09] |
| Eligible-WM parallel/perpendicular migration ratio r | 2.500 | [1.0, 3.5] |

There is no fixed WM:GM ratio or ordering. Bounds define the numerical search
region, not a Bayesian prior. The upper anisotropy bound belongs to the monotone
19-point stencil. WM tensors preserve trace 3 D_WM as r varies; only eligible
WM with valid dMRI directions becomes anisotropic. All GM and remaining WM are
isotropic. Water diffusion magnitudes are not used as cell migration rates.

## Run or replay

From `brainweb_fisher_kpp`, with the locked environment:

```bash
uv run --locked python scripts/run_pat16_recovery.py
uv run --locked jupyter lab pat16_synthetic_parameter_recovery.ipynb
```

The notebook defaults to **replaying the checked recorded results** without
external imaging. Set `RERUN = True` to generate and fit new synthetic scans.
The headless runner always computes a new audit. It uses `PAT16_SOURCE`, or
the same local source-project discovery as the pilot. Input requirements and
resampling/tensor-registration conventions are unchanged: see
[pat16_pilot.md](pat16_pilot.md). This reuses `load_pilot`, which currently
requires the full pilot derivative set, including BOLD files. BOLD is neither
loaded for functional analysis nor used in this recovery likelihood.
Missing required inputs raise an error; there is no synthetic anatomy fallback.

To execute and save the notebook:

```bash
# Replay existing results and verify code/output hashes:
uv run --locked python scripts/run_pat16_recovery_notebook.py
# Generate new results using the external PAT16 derivatives:
uv run --locked python scripts/run_pat16_recovery_notebook.py --rerun
```

Allow several minutes for the default 100 bootstrap fits and 60 independent
noise trials. For a development smoke run, use a separate output directory:

```bash
uv run --locked python scripts/run_pat16_recovery.py \
  --bootstrap 4 --coverage 4 --output /tmp/pat16-recovery-smoke
```

Small replication counts are useful for checking execution, not assessing
uncertainty calibration. Other CLI options are `--source`, `--spacing`, and
`--seed`. Python callers can customize `PAT16RecoveryDesign` and
`PAT16RecoverySettings`; the settings and resolved observation center are
saved with each run.

## Observation model and estimation

The full 3 mm anatomy is evolved with the pilot's harmonic tensor conductances,
blocked diagonal corners, no-flux tissue boundaries and exact logistic reaction.
Inference keeps snapshots in float64 for finite-difference likelihood
derivatives. Tests check agreement with the pilot solver and mass conservation.

Initialization is assumed known: lesion threshold 0.5, Gaussian smoothing of
0.7 grid voxels and normalized density amplitude 0.6. A **72 mm cube**, centered
on the initial lesion centroid before data generation, supplies observations.
A **2 mm Gaussian blur** precedes **6 mm physical voxel averaging** using 27
midpoint quadrature samples per voxel. Independent Gaussian noise with known
SD **0.015** is added after imaging, without clipping.

Scans at **20, 40, 60 days** are fitted. **75 and 90 days** are withheld from
optimization, uncertainty estimation and model selection. The generator uses
a **0.25-day time step** and inversion uses **0.5 day**, on the same spatial
anatomy. Three starts minimize standardized residuals with bounded Gaussian
maximum likelihood in four log parameters. A full-rank SVD is required; all
singular values and the parameter-correlation matrix are reported. Near-bound
fits are flagged and their local Wald intervals are not treated as calibrated.

## Uncertainty and validation

- **Local confidence intervals:** 95% marginal log-Wald intervals from the
  known-noise Jacobian information, without estimating residual variance.
- **Parametric bootstrap:** 100 fresh noisy early-scan datasets from the fitted
  model, all refitted. Marginal percentile intervals use the successful draws;
  every failed fit and boundary draw is recorded, with no replacement draws.
- **Independent known-value audit:** 60 fresh noise datasets from the known,
  finer-time generator. Report mean, relative bias, RMSE, local-interval coverage
  and exact 95% binomial intervals. Failed or boundary local intervals count as
  misses, and all attempted trials remain in the denominator. Marginal bootstrap parameter-interval
  coverage is not audited.
- **Noise-free recovery:** fitting the inverse model's own noiseless scans must
  recover all four values with maximum relative error below 10⁻⁴. A separate
  noise-free finer-time fit quantifies shifts from temporal mismatch. This does
  not test spatial convergence or independent model correctness.
- **Forecasts:** bootstrap predictions produce pointwise signal intervals.
  Final prediction intervals use quantiles of the empirical bootstrap forecast
  distribution convolved with known Gaussian measurement noise. A 20,000-draw
  comparison checks convergence to these integrated quantiles. Report truth RMSE, noisy RMSE/noise SD, interval widths and
  containment fractions over the full cube and a prespecified 24 mm sphere.
  Spatial fractions are descriptive, not independent coverage trials.
- **Negative controls:** time-zero-only fitting is rejected in every audit.
  Tests also reject absent directional information and a single tissue class.

The primary seed is fixed and never selected for interval containment. Separate
random streams serve the observed scans, bootstrap, coverage audit and noisy
forecasts; changing bootstrap counts does not change the primary scans or
coverage datasets.

## Improved predictive intervals and repeated coverage

```bash
uv run --locked python scripts/run_pat16_prediction.py --workers 8
make -C paper
```

This additional audit preserves the original recovery records. The final pointwise
95% prediction intervals integrate the known SD 0.015 Gaussian observation noise
over the 100 bootstrap forecast means. Increasing predictive draws from 100 to
1,000 and 20,000 checks tail convergence. These extra noise draws reuse the same
parameter vectors; only analytic noise integration removes predictive-noise Monte
Carlo error, while finite-bootstrap uncertainty remains.

Forty fresh datasets each receive independent training and future observations at
the known finer-time signal. Every dataset is refitted, then 100 new parametric
bootstrap datasets are generated, refitted and forecast. The legacy 100-draw,
20,000-draw and integrated methods are paired on the same bootstrap forecasts.
Withheld data are used only for scoring, without tuning interval width or tails.
Coverage and its dataset-level Student-t Monte Carlo interval are reported for the
full cube and near-center sphere at both withheld times. Exact binomial intervals
at a prespecified central voxel show the much larger uncertainty at one location.
Widths and proper 95% interval scores penalize over-wide intervals and misses.
Failures/boundaries are retained, with unusable procedures counted as misses.

The runner accepts `--trials`, `--bootstrap`, `--draws`, `--seed`, `--workers`,
`--recovery`, `--output` and `--fresh` (archives existing output before recomputing). Publication figure generation requires the documented
default settings. Matching checkpoints resume deterministically; use a new output
directory or `--fresh` after changing baseline files or numerical settings. The nested audit
requires real PAT16 derivatives and substantially more time than the original
recovery. Replaying its saved records and rebuilding the paper requires no MRI.

## Outputs

`outputs/pat16_recovery/` contains:

- `summary.json`: all settings, known values, estimates, intervals, correlations,
  coverage/binomial intervals, forecast errors, numerical checks and failures.
- `parameters.csv`: compact machine-readable parameter table.
- `recovery_data.npz`: clean/noisy/fitted scans; all parameter draws and trial
  intervals; forecast ensembles/bands; observation coordinates and region mask.
- `parameter_recovery.png`: estimates and confidence intervals against truth,
  plus the full log-parameter correlation matrix.
- `validation.png`: recovered values across independent trials and interval
  coverage with binomial uncertainty.
- `withheld_forecast.png`: clean signals, fitted forecasts and errors at the two
  withheld times.
- `provenance.json`: external input, source-code and output hashes, coordinate
  metadata and software versions. Source MRI is read in place and not copied.

`outputs/pat16_prediction/` additionally contains `primary.npz`, a hashed
`summary.json` manifest and `trials/trial_000.npz` / `.json` through trial 039.
Each checkpoint retains training and test observations, bootstrap parameters and
forecasts, all three interval sets, coverage/width/score/expected-containment
metrics and all failure/boundary flags. The paper generator checks independent
noise streams, mixture CDF endpoints and aggregate uncertainty before building
`paper/figures/pat16_prediction_validation.pdf` and the coverage table.

The recorded [results overview](../outputs/pat16_recovery/README.md) summarizes
the default run. To check the implementation:

```bash
uv run --locked pytest
```

These confidence intervals condition on known segmentation, geometry, dMRI
directions/eligibility, normalized initialization and observation calibration.
Uniform capacity cancels from the normalized dynamics; absolute capacity and
cell counts are not identified.
They omit errors in those assumptions and shared spatial/model discrepancy.
The PAT16 pre/post-operative pair spans surgery; it is not an untreated-growth
validation dataset. This experiment validates synthetic recovery against
assigned values, not biological patient rates, functional outcomes or clinical
predictions. The original PAT16 prior-sensitivity pilot and manuscript results
remain separate.
