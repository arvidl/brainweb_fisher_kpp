"""Build the PAT16 synthetic recovery notebook; execution is a separate step."""
from pathlib import Path
import nbformat as nbf


root = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(r"""# Synthetic parameter recovery on PAT16 anatomy

Estimate **four separate, jointly fitted parameters** from synthetic longitudinal observations on the real pre-operative PAT16 tissue geometry and dMRI directions:

| Parameter | Known generating value | Meaning |
|---|---:|---|
| $D_{WM}$ | 0.500 mm²/day | WM mean tumor-cell migration diffusivity |
| $D_{GM}$ | 0.100 mm²/day | GM tumor-cell migration diffusivity |
| $\rho$ | 0.040 day⁻¹ | Shared logistic proliferation rate |
| $r$ | 2.500 | Eligible WM parallel/perpendicular migration ratio |

**Run All** in the locked project environment to replay the checked recorded results. Set **RERUN = True** below to generate and refit observations, or use the headless runner. The full audit fits scans at days **20, 40, 60** and withholds **75, 90**; it uses **100 parametric bootstrap fits** and **60 independent noise trials**. A further recorded prediction audit compares 20,000 noisy draws with Gaussian-noise integration and refits 40 independent datasets, each with 100 new bootstrap fits. The original recovery takes several minutes; the nested predictive audit is a separate, longer headless run (`scripts/run_pat16_prediction.py`). The original [PAT16 pilot](pat16_anisotropic_pilot.ipynb) remains a separate assumed-parameter sensitivity experiment.

The observations here are **generated cellularity proxies**, not measured patient tumor signals. Known initialization, anatomy, directions and imaging calibration make this a conditional recovery and uncertainty audit. It does not estimate this patient's biological rates or validate a clinical forecast. See [the recovery guide](examples/pat16_recovery.md).
""")
code("""from pathlib import Path
import os, sys, json
from dataclasses import asdict
ROOT = Path.cwd()
if not (ROOT / 'src/brainweb_kpp').is_dir():
    raise RuntimeError('Start Jupyter from the brainweb_fisher_kpp directory.')
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/brainweb-mpl')
from IPython.display import display, Markdown, Image
import numpy as np
from brainweb_kpp.pat16 import find_source, load_pilot, sha256
from brainweb_kpp.pat16_inference import PAT16RecoveryDesign
from brainweb_kpp.pat16_recovery import PAT16RecoverySettings, run_pat16_recovery

OUTPUT = ROOT / 'outputs/pat16_recovery'
SETTINGS = PAT16RecoverySettings()  # 100 bootstrap fits; 60 independent noise trials
DESIGN = PAT16RecoveryDesign()     # Full 3 mm anatomy; 72 mm observation cube
RERUN = False  # True refits; False explicitly replays checked records without external MRI
print(json.dumps(asdict(SETTINGS), indent=2))
""")
md(r"""## Forward model, observation operator and fitting

The full PAT16 tissue domain evolves with the same conservative 19-point tensor stencil and blocked-corner/no-flux boundaries as the pilot. The migration tensor in eligible WM is

$$D(x)=\frac{3D_{WM}}{r+2}\left[I+(r-1)v(x)v(x)^T\right].$$

Its local mean diffusivity is $D_{WM}$ for every $r$. GM and WM without valid directions remain isotropic. The two tissue rates have no fixed ratio or imposed ordering. The tensor family assumes prolate migration ($r\ge1$); its upper search bound 3.5 is imposed for stencil monotonicity. Measured water diffusion magnitudes are not tumor migration rates, and fMRI does not enter the likelihood.

The known initial density uses the lesion mask at threshold 0.5, Gaussian smoothing of 0.7 simulation voxels and amplitude 0.6. The **fixed 72 mm observation cube** is centered on the initial lesion centroid before generating or fitting observations. A 2 mm Gaussian blur precedes 6 mm voxel averaging with $3^3$ midpoint quadrature samples. Independent Gaussian noise with **known SD 0.015** is added after imaging, with no clipping. The normalized signal is an assumed calibrated proxy. Uniform capacity cancels in $u=c/K$; capacity and absolute cell counts are not identified.

The generator uses a **0.25-day step** and inversion uses **0.5 day** on the same 3 mm anatomy. Optimization minimizes known-noise Gaussian residuals in $(\log D_{WM},\log D_{GM},\log\rho,\log r)$ using three starts. Local confidence intervals invert the full-rank Jacobian information. Boundary optima are flagged; their Wald intervals cannot be treated as calibrated. Time-zero-only inference is rejected. Float64 snapshots preserve likelihood-derivative accuracy.
""")
code("""if RERUN:
    SOURCE = find_source()  # Or find_source('/absolute/path/to/processed/project')
    pilot = load_pilot(SOURCE, spacing=3.)
    print('Real PAT16 anatomy:', pilot.mask.shape, '; active tissue voxels:', int(pilot.mask.sum()))
    summary, arrays = run_pat16_recovery(pilot, OUTPUT, design=DESIGN, settings=SETTINGS,
                                        progress=lambda message: print(message, flush=True))
else:
    provenance = json.loads((OUTPUT / 'provenance.json').read_text())
    for name, expected in provenance['software_sha256'].items():
        if sha256(ROOT / 'src/brainweb_kpp' / name) != expected:
            raise RuntimeError(f'Stale recovery code: {name}; rerun the audit.')
    for name, expected in provenance['outputs_sha256'].items():
        if sha256(OUTPUT / name) != expected:
            raise RuntimeError(f'Stale recovery output: {name}; rerun the audit.')
    summary = json.loads((OUTPUT / 'summary.json').read_text())
    with np.load(OUTPUT / 'recovery_data.npz') as saved:
        arrays = {key: saved[key] for key in saved.files}
    sys.path.insert(0, str(ROOT / 'paper'))
    from make_pat16_recovery_figures import load_records
    from make_pat16_prediction_figures import verify_records, overlay_primary
    summary, arrays = load_records()
    prediction_summary, primary_prediction = verify_records(summary, arrays)
    overlay_primary(summary, arrays, prediction_summary, primary_prediction)
    print('Replaying checked recovery and nested predictive records; no new parameter fits were performed.')
""")
md(r"""## Parameter estimates and uncertainty

Local 95% Wald intervals use the known observation noise variance; they are not rescaled by the realized residual variance. Bootstrap datasets are generated from the **fitted early-scan model**, then refitted. The empirical 2.5th/97.5th percentiles form marginal 95% intervals. No known parameter is used to center these resamples, and no later scan enters optimization or resampling. These are frequentist conditional intervals, not Bayesian posterior distributions.

All four parameters, their correlations, boundary flags, and whether each interval contains the generating value are reported. A single successful interval check does not establish coverage.
""")
code("""rows = ['| Parameter | Known | Estimate | Error (%) | Local 95% CI | Bootstrap 95% CI | Contains known (local / boot) |',
        '|---|---:|---:|---:|---|---|---|']
for row in summary['parameters']:
    local, boot = row['local_interval'], row['bootstrap_interval']
    rows.append(f\"| {row['parameter']} | {row['known']:.6g} | {row['estimate']:.6g} | {row['relative_error_percent']:+.3f} | [{local[0]:.6g}, {local[1]:.6g}] | [{boot[0]:.6g}, {boot[1]:.6g}] | {row['local_contains_known']} / {row['bootstrap_contains_known']} |\")
display(Markdown('\\n'.join(rows)))
display(Image(filename=str(OUTPUT / 'parameter_recovery.png' if RERUN else ROOT / 'paper/figures/pat16_recovery.png')))
print('Jacobian condition number:', summary['jacobian_condition_number'])
print('Boundary optimum:', summary['at_bound'])
print('Multistart estimates:', [f['estimate'] for f in summary['multistart']])
print('Bootstrap successes / boundary fits:', summary['bootstrap_successful'], '/', summary['bootstrap_boundary_fits'])
""")
md(r"""## Independent known-value validation

Each of the 60 independent datasets receives fresh noise on signals generated at the **known** parameters and finer time step. This audits bias, RMSE and the **local** confidence intervals. Exact binomial intervals show Monte Carlo uncertainty in estimated coverage. Every attempted trial stays in the denominator; failed fits and boundary Wald intervals count as misses. All failures and boundary counts are saved. This experiment does not audit marginal bootstrap parameter-interval coverage.

A separate **noise-free matched-model fit** must recover all four known values within a maximum relative error of $10^{-4}$. A noise-free finer-time fit reports systematic shifts from temporal mismatch. Sharing spatial anatomy, the stencil and initialization makes this a controlled implementation audit, not independent model or spatial convergence validation.
""")
code("""rows = ['| Parameter | Trial mean | Bias (%) | RMSE | Local coverage | Exact 95% binomial CI |',
        '|---|---:|---:|---:|---:|---|']
for row in summary['parameters']:
    ci = row['local_coverage_binomial_interval']
    rows.append(f\"| {row['parameter']} | {row['noise_trial_mean']:.6g} | {row['noise_trial_bias_percent']:+.3f} | {row['noise_trial_rmse']:.6g} | {row['local_coverage_hits']}/{row['local_coverage_trials']} | [{ci[0]:.3f}, {ci[1]:.3f}] |\")
display(Markdown('\\n'.join(rows)))
display(Image(filename=str(OUTPUT / 'validation.png')))
print(json.dumps(summary['numerical_validation'], indent=2))
print('Fit failures:', summary['failures'])
print('Coverage boundary fits:', summary['coverage_boundary_fits'])
""")
md(r"""## Withheld forecasts

Days 75 and 90 are withheld from fitting and bootstrap estimation. Each bootstrap parameter vector predicts these scans. Pointwise **signal intervals** propagate parameter uncertainty; final **prediction intervals** integrate known Gaussian observation noise over 100 bootstrap forecast means. A 20,000-draw comparison checks convergence to this mixture. More noise draws improve predictive tails without adding distinct parameter vectors. Report RMSE against the noise-free known signal and normalized RMSE against noisy held-out scans. Interval containment fractions are shown for both the full cube and a prespecified 24 mm sphere around its center. These spatial fractions are descriptive; they are not independent repeated-trial coverage estimates.
""")
code("""for key, label in [('forecast_full_region', 'Full observation cube'), ('forecast_near_center', 'Near-center sphere')]:
    display(Markdown('**' + label + '**'))
    rows = ['| Withheld day | Signal RMSE | Noisy RMSE / SD | Truth in signal interval | Observation in prediction interval |',
            '|---:|---:|---:|---:|---:|']
    for row in summary[key]:
        rows.append(f\"| {row['day']:g} | {row['truth_rmse']:.6g} | {row['observation_rmse_over_noise_sd']:.3f} | {row['signal_interval_truth_fraction']:.3f} | {row['prediction_interval_observation_fraction']:.3f} |\")
    display(Markdown('\\n'.join(rows)))
display(Image(filename=str(OUTPUT / 'withheld_forecast.png')))
""")
md(r"""## Repeated prediction-interval coverage

The predictive audit uses **40 independent complete datasets**, each with a fresh four-parameter fit and **100 new parametric bootstrap fits**. Future noise is independent of training and resampling. All three methods use the same bootstrap forecasts in each trial: 100 noisy draws, 20,000 noisy draws and Gaussian-mixture quantiles. No withheld observations tune interval widths or tails.

Coverage is averaged spatial containment across independently refitted datasets, with dataset-level Student-t Monte Carlo 95% limits. Within-image voxels are not independent fit replicates. A prespecified central voxel has exact binomial limits. Width and the proper 95% interval score (width plus miss-distance penalties) accompany coverage. These checks concern the fixed synthetic model, not simultaneous image coverage or patient calibration.
""")
code("""if not RERUN:
    rows = ['| Method | Day | Region | Mean coverage | MC 95% interval | Mean width | Interval score |',
            '|---|---:|---|---:|---|---:|---:|']
    for row in prediction_summary['coverage']:
        ci = row['coverage_mc_interval']
        rows.append(f"| {row['method']} | {row['day']:g} | {row['region']} | {100*row['mean_coverage']:.2f}% | [{100*ci[0]:.2f}, {100*ci[1]:.2f}]% | {row['mean_width']:.5f} | {row['mean_interval_score']:.5f} |")
    display(Markdown('\\n'.join(rows)))
    print('Primary endpoint RMSE (100 / 1,000 / 20,000 draws / integrated):', prediction_summary['primary_endpoint_rmse'])
    print('Fixed-voxel binomial coverage:', prediction_summary['sentinel'])
    display(Image(filename=str(ROOT / 'paper/figures/pat16_prediction_validation.png')))
else:
    print('This refit regenerated the original recovery only. Run the separate nested prediction runner to rebuild its audit, using a new output directory if the baseline manifest changed.')
""")
md(r"""## Saved results and limits

`outputs/pat16_recovery/` contains `summary.json`, `parameters.csv`, `recovery_data.npz`, `provenance.json` and three figures. The NPZ preserves clean/noisy scans, estimates, all bootstrap draws, all independent-noise fits and local intervals, parameter correlations, forecast bands and region masks. The summary retains seeds, bounds, settings, every failure and numerical checks. Provenance records input, source-code and output hashes plus software versions. Raw input MRI is read in place and is not copied.

`outputs/pat16_prediction/` additionally retains primary predictive comparisons, 40 complete trial checkpoints and a hashed manifest. The paper generator validates these records and uses integrated intervals in its forecast figures/tables.

The uncertainty is conditional on fixed anatomy, normalized initialization, directions and imaging/noise calibration. Uniform capacity cancels; absolute capacity is not identified. It does not include segmentation, registration, fiber-direction, initial-state, spatial-discretization or model error. It does not turn a lesion mask into measured cellularity. The PAT16 pre/post-operative pair spans surgery and does not validate untreated growth. Patient inference requires longitudinal calibrated tumor observations and treatment metadata; no patient growth-rate or functional forecast is claimed here.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3 (brainweb-fisher-kpp)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
})
nbf.validate(notebook)
path = root/'pat16_synthetic_parameter_recovery.ipynb'
nbf.write(notebook, path)
print(path)
