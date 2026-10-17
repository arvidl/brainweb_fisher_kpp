# BrainWeb tumor growth explorer

For the reviewed 17 October 2026 checkpoint, backup restoration and complete
rerun instructions, see [PROJECT_CHECKPOINT.md](PROJECT_CHECKPOINT.md).

A standalone 3D tumor-growth demonstration with an interactive Jupyter notebook,
a conservative numerical solver, and PowerPoint/Google Slides media exports.
Choose the original Fisher–KPP model or coupled viable tumor, oxygen and
necrosis, with an optional matching Fisher–KPP reference.
The BrainWeb examples include source, data, a `uv` project and lockfile, tests,
and generated results. The optional PAT16 multimodal pilot reads external real
MRI derivatives; see its input instructions below. The parent repository and
its Conda environment are not used.

## PAT16 multimodal pilot

[pat16_anisotropic_pilot.ipynb](pat16_anisotropic_pilot.ipynb) compares isotropic
and dMRI-directed Fisher–KPP growth on **real pre-operative PAT16 MRI anatomy**.
It matches local mean diffusivity between models, propagates 32 paired,
explicitly assumed parameter scenarios, and overlays measured resting-state
fMRI connectivity. Time, plane, slice and uncertainty-quantile controls are
synchronized. Regional simulated tumor overlap is reported with uncertainty.

This is conditional simulated growth with assumed rates, not patient parameter
recovery or clinical prediction. It requires the processed source project's T1,
tissue maps, lesion mask, diffusion tensor, registration and BOLD/confounds;
missing imaging is never replaced with synthetic data. See
[input and execution instructions](examples/pat16_pilot.md),
[day-90 comparison](outputs/pat16_pilot/day90_comparison.png) and
[uncertainty/regions](outputs/pat16_pilot/uncertainty_and_regions.png).

A separate [PAT16 synthetic parameter recovery notebook](pat16_synthetic_parameter_recovery.ipynb)
now fits WM diffusion, GM diffusion, proliferation and anisotropy to generated
longitudinal signals on that real anatomy. It reports confidence intervals,
known-value checks and withheld forecasts. Its prediction intervals now compare 20,000 predictive draws with Gaussian-noise integration and a 40-dataset nested coverage audit (100 new bootstrap fits per dataset); see [the recovery guide](examples/pat16_recovery.md).

At day 90, the 32 scenarios give:

| Simulated quantity | Isotropic median [5th, 95th] | Anisotropic median [5th, 95th] | Paired difference median [5th, 95th] |
|---|---|---|---|
| Normalized burden (mL equivalents) | 110.42 [86.30, 137.58] | 106.74 [85.28, 133.17] | −2.98 [−6.80, −0.85] |
| Volume with u ≥ 0.16 (mL) | 167.48 [142.55, 199.30] | 161.06 [138.24, 195.44] | −6.36 [−12.83, −2.46] |

These are **assumed-parameter sensitivity percentiles**, not fitted patient
intervals. Paired differences are computed within scenarios and need not equal
the difference of marginal medians. The threshold is illustrative; normalized
burden is neither observed MRI volume nor an absolute cell count.
The paper includes the methods, results, two new figures and technical appendix.
[Compact published results](outputs/pat16_pilot/README.md) retain all-scenario
metrics, all-time field summaries, measured FC, regional CSV and checksum
provenance; raw MRI and the full local state cache are excluded.

## Brief history and clinical perspective

**Fisher–KPP** stands for Fisher–Kolmogorov–Petrovskii–Piskunov. In 1937,
[Fisher](https://doi.org/10.1111/j.1469-1809.1937.tb02153.x) and
[Kolmogorov–Petrovskii–Piskunov](https://math.i-learn.unito.it/mod/resource/view.php?forceview=1&id=73373)
developed reaction–diffusion descriptions of biological spread. Glioma models
later combined cell proliferation with invasion through brain tissue;
[Swanson et al. (2000)](https://doi.org/10.1046/j.1365-2184.2000.00177.x)
distinguished migration in gray and white matter. Extensions introduced
[oxygen, hypoxic phenotypes and necrosis](https://doi.org/10.1007/s11538-012-9786-1).

Recent work connects these models to measurements. A
[2025 study](https://doi.org/10.1186/s12885-025-14557-3) used serial MRI and
data assimilation to forecast treatment response in 21 patients. In 2026,
[MRI-derived cellularity, perfusion and hypoxia habitats](https://doi.org/10.1016/j.mri.2026.110744)
showed high test–retest repeatability in 24 glioma-bearing rats. A
[poromechanical-model preprint](https://arxiv.org/abs/2605.05013) reported
held-out tumor-volume errors of 4.7–36.0% in four rats; a
[September 2026 preprint](https://arxiv.org/abs/2609.22363) used simulated
biopsies to show that infiltration length can be estimated without tumor age,
whereas growth speed requires age information.

The clinical motivation is to estimate infiltration beyond visible MRI and
support individualized growth forecasts and treatment planning. Translation
requires identifiable parameters, uncertainty estimates and independent,
prospective validation. This repository provides a reproducible research
example: at day 90, oxygen–necrosis coupling reduced viable cell count by about
14.8% versus Fisher–KPP, while the volume above 20% of carrying capacity changed
by only about 1.9%. These BrainWeb-phantom results use assigned parameters and
are not patient predictions; see the manuscript for numerical limitations.

## Roadmap

These are next steps, building on the phantom demonstrations and the conditional
PAT16 multimodal pilot:

1. **Estimate parameters from multimodal MRI.** Register serial T1, T1-Gd,
   T2/FLAIR and dMRI, with a calibrated observation model for each sequence.
   Build on the PAT16 pilot's tissue-dependent and dMRI-directed transport;
   estimate migration magnitudes and proliferation from longitudinal data.
   Water diffusion is not tumor-cell
   migration. Enhancement and FLAIR are not universal cell-density contours.
   Add oxygen/death parameters only with independent physiological information
   or justified priors; routine BOLD does not measure tissue oxygen concentration.
2. **Quantify uncertainty.** Extend the homogeneous, BrainWeb and PAT16 recovery audits
   across rates, seed locations and scan schedules, then infer parameter
   distributions from calibrated measurements. Propagate measurement,
   segmentation, registration, tract-direction, initialization, prior, model and
   numerical uncertainty into
   forecast intervals and spatial maps. Check interval coverage and sensitivity
   to priors, following work on
   [glioma-model identifiability](https://doi.org/10.1098/rsta.2024.0212).
3. **Validate progressively.** Fit early scans and predict withheld later scans,
   comparing separately calibrated coupled and Fisher–KPP models with simple
   baselines. Validate oxygen and necrosis against independent measurements.
   Progress to external cohorts and prospective, blinded forecasts, reporting
   spatial/volume errors and uncertainty calibration; distinguish untreated
   growth from postoperative and treatment response. Use localized pathology
   for biological targets and patient-level splits for cohort evaluation.
4. **Connect growth to a brain digital twin.** Link evolving tumor states to
   biomechanics, tract involvement, structural connectivity and neural dynamics,
   building on the related
   [Finse 2026 prototype](https://github.com/arvidl/glioma-brain-digital-twin-public).
   Use fMRI to constrain functional/hemodynamic models and test later functional
   and cognitive outcomes. Account for
   [neurovascular uncoupling](https://doi.org/10.1148/radiol.2019190089); weak BOLD
   does not establish loss of function. Feedback motivated by
   [neuron–glioma interactions](https://doi.org/10.1038/s41586-023-06036-1) remains
   a hypothesis requiring separate evidence, rather than a rate map derived
   directly from resting-state connectivity.
5. **Model treatment explicitly.** Add resection, radiation and drug effects,
   including oxygen-dependent radiosensitivity and treatment-related tissue
   injury. Evaluate viable burden and recurrence patterns alongside normal-brain
   toxicity, tract/network function and plan deliverability. Serial MRI could
   update forecasts during
   treatment, as explored in
   [MRI-based data assimilation](https://doi.org/10.1186/s12885-025-14557-3).
6. **Evaluate new strategies.** Use virtual cohorts to compare adaptive radiation
   schedules, spatial dose painting, hypoxia-modifying combinations and treatment
   sequencing under uncertainty. Select informative scan times and prioritize
   robust treatment hypotheses for experimental testing. Prospective controlled
   trials must establish effects on survival, neurological function and quality
   of life before model-guided strategies can claim clinical benefit.

The [BTC dataset](https://doi.org/10.1038/s41597-022-01806-4) underlying the
related prototype contains T1, dMRI and resting-state fMRI; T1-Gd, T2 and FLAIR
would be additions. Its preoperative cohort comprises **11 glioma patients,
14 meningioma patients and 11 controls**, with incomplete follow-up around six
months after surgery. These tumor types need separate analyses. The new PAT16
notebook implements anatomical integration, directional transport and static
measured fMRI overlays with assumed growth-rate uncertainty. It does not model
the observed resection or predict network dynamics; that scan pair cannot
directly measure untreated growth. Serial tumor MRI and treatment metadata
are needed for growth estimation. Coupled mechanics, neural dynamics and
clinical validation remain proposed extensions. The manuscript now reports the
PAT16 pilot separately from the BrainWeb simulations and synthetic recovery.

## Synthetic parameter recovery

Open [synthetic_parameter_recovery.ipynb](synthetic_parameter_recovery.ipynb)
and **Run All** for two synthetic inference experiments. **Part A** estimates
diffusion D and proliferation ρ from noisy MRI-like observations of a homogeneous
3D phantom. Known values are D = 0.300 mm²/day and
ρ = 0.045 day⁻¹. A finer simulation generates blurred, voxel-averaged signals;
scans at days 20/40/60 are fitted and days 75/90 are withheld for forecasting.

The notebook includes multiple starting guesses, local 95% confidence intervals,
100 parametric bootstrap fits, forecast intervals and 60 independent noise
realizations to check recovery bias and local-interval coverage. It also measures
numerical discrepancy and rejects an unidentifiable time-zero-only experiment.
The initial state, imaging calibration, capacity and noise level are known:
these conditional uncertainty estimates demonstrate synthetic recovery, not
patient MRI validation. No Bayesian prior is used.

**Part B** estimates three independent rates on the **full BrainWeb anatomy**:
white-matter diffusion D_WM, gray-matter diffusion D_GM and a shared proliferation
rate ρ. Known values are **0.500 mm²/day, 0.100 mm²/day and 0.040 day⁻¹**.
There is no fixed WM:GM ratio. Gray/glial labels share D_GM; CSF and extracerebral
tissue have zero transport and growth. The existing harmonic face-flux solver
evolves the full 3 mm tissue domain, while a fixed 60 mm observation cube supplies
blurred, 6 mm voxel-averaged cellularity proxies. The BrainWeb T1 provides context;
the tumor signals are generated synthetically, not measured from that T1 image.

Part B has its own 100 bootstrap fits, 60 independent noise trials, three-rate
correlations, known-value checks and withheld forecasts. The primary generator
shares the 3 mm anatomy but uses a finer time step. A separate **2 mm anatomy/grid
stress test** quantifies sensitivity to the shared spatial discretization and
reports noise-free rate shifts as well as a paired noisy fit. This changes both
the mesh and tissue geometry; it is not a pure convergence study. Forecasts are
checked over the full observation region and a prespecified near-seed region.
Known tissue labels and initialization remain strong assumptions. The notebook
also rejects time-zero-only inference, and tests reject fitting two tissue rates
in a domain with only one tissue class.

The recorded Part B realization gives:

| Parameter | Known | Estimate | Bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| D_WM (mm²/day) | 0.500 | 0.498063 | [0.450332, 0.548917] |
| D_GM (mm²/day) | 0.100 | 0.082643 | [0.055430, 0.107485] |
| ρ (day⁻¹) | 0.040 | 0.039479 | [0.038857, 0.039920] |

The true proliferation rate falls just outside both the local and bootstrap
95% intervals for this one noise realization; the seed is retained. Across 60
independent noise trials, local-interval coverage is 57/60, 55/60 and 59/60 for
D_WM, D_GM and ρ, with exact binomial intervals reported in the notebook.
Gray-matter diffusion has the broadest relative uncertainty. In the separate
noise-free 2 mm anatomy/grid stress test, fitting with the 3 mm model shifts
D_GM to 0.114350 (+14.35%); the primary noise-based intervals do not quantify
this anatomy/discretization uncertainty.

```bash
uv run --locked jupyter lab synthetic_parameter_recovery.ipynb
# Or execute and save the notebook without opening Jupyter:
uv run --locked python scripts/run_recovery_notebook.py
```

Allow roughly 3–5 minutes for both default experiments after compilation. Each
part can also be run separately. The executed notebook and recorded results for
[Part A](outputs/synthetic_recovery/summary.json) and
[Part B](outputs/brainweb_recovery/summary.json) include parameter estimates,
uncertainty, withheld errors, coverage diagnostics and source/data hashes. Arrays
and five/six figures respectively are saved alongside the summaries. Increase
the replication counts for a more precise uncertainty audit; the defaults keep
the demonstration practical. The coverage audit evaluates local Jacobian
intervals; it does not establish bootstrap coverage.

### Four-parameter recovery on PAT16 anatomy

[pat16_synthetic_parameter_recovery.ipynb](pat16_synthetic_parameter_recovery.ipynb)
extends recovery to the real PAT16 pre-operative tissue geometry and dMRI
directions. It independently estimates **D_WM, D_GM, shared proliferation ρ,
and the eligible-WM parallel/perpendicular migration ratio r**. The generating
values are **0.500 mm²/day, 0.100 mm²/day, 0.040 day⁻¹ and 2.500**.
Directional transport preserves local mean diffusivity while changing r;
GM and WM without valid directions stay isotropic.

Synthetic blurred, voxel-averaged cellularity proxies at days 20/40/60 calibrate
the fit. Days 75/90 are withheld. Three optimization starts, local 95% confidence
intervals, **100 parametric bootstrap fits**, **60 independent noise trials**,
parameter correlations, exact binomial intervals for local coverage, and
signal/observation forecast intervals are saved. A noise-free matched-model fit
must recover the known values; a finer-time generator checks temporal mismatch.
Time-zero-only inference, missing directional information and missing tissue
classes are rejected as unidentifiable.

The notebook replays hash-checked recorded results without external MRI by
default. Set `RERUN = True` to refit with the real derivatives, or run:

```bash
# From brainweb_fisher_kpp; uses PAT16_SOURCE or the existing local source project
uv run --locked python scripts/run_pat16_recovery.py
# Replay records and save the executed notebook (no source MRI needed):
uv run --locked python scripts/run_pat16_recovery_notebook.py
# Refit within the notebook and save its executed outputs:
uv run --locked python scripts/run_pat16_recovery_notebook.py --rerun
```

Allow several minutes for the full audit. [Recorded results](outputs/pat16_recovery/README.md)
include a CSV parameter table, JSON summary/provenance, complete resampling
arrays and three figures. The [implementation and assumptions](examples/pat16_recovery.md)
distinguish fitted conditional confidence intervals from the original PAT16
pilot's assumed-parameter sensitivity bands. Anatomy, initial density,
directions and imaging/noise calibration remain known; spatial/model errors
are not included. These are synthetic recovery results, not fitted patient
rates or validation against longitudinal patient tumor measurements. The
manuscript now includes the PAT16 recovery Methods, Results, technical appendix,
four validation/uncertainty figures and four numerical tables, generated from
these checked records.

## Scientific manuscript

The [paper (PDF)](paper/output/pdf/brainweb_fisher_kpp_manuscript.pdf) and
[LaTeX/BibTeX sources](paper/README.md) report the 90-day coupled comparison,
controlled ablations, numerical refinement, homogeneous two-rate recovery and
BrainWeb three-rate recovery with anatomy/grid sensitivity, PAT16 four-parameter
recovery with conditional uncertainty and known-value validation, and the separate
PAT16 MRI-informed directional pilot. The recovery Methods
and Results include local/bootstrap confidence intervals, independent noise
trials, regional withheld-scan forecasts and the retained proliferation interval
miss. The Discussion addresses the +14.35% GM rate shift under anatomy/grid
mismatch. Dedicated appendices document both imaging operators, optimization,
resampling and sensitivity designs. Figures and tables are generated
from recorded arrays and metrics. The anatomical simulations use assigned
parameters; the separate recovery experiment fits synthetic observations.
No patient fitting or biological validation is claimed. Build with `make -C paper`,
or rerun the forward cases and both recovery experiments with
`make -C paper reproduce` after `uv sync --locked`. Use `make -C paper recovery`
to rerun both notebook recovery sections. The build verifies each recovery
experiment's source/code/data hashes before generating manuscript assets.
`make -C paper pat16-recovery` reruns the additional MRI-dependent PAT16 recovery
audit; ordinary PDF builds use its recorded results without source MRI.
The Discussion also sets out multimodal MRI observation roles and a proposed
growth--brain digital twin, with explicit BTC data limits, functional validation
targets and treatment-evaluation perspectives. PAT16 Methods, Results and a
technical appendix distinguish measured inputs, assumed priors, numerical
verification and untested clinical/functional predictions. `make -C paper pat16`
reruns and exports the MRI-dependent pilot when `PAT16_SOURCE` is available;
ordinary PDF builds use checked public results without those external inputs.

## Start the notebook

In Terminal, change to this folder, then run:

```bash
uv sync --locked
uv run --locked jupyter lab tumor_growth_explorer.ipynb
```

Choose **Run All** in Jupyter. The application appears in the third cell. It uses
ipywidgets and the ipympl Matplotlib backend. Start Jupyter through this command
so the kernel uses this project's environment. VS Code can also use the local
`.venv/bin/python` kernel.

On macOS you can also double-click `launch_notebook.command` in Finder.

The existing environment is installed for this Mac. If you copy the folder to a
different location or computer, recreate `.venv` with `uv sync --locked`; virtual
environments themselves are not portable. A fresh install needs internet access
for Python packages (and Python 3.12 if absent). After installation, the notebook,
solver and animation generator run offline using the bundled data and ffmpeg.
`uv` itself must be installed separately if absent.

## Explore

- **Origin & density:** RAS x/y/z sliders or an MRI click place the seed. Choose
  three radii, a Gaussian or uniform ellipsoid, and peak density in cells/mm³.
  The center must be inside gray/white/glial tissue. Gaussian radii are σ and its
  support stops at 3σ; ellipsoid radii are semiaxes. Initial cells are restricted
  to gray/white/glial tissue.
- **Diffusion & growth:** set white/gray/CSF D in mm²/day, a smooth spatial
  multiplier, proliferation ρ in day⁻¹, and K in cells/mm³. The multiplier is
  centered at the seed and tends to one away from it. View the D map or tissue
  classes before running. A peak density independent of K makes K exploration
  meaningful; peak density must be ≤ K.
- **Model & oxygen:** Fisher–KPP remains the default. Select **Tumor + oxygen +
  necrosis** to enable all seven coupling parameters. The optional reference
  uses the same seed, grid, diffusion, proliferation and capacity. Coupled mode
  excludes CSF; the reference also disables CSF transport to match that domain.
- **Grid & time:** choose Δx, requested maximum Δt, duration, and saved-frame
  cadence. Automatic subdivision enforces diffusion stability. Disabling it
  rejects an unsafe Δt. Output times include t=0 and the exact final day.
- **Playback:** after running, use Play or the frame slider. With 3D storage
  enabled, the view sliders explore other slices. With it disabled, playback is
  through the seed's three planes. Editing model controls clears the old run.
- **Result view:** switch between viable tumor, oxygen, necrotic material and
  the reference, when available. Time and slice coordinates stay synchronized.
  The summary reports whole-volume metrics for the current day and both models.
- **Display → Side by side:** compare the coupled fields and optional reference
  in columns, with axial/coronal/sagittal slices in rows. One set of frame,
  playback and RAS slice controls updates every panel. **Planes** can show one
  orientation; all scales stay fixed at 0–1, with identical viable/reference
  colors. Collapse
  **Simulation parameters & export** to make more room for the comparison.
- **Export:** saves the current single-field or side-by-side display as MP4/GIF/poster,
  plus parameters, NIfTI fields, replay data, metrics and provenance. Four-field
  exports use the displayed slice coordinates and plane selection; single-field
  exports pass through the seed. Existing
  files in the selected output directory are replaced.

Open `coupled_example.ipynb` and **Run All** for a ready-to-use 90-day,
2 mm comparison with full 3D slice playback and the Fisher–KPP reference.

The default seed is at RAS (−30, 25, 30) mm in the left anterior brain. Parameters
are illustrative: D_WM=0.5, D_GM=0.1 mm²/day, D_CSF=0, ρ=0.04/day,
K=100,000 cells/mm³ and a 20,000 cells/mm³ Gaussian peak. The simulation lasts
180 days. The notebook saves every 3 days; the supplied presentation example
saves every 1.5 days for smoother playback. ρ is editable up to 0.3/day in the
GUI, including the 0.2/day used in the supplied slide example.

## Generate animations independently

The notebook does not need to be opened or executed:

```bash
uv run --locked python scripts/generate_animation.py \
  --config examples/presentation.json --dx 1 --output outputs/my_1mm_run
```

For a faster coarse run:

```bash
uv run --locked brainweb-kpp --dx 2 --days 90 --output outputs/quick
```

For a coupled demonstration with a dense initial seed and a matched reference:

```bash
uv run --locked brainweb-kpp --config examples/coupled.json --compare --dx 2 \
  --output outputs/coupled_demo --view viable
```

The example uses an 80,000 cells/mm³ Gaussian peak to demonstrate necrosis
early. In the GUI, choose coupled mode and set this peak for a similar run.
The default seed density remains 20,000 cells/mm³ for both models.

### Recorded 90-day comparison

Open [coupled_example.ipynb](coupled_example.ipynb) in the project's Jupyter
environment and choose **Run All** for the interactive four-field comparison.
The saved example is also available without running Python:

- [Side-by-side MP4](outputs/coupled_example/comparison.mp4) and
  [GIF](outputs/coupled_example/comparison.gif).
- [Slower 20-second MP4](outputs/coupled_example/comparison_slow_20s.mp4), with a
  six-second pause at day 90.
- [Day-90 axial close-up](outputs/coupled_example/day90_zoom_axial.png) and
  [all three anatomical planes](outputs/coupled_example/day90_zoom_all_planes.png).
- [Coupled metrics](outputs/coupled_example/growth_metrics.csv),
  [matched Fisher–KPP metrics](outputs/coupled_example/reference_fisher_kpp/growth_metrics.csv),
  [parameters](outputs/coupled_example/parameters.json), and
  [run provenance](outputs/coupled_example/run_metadata.json).
- [Paper experiment records](paper/data/experiments.json), including oxygen-only
  and fixed-supply ablations and time/grid refinement.

![Day-90 coupled and Fisher–KPP comparison](outputs/coupled_example/day90_zoom_axial.png)

These are simulations on normal BrainWeb anatomy with assigned parameters.
The interactive/media necrosis color scale stops at `z/K = 1`; numerical values
can exceed one because this state counts cell equivalents rather than a tissue
volume fraction. The paper displays the full range and reports the sensitivity
of the small necrotic focus to grid spacing.

Render another field from the saved result without re-solving:

```bash
uv run --locked brainweb-kpp --replay outputs/coupled_demo \
  --output outputs/coupled_demo --view oxygen
uv run --locked brainweb-kpp --replay outputs/coupled_demo \
  --output outputs/coupled_demo --view necrosis
uv run --locked brainweb-kpp --replay outputs/coupled_demo \
  --output outputs/coupled_demo --view comparison --fps 5
```

`--model fisher_kpp|coupled` overrides the JSON model. `--compare` is valid for
coupled simulations; `--view reference` renders that matched reference. All
biological and oxygen-solver parameters can be set in the JSON config.

`--view comparison` exports viable tumor, oxygen, necrosis and the reference
in four synchronized columns, with all three anatomical planes in rows (4:3).
Use `--plane Axial|Coronal|Sagittal` for a single row (16:9). Every panel uses
the same saved time index and fixed 0–1 scale. Comparison exports require the
coupled model and reference. The MP4/GIF include initial and final holds.

Re-render an existing simulation without solving again:

```bash
uv run --locked brainweb-kpp --replay outputs/example_1mm \
  --output outputs/reformatted --width 1280 --fps 10
```

Use `--formats mp4` or `--formats gif` to generate only one format, or
`--simulate-only` to save the numerical result without media. `--width` must be
at least 640 and divisible by 32. A copied JSON file can set any field in
`Parameters` (see `examples/presentation.json`). CLI Δx defaults to 1 mm; the
notebook defaults to 2 mm. Both call the same 3D solver. Rendering saved planes
does not require holding 3D history in memory.

### Supplied artifacts

`outputs/example_1mm/` contains:

| File | Purpose |
| --- | --- |
| `tumor_growth.mp4` | 1920 × 1080, H.264, yuv420p, silent, 12 fps, faststart |
| `tumor_growth.gif` | 960 × 540, shared palette, continuous loop |
| `poster.png` | Final frame for a slide thumbnail |
| `parameters.json` | Exact physical and numerical inputs |
| `run_metadata.json` | Grid, actual step, runtime, versions and data provenance |
| `media_metadata.json` | Dimensions, cadence and playback duration |
| `simulation.npz` | Plane histories, T1 backgrounds, times and metrics for replay |
| `final_density_cells_mm3.nii.gz` | Final 3D density with its physical affine |
| `growth_metrics.csv` | Total cells and 20%/80% K threshold volumes |

Coupled exports additionally contain `final_oxygen_relative.nii.gz`,
`final_necrosis_cells_mm3.nii.gz`, all three fields' plane histories, and the
parenchymal mask. Oxygen outside that mask is zero as an excluded-domain
sentinel, not an oxygen estimate. The viewer hides those voxels. Reference
results are saved in `reference_fisher_kpp/`. Original replays remain readable.
The viable movie/poster filenames are retained; other views use `oxygen.mp4`,
`necrosis.mp4`, `reference.mp4` and corresponding GIF/poster/metadata filenames,
so exporting another view does not replace the viable movie. Replay saves seed
planes and final fields; arbitrary 3D history is kept only in the live explorer.
Four-field exports use `comparison.mp4`, `comparison.gif`, `comparison_poster.png`
and `comparison_media_metadata.json`. Their metadata records the shared saved
times, rendered frame order, fields, slices and anatomical planes.

The MP4 is intended for insertion as video in PowerPoint. For Google Slides,
upload the MP4 to Google Drive and insert it as a Drive video. The looping GIF
can be inserted as an image in either application. Use slideshow/presentation
mode to check playback; the editor or static preview may only show one frame.
These files are media assets; no existing presentation is modified.

## Numerical model

The model is

```text
∂n/∂t = ∇·(D(x) ∇n) + ρ(x) n (1 − n/K(x))
(D ∇n)·ν = 0
```

It models simultaneous invasion and proliferation. It does not explicitly
implement a go-or-grow phenotype switch. There is no treatment, tissue
deformation/mass effect. This original explorer uses scalar transport; the
separate PAT16 notebook adds DTI-directed tensors. The original model has no
necrosis. D is an isotropic
scalar field within each tissue, optionally modulated smoothly in space.

Cell-centered fluxes use a symmetric harmonic mean of D at neighboring voxels.
This implements the divergence-form operator at heterogeneous tissue interfaces,
conserves cells under pure diffusion, and sets flux to zero at impermeable
interfaces. Domain-edge faces are closed. No periodic wrapping is used.

Each step uses explicit diffusion followed by the exact logistic reaction:

```text
a_ij = harmonic_mean(D_i, D_j) / h²
n*_i = n_i + Δt Σ_j a_ij (n_j − n_i)
n_new_i = n*_i / [exp(−ρ_i Δt) + (n*_i/K_i)(1 − exp(−ρ_i Δt))]
```

The safe positivity bound is `Δt <= 0.9 / max_i Σ_j a_ij`, reducing to
`0.9 h²/(6 D)` for constant diffusion in 3D. The method is first order in time;
the centered diffusion stencil is second order in smooth regions, but sharp
tissue interfaces and label resampling limit spatial accuracy. No density
clipping conceals instability. Output frames and solver steps are independent.

Diffusion and logistic flow preserve nonnegative density. Uniform K is an upper
bound when the initial field is ≤ K. Spatially varying K can be exceeded locally
by incoming diffusion; such results are not clipped.

The Python `simulate` API accepts optional `diffusion`, `proliferation` and
`capacity` coefficient arrays on the current grid. The GUI exposes tissue and
spatial diffusion with uniform tissue proliferation and uniform carrying
capacity. Extracerebral transport is always disabled, CSF reaction is zero,
and glial class 8 uses gray matter defaults. Custom maps are marked in metadata;
save those input arrays yourself for custom-map reproducibility. The standard
slide renderer's n/K scale uses the scalar K in `Parameters`, so use separate
visualization for a custom spatial K.

The media show axial, coronal and sagittal planes through the seed, a common
fixed inferno n/K scale, and whole-volume metrics. Overlay values below 0.02 K
are hidden. Volumes at 0.2 K and 0.8 K are illustrative thresholds and are not
calibrated MRI detection boundaries. The faint complete curves provide context;
the vertical cursor marks the current time.

## Coupled tumor / oxygen / necrosis model

The viable field `c` uses the existing density units (cells/mm³); `z` stores
immobile necrotic tumor material in cell equivalents/mm³. Relative oxygen `o`
has a normal-tissue baseline of one. The reduced model is

```text
φ = (c + z) / K                 [a]₊ = max(a, 0)
g(o) = min(1, o / θp)           h(o) = [1 − o / θd]₊
p = 1 / (1 + α φ)

∂c/∂t = ∇·(D ∇c) + ρ g(o) c [1 − φ]₊ − δ h(o) c
∂z/∂t = δ h(o) c − λz z
0     = ℓo² ∇²o + p (1 − o) − β (c/K) o
```

Both diffusion operators have closed/no-flux boundaries on GM/WM/glial tissue.
Oxygen is solved at initialization and after every tumor step. It is consumed
by viable tumor and restored by a distributed exchange term. Baseline normal
metabolism is implicit in the normal-tissue reference; there is no explicit
healthy-cell population. Necrotic material consumes no oxygen and limits births.
The perfusion closure is a phenomenological modeling assumption; setting α = 0
holds oxygen delivery fixed. This version has no dynamic vascular state,
phenotype switching, directed hypoxic migration, or mass effect.

| Parameter | Meaning | Units | Default |
| --- | --- | --- | --- |
| `oxygen_length_mm` (ℓo) | Effective diffusion/exchange length | mm | 0.2 |
| `oxygen_uptake_ratio` (β) | Tumor uptake / baseline restoration | dimensionless | 4 |
| `perfusion_suppression` (α) | Delivery reduction with total burden | dimensionless | 4 |
| `oxygen_prolif_level` (θp) | Oxygen for unrestricted proliferation | relative oxygen | 0.3 |
| `oxygen_death_level` (θd) | Threshold for hypoxic death | relative oxygen | 0.1 |
| `necrosis_rate` (δ) | Maximum transfer rate to necrosis | day⁻¹ | 0.5 |
| `necrosis_clearance` (λz) | Necrotic removal rate | day⁻¹ | 0 |

Require `0 < θd < θp < 1`. All new defaults are illustrative, not calibrated
patient parameters. Necrotic equivalents do not measure all injured host tissue.
At millimeter grids, oxygen is a voxel-averaged availability proxy; the default
0.2 mm exchange length and individual capillary gradients are not resolved.
`K` regulates births rather than enforcing hard packing: incoming diffusion may
exceed `c+z=K`. Densities are not clipped.

The oxygen system is `[-ℓo² L + diag(p + β c/K)] o = p`. A compact matrix-free
stencil with Jacobi-preconditioned conjugate gradients avoids a full-grid sparse
matrix. `oxygen_rtol=1e-7` and `oxygen_maxiter=200` can be changed in JSON/Python;
failed convergence raises an error. Solver iterations and maximum relative
residual are recorded. Tiny oxygen bound deviations at solver tolerance are
not hidden by clipping.

Tumor steps retain the existing conservative explicit diffusion bound. With
oxygen and necrosis frozen, births use exact logistic flow with
`K_eff=K-z`, `rho_eff=ρ g(o)(1-z/K)` whenever `z<K` and `c<K-z`; otherwise births
are zero. Exact exponential death transfers the same amount from `c` into `z`.
Clearance is exponential. This Lie splitting is first order; the diffusion
positivity limit does not establish reaction/coupling accuracy. Refine Δt for
quantitative comparisons.

With zero oxygen uptake and no initial necrosis, oxygen is one and the coupled
model recovers standard constant-K Fisher–KPP for initial density ≤ K. In Python:

```python
from brainweb_kpp import Parameters, load_brainweb, simulate
p = Parameters(model="coupled")
result = simulate(load_brainweb(2), p, compare=True)
# Existing final_density / planes / volumes remain the viable field.
# Also: final_oxygen, final_necrosis, oxygen_planes, necrosis_planes,
# oxygen_volumes, necrosis_volumes, tissue_mask, comparison.
```

Coupled metrics add necrotic cell equivalents, necrotic volume above 0.2 K,
parenchymal volume below θd, and minimum parenchymal oxygen. View scales are
fixed 0–1: inferno for viable density, viridis for oxygen, magma for necrosis.
Only density overlays hide values below 0.02 K. No threshold is an MRI boundary.

Biological motivation: Martínez-González et al. (2012), *Hypoxic Cell Waves Around
Necrotic Cores in Glioblastoma*, [DOI](https://doi.org/10.1007/s11538-012-9786-1).
The three-field closure here is a proposed reduction, not that paper's full model.

## Data and provenance

The bundled T1 is `T1 ICBM normal 1mm pn0 rf0`, copied without modification from
the repository's existing BrainWeb NIfTI. The matching **original normal**
discrete phantom was downloaded from BrainWeb and converted from MINC to NIfTI
with nibabel. T1 and labels are canonical RAS+, 181 × 217 × 181, 1 mm isotropic,
starting at (−90, −126, −72) mm. Affines match exactly. These are simulated
normal-anatomy images; no real tumor is present in the source data.

The label map supplies tissue ground truth: CSF=1, GM=2, WM=3, glial matter=8.
This avoids assigning tissue from T1 intensity thresholds. Other classes remain
outside the growth domain. T1 is resampled linearly, labels by nearest neighbor.
The coarse-grid upper edge may shift by less than one requested spatial step.

`data/manifest.json` records the source aliases and SHA-256 checksums. Notebook
startup and CLI verify those checksums. To restore missing NIfTI data:

```bash
uv run --locked python scripts/fetch_data.py
```

The script uses the official BrainWeb download form and converts MINC directly;
FreeSurfer is unnecessary. It leaves optional personal information blank. MINC
to NIfTI conversions may produce a different file encoding/checksum; review the
manifest when regenerating data. BrainWeb data retain their original terms and
attribution rather than inheriting the parent repository's software license.

Sources to acknowledge in presentations/publications:

- [BrainWeb simulated brain database](https://brainweb.bic.mni.mcgill.ca/brainweb/)
  and [normal anatomical model](https://brainweb.bic.mni.mcgill.ca/anatomic_normal.html).
- Collins DL et al. (1998). *Design and construction of a realistic digital brain
  phantom*. IEEE TMI 17(3), 463–468. [DOI](https://doi.org/10.1109/42.712135).
- Cocosco CA et al. (1997). *BrainWeb: Online Interface to a 3D MRI Simulated Brain
  Database*. NeuroImage 5(4), S425.
- Everett R et al. (2020). *A tutorial review of mathematical techniques for
  quantifying tumor heterogeneity*. MBE 17(4), 3660–3709.
  [DOI](https://doi.org/10.3934/mbe.2020207), referenced in the supplied slides.

## Performance on Apple Silicon

The solver uses float32 arrays and compiled Numba CPU kernels, up to eight
threads, without CUDA or GPU dependencies. It runs natively on an M4 Max.
The command-line generator stores only planes plus the final volume. Notebook
3D history costs about 28 MB per 1 mm frame (about 1.7 GB for 61 frames); 2 mm
history is roughly eight times smaller. Coupled history stores three fields;
a matched reference adds a fourth, for about 6.8 GB at 61 native-resolution frames. The 128 GB machine has ample memory for
these defaults. At 0.5 mm, both memory and the number of stable steps rise
substantially. An 8 GB history guard asks for coarser output if necessary.

Choose 2 mm while exploring, then rerun at 1 mm. For quantitative use, compare
smaller time steps and successively finer grids with the same physical seed
and coefficients. Anatomical resolution remains 1 mm even if Δx=0.5 mm.
Measured example runtime is recorded in `outputs/example_1mm/run_metadata.json`;
the first simulation also pays a Numba compilation cost. Rendering is separate.

## Verify

```bash
uv run --locked pytest -q
```

Tests also check uniform and nonuniform analytical oxygen solutions, no-flux
oxygen barriers, exact death transfer and clearance, occupied capacity, recovery
of Fisher–KPP, live-plus-dead conservation, coupled time refinement against an
independent RK4 solution, solver failure reporting, all field renders, and old/new
replay compatibility. Notebook checks exercise both models, all field views,
matched references, arbitrary slices, and plane-only playback.

Tests check exact logistic growth without diffusion, heterogeneous-diffusion
mass conservation and positivity, impermeable CSF/extracerebral boundaries,
absence of periodic wrapping, unstable-step rejection, the uniform capacity
bound, a reflecting-boundary cosine eigenmode, temporal refinement, exact output
times, parameter validation, and bundled BrainWeb alignment/tissue counts.
Inference tests also check noiseless two-rate recovery and withheld forecasts,
imaging mass conservation, covariance scaling with known noise, and rejection of
unidentifiable time-zero observations and invalid input. Three-rate tests check
independent tissue-rate recovery and withheld prediction, harmonic-interface
equivalence, transport conservation, physical voxel alignment across grids,
exclusion of CSF, and rank-deficiency rejection when a tissue class is absent.

The notebook is also executed as a smoke test; the actual GUI can be opened in
JupyterLab for clicks and playback. The generated MP4 and GIF are decoded and
visually inspected. These checks validate implementation behavior; they do not
establish patient-level biological validity.

To repeat the notebook and short widget simulation check:

```bash
uv run --locked python scripts/check_notebook.py
```
