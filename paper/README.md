# Scientific manuscript

The working manuscript uses conventional Introduction, Methods, Results and
Discussion sections, followed by technical appendices and a linked bibliography.
It reports the saved 90-day BrainWeb comparison, controlled ablations and
time/grid refinement, and two synthetic parameter recovery experiments.
The anatomical simulations use **assigned illustrative values**. The recovery
experiments estimate known homogeneous Fisher–KPP diffusion/proliferation and
separate BrainWeb white-/gray-matter diffusion with shared proliferation from
noisy MRI-like signals. Both report local/bootstrap confidence intervals,
withheld forecasts and independent repeated-noise coverage audits. The BrainWeb
experiment also tests sensitivity to changing anatomy/grid resolution. No patient
parameters or coupled oxygen/death parameters have been fitted. Measurement-based
patient inference and independent biological validation remain proposed work.
The Discussion includes a staged MRI-inference and uncertainty roadmap, external
and prospective validation, and proposed evaluation of adaptive radiation,
dose painting, hypoxia-modifying combinations and treatment sequencing.
Treatment effects and clinical benefit have not been simulated or demonstrated
in this study.

## Read and edit

- [Compiled paper (PDF)](output/pdf/brainweb_fisher_kpp_manuscript.pdf).
- [main.tex](main.tex) and [sections/](sections/): editable LaTeX source.
- [references.bib](references.bib): verified bibliography, including 2025–2026 literature.
- `figures/`: vector PDF figures and PNG previews.
- `data/`: anatomical experiment settings, metrics, lossless figure arrays,
  provenance and the recorded 50-test result.
- [Recovery records](../outputs/synthetic_recovery/summary.json): settings,
  source/data hashes, estimates, intervals, coverage and withheld errors;
  the adjacent NPZ contains observations and resampling arrays.
- [BrainWeb recovery records](../outputs/brainweb_recovery/summary.json):
  three-rate estimates, uncertainty, coverage, regional forecast metrics and
  the paired/noiseless 2 mm anatomy sensitivity fits, with arrays alongside.
- [Executed recovery notebook](../synthetic_parameter_recovery.ipynb):
  homogeneous Part A and BrainWeb Part B, with five and six supplementary
  figures respectively. Both experiments are incorporated into the manuscript.
- [BrainWeb recovery figure](figures/brainweb_recovery.pdf) and
  [anatomy sensitivity figure](figures/brainweb_anatomy_sensitivity.pdf): vector
  manuscript figures generated directly from verified records.
- `generated/`: numerical macros and table rows generated from the data.

Sources, figures and generated tables are tracked together in this directory.
The supplied figures and tables allow a direct LaTeX/BibTeX build without rerunning
the solver; the Makefile also rebuilds them from the recorded numerical data.

The three 2026 arXiv records are explicitly identified as preprints. The 2026
MRI repeatability article was published online in July, with a November issue
assignment. The original motivating illustration is cited and is not reproduced;
all manuscript figures were generated from recorded simulations and synthetic
observations. The bootstrap and SciPy references document the recovery methods.

## Build

From the repository root, with the locked Python environment installed and a
LaTeX distribution containing `latexmk`, pdfLaTeX and BibTeX on `PATH`:

```bash
uv sync --locked
make -C paper
```

This rebuilds figures and the paper from recorded data without refitting. The
recovery figure generator rejects mismatched source or array hashes. To rerun
all nine anatomical cases, including the native 1 mm comparison, and the
synthetic recovery notebook before building:

```bash
make -C paper reproduce
```

To rerun only synthetic recovery, then rebuild the paper:

```bash
make -C paper recovery
make -C paper
```

Each notebook experiment uses 100 bootstrap refits and 60 independent noise trials
by default. Each part records its own scoped code and data hashes. Both recovery
figure generators reject stale sources or arrays; the BrainWeb generator also
recomputes coverage, bias, forecast errors and numerical discrepancies.
Changing these or other settings in the notebook requires rerunning it before
rebuilding its manuscript assets. The coverage audit evaluates local Jacobian
intervals, not bootstrap coverage, and its binomial uncertainty is reported.

To rerun the software checks:

```bash
make -C paper check
```

The build accepts `PYTHON=/absolute/path/to/python` and `LATEXMK=/path/to/latexmk`
overrides. The bundled BrainWeb data are verified before experiments. Their
original attribution and terms apply independently of the software license.

## Interpretation and release status

The paper records the sensitivity of the tiny necrotic focus to spatial
resolution. Neither the 1 mm nor 2 mm grid resolves the assigned 0.2 mm oxygen
length. Necrotic cell equivalents can exceed the density scale locally and must
not be interpreted as a physical tissue volume fraction.

Synthetic recovery assumes a known seed, elapsed time, normalized signal scale,
capacity, blur and independent post-imaging Gaussian noise. The known rates
0.300 mm²/day and 0.045 day⁻¹ were recovered as 0.3033 and 0.04510. Both lie in
the reported bootstrap 95% intervals. This supports controlled two-rate recovery
and does not validate clinical imaging calibration or oxygen/necrosis inference.

BrainWeb recovery estimates D_WM = 0.498063 mm²/day, D_GM = 0.082643 mm²/day and
ρ = 0.039479 day⁻¹ from known values 0.500, 0.100 and 0.040. Both diffusion
truths lie in the local and bootstrap 95% intervals. The true proliferation rate
falls just outside both intervals in this one retained noise realization.
Independent local-interval coverage is 57/60, 55/60 and 59/60, with broad exact
binomial intervals; bootstrap coverage has not been audited. The day-90
prediction-interval voxel fraction is 94.2% over the whole FOV and 89.2% near the
seed, so background-inclusive metrics do not describe all local performance.

The primary generator and inverse model share the 3 mm anatomy but use different
time steps. Fitting noise-free 2 mm anatomy with the 3 mm model shifts D_GM by
**+14.35%**. This combines tissue geometry, mesh, seed sampling and discrete
imaging changes. It is not a pure convergence study or an estimate of segmentation
uncertainty, and the primary noise-based intervals do not include it. Methods,
Results, Discussion and a dedicated appendix distinguish these limitations from
conditional parameter recovery.

The manuscript, notebooks, code and recorded results are maintained together in
<https://github.com/arvidl/brainweb_fisher_kpp>.
The experiments were generated before their source commit, so provenance records
working-file hashes alongside the base Git revision. Before journal submission, fill in
authorship, affiliations, funding, contributions, competing interests and any
required AI-use declaration, and add an immutable release identifier/archive DOI
to the availability statement if required by the journal.
