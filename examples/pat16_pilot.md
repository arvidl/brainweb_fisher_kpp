# PAT16 multimodal Fisher–KPP pilot

Open [`pat16_anisotropic_pilot.ipynb`](../pat16_anisotropic_pilot.ipynb) and Run All.
It reads **real pre-operative** MRI derivatives and simulates conditional,
untreated growth for 90 days. It does not infer patient rates or validate a
clinical digital twin. The original BrainWeb explorer and oxygen–necrosis
model are unchanged.

## Run

From this repository, using the existing locked environment:

```bash
export PAT16_SOURCE=/absolute/path/to/processed/glioma-brain-digital-twin
uv run --locked jupyter lab pat16_anisotropic_pilot.ipynb
# Execute and save all outputs without opening Jupyter:
uv run --locked python scripts/run_pat16_notebook.py
```

The notebook also recognizes the locally available Finse project directory.
`PAT16_SOURCE` should identify the project root containing `data/` and
`derivatives/`, with the paths below. A missing required file raises an error;
there is no synthetic imaging fallback. This pilot does not rerun the expensive
MRI preprocessing pipelines. The public source project provides processing and
data-download instructions:
[glioma-brain-digital-twin-public](https://github.com/arvidl/glioma-brain-digital-twin-public).
Its small teaching releases may omit the tensor or full BOLD derivatives;
use the full processed outputs when those files are absent.

## Required inputs

All are for `sub-PAT16`, `ses-preop`. The exact relative paths are defined by
`source_paths()` in `src/brainweb_kpp/pat16.py`.

| Directory under the source root | Files |
|---|---|
| `derivatives/fmriprep/sub-PAT16/ses-preop/anat/` | Preprocessed T1, brain mask, dseg, **named** CSF/GM/WM probability maps |
| `data/BIDS_preop_postop/derivatives/tumor_masks/sub-PAT16/anat/` | `sub-PAT16_space_T1_label-tumor.nii.gz` |
| `derivatives/mrtrix_recon/sub-PAT16/ses-preop/` | `dti.mif` (embedded, uncompressed Float32LE, six tensor components) |
| `derivatives/registered_t1w/sub-PAT16/ses-preop/transforms/` | `acpc_to_t1w_mrtrix.txt` |
| `derivatives/registered_t1w/sub-PAT16/ses-preop/` | `fa_space-T1w.nii.gz`, `aparc+aseg_space-T1w.nii.gz` |
| `derivatives/fmriprep/sub-PAT16/ses-preop/func/` | T1-space preprocessed resting BOLD, BOLD JSON, confounds TSV |

The underlying BTC study is [Aerts et al., 2022](https://doi.org/10.1038/s41597-022-01806-4),
[OpenNeuro ds001226](https://openneuro.org/datasets/ds001226). Its T1, dMRI and
resting-state fMRI support this integration experiment. T1-Gd, T2 and FLAIR
are absent. A pre/post-operative pair spans surgery and cannot identify
untreated proliferation and migration. Local derivative hashes identify what
was actually used; filenames alone are insufficient provenance.

## Model and numerical implementation

Both models solve normalized Fisher–KPP with shared tissue-dependent rates,
initial density and no-flux tissue boundaries. Anisotropic WM transport uses
the principal dMRI direction and a prolate migration tensor. Its trace is
matched to the isotropic tensor at every voxel. Water diffusion magnitudes
are not used as cell migration rates. GM and WM with uncertain directions
remain isotropic; DTI cannot resolve crossing fibers.

The 19-point monotone tensor stencil retains all mixed derivative terms.
Nonnegative axis/face-diagonal coefficients, harmonic edge conductances and
an explicit CFL bound preserve density bounds. Each edge has equal and opposite
flux; diagonal shortcuts across non-tissue corners are blocked. Exact logistic
reaction follows each diffusion step. Anisotropy ratios above 3.5 are rejected
because this stencil cannot guarantee monotonicity for arbitrary orientations.
Larger ratios require another discretization. At ratio one the operator is the
existing harmonic face-flux reference.

MRtrix signed/permuted data layouts are read explicitly, including negative
zero. Tensor components use scanner RAS. The saved rigid matrix is a **T1-to-ACPC
pullback** despite its filename: sampling uses that matrix and reorientation
uses its inverse rotation. Registered FA consistency is checked. All scalar
resampling uses full affines even if source and target shapes happen to match.
Probability-map names determine tissue identity; this fMRIPrep dseg's integer
order differs from BrainWeb. Lesion voxels classified as CSF by the healthy-tissue segmenter are included
with explicitly assumed GM-like transport. These choices remain sensitivity
targets for subsequent work.

## Uncertainty and fMRI

The 32 paired Latin-hypercube scenarios vary WM/GM migration, proliferation,
anisotropy ratio, initial density, lesion threshold and a small direction
perturbation. All distributions are declared in the notebook. Reported bands
are empirical 5th–95th **prior-sensitivity percentiles**, not fitted confidence
intervals or posterior credible intervals. Increasing the ensemble size and
testing alternative priors are needed before interpreting tails.

The notebook recomputes 82-ROI correlations from actual preprocessed BOLD. It
resamples the Desikan–Killiany/subcortical atlas onto the BOLD grid, censors
high-motion and nonsteady frames, and jointly regresses motion, five aCompCor
components, supplied high-pass cosine terms, trend and intercept. It does not
reuse precomputed FC files with uncertain provenance. Static positive/negative
FC edges are projected through a labeled slice slab over simulated growth.
Regional mean simulated density is an overlap measure. Neither the edges nor
overlap predict cognitive decline, surgical eloquence or loss of function.
Neurovascular uncoupling, motion and atlas errors require further investigation.

## Outputs and checks

`outputs/pat16_pilot/` contains:

- `input_qa.png`: lesion alignment, tissue classes, principal dMRI axes and FC.
- `day90_comparison.png`: synchronized isotropic/anisotropic densities,
  paired differences and threshold-exceedance fractions.
- `uncertainty_and_regions.png`: prior-sensitivity curves and regional overlap.
- `summary.json`, `regional_exposure.csv`, `provenance.json`, `verification.json`.
- `sample_metrics.npz`: all 32 paired scenario metrics, rate draws and measured FC.
- `summary_fields.npz`: all-time field quantiles, paired median differences and
  threshold-exceedance fractions, with physical-coordinate metadata and one T1 slice.
- `functional_connectivity.csv`: labeled measured 82-by-82 correlations.
- `exports.json`: compact artifact hashes and executed notebook code hash.
- `ensemble_cache.npz`: full paired states, parameter draws and FC (local, Git-ignored).

Inputs are read in place, not copied. No new dependency is needed. Run:

```bash
uv run --locked pytest
```

Tests cover tensor trace and direction-sign invariance, mixed derivatives,
conservation, bounds, the isotropic limit, corner barriers, MIF layouts,
reorientation and affine-aware resampling. The notebook also repeats a central
scenario at half the time step. This verifies implementation and checks
temporal sensitivity; spatial convergence and independent patient validation
remain future work. The paper includes this pilot as a separate experiment; its
existing BrainWeb results are unchanged.

After rerunning the notebook, export the compact results and rebuild the paper:

```bash
uv run --locked python scripts/export_pat16_results.py
make -C paper
```

The export verifies input and code hashes. Paper figures and numerical tables
are regenerated from public compact records with hash and metric checks; the
PDF build needs neither the external MRI nor the ignored 64 MB full state cache.
`make -C paper pat16` combines notebook execution and compact export when the
required MRI derivatives are available.
