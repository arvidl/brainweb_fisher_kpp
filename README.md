# BrainWeb Fisher–KPP explorer

A standalone 3D tumor-growth demonstration with an interactive Jupyter notebook,
a conservative numerical solver, and PowerPoint/Google Slides media exports.
Everything needed is inside this folder: source, BrainWeb data, a `uv` project
and lockfile, a local `.venv`, tests, and generated examples. The parent repository
and its Conda environment are not used.

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
- **Grid & time:** choose Δx, requested maximum Δt, duration, and saved-frame
  cadence. Automatic subdivision enforces diffusion stability. Disabling it
  rejects an unsafe Δt. Output times include t=0 and the exact final day.
- **Playback:** after running, use Play or the frame slider. With 3D storage
  enabled, the view sliders explore other slices. With it disabled, playback is
  through the seed's three planes. Editing model controls clears the old run.
- **Export:** saves MP4, GIF, poster, parameters, NIfTI density, replay data,
  metrics and provenance. Export planes always pass through the seed. Existing
  files in the selected output directory are replaced.

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
deformation/mass effect, necrosis, or DTI fiber anisotropy. D is an isotropic
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
history is roughly eight times smaller. The 128 GB machine has ample memory for
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

Tests check exact logistic growth without diffusion, heterogeneous-diffusion
mass conservation and positivity, impermeable CSF/extracerebral boundaries,
absence of periodic wrapping, unstable-step rejection, the uniform capacity
bound, a reflecting-boundary cosine eigenmode, temporal refinement, exact output
times, parameter validation, and bundled BrainWeb alignment/tissue counts.

The notebook is also executed as a smoke test; the actual GUI can be opened in
JupyterLab for clicks and playback. The generated MP4 and GIF are decoded and
visually inspected. These checks validate implementation behavior; they do not
establish patient-level biological validity.

To repeat the notebook and short widget simulation check:

```bash
uv run --locked python scripts/check_notebook.py
```
