# PAT16 recorded pilot results

These files describe hypothetical untreated Fisher–KPP growth initialized from
real preoperative PAT16 MRI. Rates and initial normalized density are assumed;
there is no patient growth-rate fit, clinical validation or functional forecast.
Bands are empirical 5th/50th/95th sensitivity percentiles of 32 paired scenarios.

- `summary.json`: settings, all parameter draws and global outcome quantiles.
- `sample_metrics.npz`: every scenario's burden, threshold volume and regional
  mean simulated density at all seven saved times; parameters, time steps, FC
  and ROI IDs are also retained. Axes are scenario, model, time, then ROI where
  applicable; models are isotropic (0) and anisotropic (1).
- `summary_fields.npz`: pointwise density quantiles with axes quantile, model,
  time, active voxel; quantile values are 0.05/0.50/0.95. It also retains
  paired-median differences and anisotropic fractions exceeding u = 0.16
  (time, active voxel), active C-order indices, grid shape and RAS affine.
  These are scenario fractions, not calibrated probabilities of tumor presence.
  One axial T1/lesion slice, slice index and ROI/lesion centers reproduce the
  manuscript views; slice arrays are transposed for R-horizontal/A-vertical display.
- `regional_exposure.csv`: named 82-ROI mean simulated density quantiles.
- `functional_connectivity.csv`: labeled measured static ROI correlations, with diagonal
  set to zero for the graph representation (off-diagonal Pearson correlations).
- `provenance.json`: source-relative paths and SHA256 hashes of 14 MRI inputs,
  transformations, input QA, fMRI QC, code hashes and software versions.
- `verification.json`: density bounds, shared initialization, conservation and
  representative half-time-step check. Spatial convergence remains untested.
- `exports.json`: artifact checksums, notebook code hash and local-cache hash.
- Three PNG figures retain the input QA, day-90 comparison and sensitivity plots.

Raw MRI and the 64 MB `ensemble_cache.npz` containing individual 3D trajectories
remain local. Rerun the executed notebook with the required external derivatives,
then `python scripts/export_pat16_results.py` to regenerate compact exports.
No external MRI is needed for `make -C paper`: manuscript assets are regenerated
from the public compact results after hash and numerical checks.
See [input and execution instructions](../../examples/pat16_pilot.md).

The 90 days are simulated time from preoperative initialization; the experiment
does not simulate observed surgery or use postoperative imaging as an untreated
growth target. The illustrative density threshold is not calibrated to MRI.
Normalized burden (integral u dV) is expressed in mL equivalents and is neither
an absolute cell count nor a measured MRI volume. Static FC and regional overlap
cannot establish functional impairment or an alternative treatment outcome.
