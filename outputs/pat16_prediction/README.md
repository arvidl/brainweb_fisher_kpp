# PAT16 predictive intervals — recorded validation

Pointwise nominal 95% prediction intervals for synthetic observations on the fixed real PAT16 anatomy. These are conditional numerical checks, not patient forecasts.

The original four-parameter recovery and its confidence intervals are unchanged in [../pat16_recovery/](../pat16_recovery/README.md). Final prediction intervals integrate known Gaussian observation noise (SD 0.015) over 100 bootstrap forecast means. A 20,000-draw comparison checks convergence to this empirical Gaussian mixture; it reuses the same parameter vectors.

Forty independent complete datasets each refit early days 20/40/60 and 100 new parametric bootstrap datasets, then predict independent observations at days 75/90. All 40 primary and 4,000 bootstrap fits succeeded without a boundary optimum. The generator uses a maximum 0.25-day step and inversion 0.5 day. No withheld observations tune the tails or interval widths.

| Method | Day | Region | Mean coverage [MC 95% interval] | Mean width | Mean interval score |
|---|---:|---|---|---:|---:|
| legacy_100 | 75 | full | 93.37% [93.22, 93.53] | 0.05644 | 0.07251 |
| legacy_100 | 90 | full | 93.32% [93.14, 93.50] | 0.05643 | 0.07288 |
| legacy_100 | 75 | near | 93.16% [92.72, 93.60] | 0.05654 | 0.07339 |
| legacy_100 | 90 | near | 92.99% [92.53, 93.45] | 0.05649 | 0.07408 |
| mc_20000 | 75 | full | 95.13% [94.99, 95.28] | 0.05886 | 0.07007 |
| mc_20000 | 90 | full | 95.02% [94.86, 95.19] | 0.05889 | 0.07024 |
| mc_20000 | 75 | near | 94.84% [94.51, 95.17] | 0.05888 | 0.07023 |
| mc_20000 | 90 | near | 94.72% [94.31, 95.14] | 0.05887 | 0.07119 |
| gaussian_mixture | 75 | full | 95.14% [95.01, 95.28] | 0.05887 | 0.07008 |
| gaussian_mixture | 90 | full | 95.03% [94.87, 95.19] | 0.05890 | 0.07023 |
| gaussian_mixture | 75 | near | 94.91% [94.59, 95.24] | 0.05889 | 0.07023 |
| gaussian_mixture | 90 | near | 94.75% [94.35, 95.15] | 0.05889 | 0.07114 |

Monte Carlo limits use the independent dataset-level coverage fractions, with Student-t uncertainty. Voxels within an image share a fitted model and are not pooled as independent binomial trials. Lower proper interval score is better; it penalizes width and distance beyond either endpoint. Day-75 integrated full-cube coverage is slightly conservative, with the Monte Carlo lower limit just above 95%. This supports near-nominal average pointwise behavior under this design, not exact calibration at every voxel or simultaneous image coverage.

The prespecified central voxel has integrated coverage 39/40 (exact binomial 95% interval 86.8–99.9%) and 38/40 (83.1–99.4%) at days 75 and 90. These wide limits illustrate uncertainty in location-specific coverage.

For the primary withheld images, integrated intervals contain 94.85%/95.49% of full-cube measurements (widths 0.05886/0.05889), versus legacy containment 93.17%/93.11%. Endpoint RMSE relative to integration falls from 0.00383 (100 draws) to 0.00128 (1,000) and 0.00028 (20,000). Finite-bootstrap and structural uncertainty remain.

## Records and replay

- `summary.json`: settings, source/baseline/output hashes, aggregate and paired coverage uncertainty, widths, scores, expected containment and fixed-voxel binomial results.
- `primary.npz`: primary intervals and metrics at 100, 1,000, 20,000 draws and Gaussian integration.
- `trials/trial_000.npz` through `trial_039.npz` and adjacent JSON checkpoints: independent training/test data, fit/bootstrapped parameters, forecasts, all three intervals, metrics, failure and boundary flags.
- `anatomy_validation.json`: matching MRI input hashes, exact reproduction of the baseline known signal, runner hash and software versions. Raw MRI is not copied.

```bash
uv run --locked python scripts/run_pat16_prediction.py --workers 8
make -C paper
```

Matching checkpoints resume; `--fresh` archives the existing output before recomputing. Use it after changing the baseline or numerical settings. The fixed seed is 161027. The recorded run took 25.7 minutes with eight workers. Recomputing requires real PAT16 derivatives; manuscript generation and notebook replay use these checked records without MRI.

[Validation figure](../../paper/figures/pat16_prediction_validation.pdf) · [Methods and rerun guide](../../examples/pat16_recovery.md)
