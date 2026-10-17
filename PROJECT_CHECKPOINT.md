# Reviewed project checkpoint — 17 October 2026

The annotated tag `pat16-closeout-2026-10-17` preserves this working research
version. Its backup manifest identifies the exact commit, tree, file sizes and
SHA-256 hashes. Journal submission and an archival DOI remain separate steps.

## Contents and review

The checkpoint includes all versioned code, tests, notebooks, `pyproject.toml`,
`uv.lock`, manuscript sources, bibliography, generated tables, figures and the
final **52-page PDF** at `paper/output/pdf/brainweb_fisher_kpp_manuscript.pdf`.
It also includes the existing phantom/BrainWeb and PAT16 pilot records, and:

- `outputs/pat16_recovery/`: the four-parameter recovery, 100 bootstrap fits,
  60 independent local-interval trials, complete arrays and provenance.
- `outputs/pat16_prediction/`: the 20,000-draw comparison, Gaussian-mixture
  intervals, all 40 nested JSON/NPZ trial pairs (100 new bootstrap refits each),
  anatomy validation and execution log.
- `paper/data/pytest.xml` and both PAT16 claims-audit JSON files.

The final PDF's SHA-256 is
`8bb780611cc9b437d501eb93c54c0043dcbd08ac13288e0810fa765aa651b7a6`.

At closeout, the changed code, tests, documentation, manuscript integration and
file inventory were reviewed. Fresh checks gave **77 passing tests**, with no
failures or skips. The manuscript's PAT16 record-verification functions were
rerun without MRI or refitting: source/output hashes, independent noise streams,
intervals, all nested metrics, Gaussian-mixture CDF endpoints and coverage
uncertainty agree with the saved arrays. All 40 primary and 4,000 nested bootstrap
fits succeeded without boundary optima. All six recovery-notebook code cells
have execution counts and no saved errors. `git diff --check` is clean.
The generated recovery CSV retains its original hashed CRLF line endings;
`.gitattributes` preserves those bytes and permits them in whitespace checks.
The final PDF's layout was inspected at the final manuscript build; these PDF
bytes are preserved here. Review supports the recorded default design, rather
than certifying every alternative CLI setting or observation model.

No credential-pattern matches were found in changed text files. Source PAT16 MRI
is not included. The backup excludes `.venv`, ignored caches, temporary build
products, ignored PAT16 inputs and the full pilot state cache. Included BrainWeb
phantom data retain their existing attribution and access terms.

## Backup and restoration

The backup lives outside this checkout in the sibling `../project-checkpoints/`
directory. Each dated backup contains:

- `checkpoint.bundle`: self-contained Git history and refs, including the tag.
- `snapshot.tar.gz`: exactly the checkpoint files, prefixed `brainweb_fisher_kpp/`.
- `manifest.json`: commit/tag, environment and every file's size and SHA-256.
- `verification.json`: restore/check results.
- `SHA256SUMS`: checksums of the bundle, snapshot and backup metadata.
- `README.md`: backup-specific restore and verification instructions.

From an empty destination, replace the placeholder with the dated backup path:

```bash
CHECKPOINT_BACKUP="/absolute/path/to/project-checkpoints/dated-backup"
(cd "$CHECKPOINT_BACKUP" && shasum -a 256 -c SHA256SUMS)
git clone "$CHECKPOINT_BACKUP/checkpoint.bundle" restored-brainweb
git -C restored-brainweb checkout --detach pat16-closeout-2026-10-17
git -C restored-brainweb bundle verify "$CHECKPOINT_BACKUP/checkpoint.bundle"
git -C restored-brainweb fsck --full
```

Alternatively, extract `snapshot.tar.gz` into an empty directory for the files
without Git history. Both restore routes were checked against every file in the
manifest, including the PDF and all PAT16 trial records. The local backup shares
this computer's storage: copy the **complete dated backup directory** to
independent storage for protection against disk loss, then verify checksums there.
This checkpoint does not include a remote push or public release.

## Replay and validation without PAT16 MRI

Run commands from the repository root. Python must be **3.12**; `uv.lock` pins
the packages. The checked system used Python 3.12.2 and uv 0.13.1. Install `uv`
if necessary, then:

```bash
uv sync --locked
uv run --locked python -m pytest tests -q
```

Environment installation may need package-repository access. The backup contains
the lockfile, not Python, package wheels or TeX. Recorded scientific versions are
NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.0, NiBabel 5.4.2 and Matplotlib 3.11.2.

Check PAT16 records without modifying notebooks or manuscript assets:

```bash
MPLCONFIGDIR=/tmp/brainweb-checkpoint-mpl uv run --locked python - <<'PY'
import sys
sys.path.insert(0, 'paper')
from make_pat16_recovery_figures import load_records
from make_pat16_prediction_figures import verify_records
summary, arrays = load_records()
verify_records(summary, arrays)
print('PAT16 recovery and nested prediction records verified.')
PY
```

To replay and resave the executed recovery notebook:

```bash
uv run --locked python scripts/run_pat16_recovery_notebook.py
```

The notebook defaults to replay and does not recompute the 4,000 nested refits.
Jupyter execution requires local kernel-socket permissions.

## Rebuild the manuscript from saved records

Install a LaTeX distribution with `latexmk`, pdfLaTeX and BibTeX on `PATH`
(checked: Latexmk 4.86a, pdfTeX 1.40.27), then:

```bash
make -C paper
```

This verifies records, regenerates figures/tables and writes the manuscript PDF;
it needs neither external PAT16 MRI nor parameter refits. Hash-pinned code or
record changes cause validation to fail: reconcile experiments and provenance
before rebuilding instead of bypassing checks. PDF/figure creation metadata and
typography can change with the TeX environment; a rebuilt PDF needs numerical
and visual review. The manifest checks byte identity of the preserved original.

`make -C paper reproduce` reruns the self-contained phantom experiments and
recovery notebook before building; it does not recompute MRI-dependent PAT16 work.

## Recompute PAT16 with external derivatives

See [the pilot guide](examples/pat16_pilot.md) for inputs and processing, and
[the recovery guide](examples/pat16_recovery.md) for the design and CLI details.
`PAT16_SOURCE` must name the processed digital-twin project root, not this repo
or a single image. `source_paths()` in `src/brainweb_kpp/pat16.py` lists all 14
required files. The shared loader requires the full pilot derivative set,
including BOLD files, although BOLD is not used in the recovery likelihood.
Missing inputs raise an error; there is no synthetic anatomy fallback.

Relative input paths, hashes and geometry are preserved in
`outputs/pat16_recovery/provenance.json`. The prediction
`anatomy_validation.json` confirms matching anatomy and exact regeneration of the
known finer-time signal. Preserve or reacquire those derivatives under their
applicable access terms for a full rerun.

Recompute the default design in a **new temporary directory** to preserve the
frozen records:

```bash
export PAT16_SOURCE="/absolute/path/to/processed/glioma-brain-digital-twin"
PAT16_RERUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pat16-rerun.XXXXXX")
echo "$PAT16_RERUN_DIR"

uv run --locked python scripts/run_pat16_recovery.py \
  --source "$PAT16_SOURCE" --output "$PAT16_RERUN_DIR/pat16_recovery" \
  --bootstrap 100 --coverage 60 --seed 161026 --spacing 3

uv run --locked python scripts/run_pat16_prediction.py \
  --source "$PAT16_SOURCE" --recovery "$PAT16_RERUN_DIR/pat16_recovery" \
  --output "$PAT16_RERUN_DIR/pat16_prediction" \
  --trials 40 --bootstrap 100 --draws 20000 --seed 161027 --workers 8
```

Move completed temporary outputs to durable storage if retaining the rerun.
The recorded durations were 7.7 minutes for recovery and 25.7 minutes for the
nested audit with eight workers; other machines may differ. Matching prediction
checkpoints resume. After changing configuration/baseline files, choose a new
output directory or use `--fresh`, which archives old prediction outputs.
The recovery runner always recomputes into its selected output directory.

Manuscript generators read the default `outputs/pat16_recovery/` and
`outputs/pat16_prediction/` paths and require the documented default settings.
An isolated rerun does not update the manuscript. Promoting new records is a new
revision: retain this tag, compare results, rebuild assets, review the PDF and
save another checkpoint. Fixed seeds and the lockfile do not guarantee
byte-identical recomputed archives or floating-point results on every platform.

## Scientific scope and deferred work

Recovery uses synthetic observations on one fixed patient-derived anatomy.
Parameter intervals are marginal and conditional on known initialization,
directions and imaging/noise calibration. The 60-trial audit checks local
parameter intervals; the separate 40-trial nested audit checks predictive
intervals. Mean integrated full-cube coverage is 95.14% at day 75 and 95.03% at
day 90. Day-75 coverage is slightly conservative; fixed-voxel binomial limits
remain broad. These results do not establish simultaneous image coverage,
calibration at every location or patient forecasting performance.

Finite-bootstrap uncertainty, anatomy/direction/initial-state errors, correlated
or misspecified noise and shared spatial/model error remain. Future priorities
are broader anatomy/lesion designs, observation-model stress tests and spatial
convergence or independent-solver checks. No patient rates, clinical benefits
or functional outcomes are inferred.

Before submission, confirm secondary-use requirements; finalize authorship,
affiliations, funding, competing interests, contributions and any required AI-use
declaration; and prepare an immutable public release and archival DOI. Creating
this local checkpoint does not complete those steps.
