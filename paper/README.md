# Scientific manuscript

The working manuscript uses conventional Introduction, Methods, Results and
Discussion sections, followed by technical appendices and a linked bibliography.
It reports the saved 90-day BrainWeb comparison plus controlled ablations and
time/grid refinement runs. All biological parameters are **assigned illustrative
values**, not fitted patient estimates or empirical prior means. Parameter
inference and independent biological validation are proposed protocols.
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
- `data/`: experiment settings, metrics, lossless figure arrays, provenance,
  and the recorded 35-test result.
- `generated/`: numerical macros and table rows generated from the data.

Sources, figures and generated tables are tracked together in this directory.
The supplied figures and tables allow a direct LaTeX/BibTeX build without rerunning
the solver; the Makefile also rebuilds them from the recorded numerical data.

The three 2026 arXiv records are explicitly identified as preprints. The 2026
MRI repeatability article was published online in July, with a November issue
assignment. The original motivating illustration is cited and is not reproduced;
all manuscript figures were generated from the simulator's own numerical output.

## Build

From the repository root, with the locked Python environment installed and a
LaTeX distribution containing `latexmk`, pdfLaTeX and BibTeX on `PATH`:

```bash
uv sync --locked
make -C paper
```

This rebuilds figures and the paper from recorded experiment data. To rerun all
nine simulations, including the native 1 mm comparison, before building:

```bash
make -C paper reproduce
```

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

The manuscript, notebooks, code and recorded results are maintained together in
<https://github.com/arvidl/brainweb_fisher_kpp>.
The experiments were generated before their source commit, so provenance records
working-file hashes alongside the base Git revision. Before journal submission, fill in
authorship, affiliations, funding, contributions, competing interests and any
required AI-use declaration, and add an immutable release identifier/archive DOI
to the availability statement if required by the journal.
