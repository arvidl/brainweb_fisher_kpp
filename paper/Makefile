PYTHON ?= ../.venv/bin/python
LATEXMK ?= latexmk
PDF = output/pdf/brainweb_fisher_kpp_manuscript.pdf

.PHONY: all figures pdf experiments recovery reproduce check clean
all: pdf

figures:
	MPLCONFIGDIR=/tmp/brainweb-paper-mpl $(PYTHON) make_figures.py
	MPLCONFIGDIR=/tmp/brainweb-paper-mpl $(PYTHON) make_recovery_figures.py
	MPLCONFIGDIR=/tmp/brainweb-paper-mpl $(PYTHON) make_brainweb_recovery_figures.py

pdf: figures
	mkdir -p build output/pdf
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf $(PDF)

experiments:
	$(PYTHON) run_experiments.py

recovery:
	$(PYTHON) ../scripts/run_recovery_notebook.py

reproduce:
	$(PYTHON) run_experiments.py
	$(MAKE) recovery
	$(MAKE) pdf

check:
	$(PYTHON) -m pytest ../tests -q --junitxml=data/pytest.xml

clean:
	$(LATEXMK) -C -outdir=build main.tex
