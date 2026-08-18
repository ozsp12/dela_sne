# Manuscript

This directory is the self-contained source tree for the manuscript **Feature and Neighbor Ensembles in Locally Adaptive Clustering and Stochastic Neighbor Embedding: Response Functions and Perplexity Tolerance**.

## Structure

```text
paper/
├── main.tex
├── references.bib          # synchronized copy of ../references/references.bib
├── figures/                # generated vector figures consumed by LaTeX
└── experiments/
    ├── run_experiments.py  # canonical entrypoint
    ├── _experiment_body.py # paper-specific diagnostics and plotting
    ├── README.md
    ├── MANIFEST.txt
    └── assets/
```

The paper is self-contained with respect to LaTeX sources and bibliography, but the numerical driver intentionally imports the installed `dela_sne` package so that manuscript LAC results validate the same implementation distributed as software.

## Environment

From the repository root:

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python scripts/sync_bibliography.py --check
```

## Reproduce experiment assets

```bash
python paper/experiments/run_experiments.py
```

Experiment outputs use stable names. Git records history; timestamps and workflow identifiers are not encoded in manuscript asset names.

Numerical CSV baselines are compared with floating-point tolerances in CI. Figures are regenerated and checked for successful creation rather than byte identity because PDF/PNG bytes can change for renderer-level reasons that do not change the scientific result.

## Compile

From `paper/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`paper/main.pdf` is a build product and is not versioned. CI uploads the compiled PDF as an artifact; tagged releases should attach the PDF as a release asset.
