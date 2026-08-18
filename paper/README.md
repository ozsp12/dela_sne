# Manuscript

This directory is the self-contained source tree for the manuscript **Feature and Neighbor Ensembles in Locally Adaptive Clustering and Stochastic Neighbor Embedding: Response Functions and Perplexity Tolerance**.

## Structure

```text
paper/
├── main.tex
├── references.bib
├── figures/                 # final PDF figures consumed by LaTeX
└── experiments/
    ├── README.md
    ├── MANIFEST.txt
    ├── requirements.txt
    ├── run_experiments.py
    └── assets/
        ├── source_images/   # PNG companion/source exports
        └── *.csv            # numerical outputs
```

## Reproduce the paper assets

```bash
python -m pip install -r paper/experiments/requirements.txt
python paper/experiments/run_experiments.py
```

The experiment driver overwrites stable artifact names. Git provides the historical record; file names do not accumulate timestamps or run identifiers.

## Compile

From `paper/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The repository workflow regenerates the experiment outputs, verifies that committed assets are current, and compiles the manuscript.
