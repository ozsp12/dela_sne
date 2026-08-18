# Paper experiments

This directory contains the numerical workflow supporting the manuscript.

`run_experiments.py` is the public entrypoint. It imports the installed `dela_sne` package and binds the paper experiment layer to the canonical LAC primitives before executing the study. `_experiment_body.py` contains manuscript-specific datasets, response-temperature rules, statistical diagnostics, tables, and plotting routines; it is not a second public LAC API.

The experiment layer reproduces the 30-realization synthetic study, fixed/adaptive/frozen/clusterwise temperature comparisons, convergence and cycle diagnostics, real-data benchmarks, deterministic-annealing control, and SNE density/response analyses.

## Environment

From the repository root:

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

## Run

```bash
python paper/experiments/run_experiments.py
```

## Output contract

Each run overwrites stable names:

- `assets/*.csv`: numerical results and summaries;
- `assets/source_images/*.png`: raster companion exports;
- `../figures/*.pdf`: vector figures included by `main.tex`.

Seeds 1729-1758 are used for the 30-realization experiments. Real datasets are loaded from scikit-learn and require no network request.

CI preserves the committed CSV baselines before execution and compares regenerated numeric values using explicit tolerances. It does not require byte-identical figures.
