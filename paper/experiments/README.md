# Paper experiments

This directory contains the complete numerical workflow supporting the manuscript.

`run_experiments.py` reproduces the 30-realization synthetic LAC study, adaptive/frozen/clusterwise temperature comparisons, convergence and cycle diagnostics, real-data benchmarks, deterministic-annealing control, and SNE density/response analyses.

## Output contract

Each run overwrites stable names:

- `assets/*.csv`: numerical results and summaries;
- `assets/source_images/*.png`: raster companion exports from the same plotting routines as the manuscript figures;
- `../figures/*.pdf`: vector figures included directly by `main.tex`.

Run from the repository root:

```bash
python -m pip install -r paper/experiments/requirements.txt
python paper/experiments/run_experiments.py
```

Seeds 1729-1758 are used for the 30-realization experiments. The real datasets are loaded from scikit-learn and require no network request.
