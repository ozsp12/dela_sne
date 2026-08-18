# Paper experiments

This directory contains the reproducibility workflow and versioned assets for the numerical experiments supporting the manuscript **Feature and Neighbor Ensembles in Locally Adaptive Clustering and Stochastic Neighbor Embedding: Response Functions and Perplexity Tolerance**.

## Layout

- `run_experiments.py` — complete experiment driver.
- `requirements.txt` — Python dependencies.
- `MANIFEST.txt` — compact inventory of the reproducibility package.
- `assets/` — generated figures and numerical result tables used by the manuscript.

## Reproduce

```bash
python -m pip install -r paper_experiment/requirements.txt
python paper_experiment/run_experiments.py
```

The versioned assets are stored under `paper_experiment/assets/`. The experiment driver reproduces the 30-realization synthetic LAC validation, convergence and cycle diagnostics, frozen two-pass response rule, common-grid oracle comparison, plateau-width analysis, clusterwise-temperature experiment, feature-standardization sensitivity, real-data clustering benchmarks, deterministic-annealing control, and SNE density/response diagnostics.

The synthetic multiseed runs use deterministic seeds 1729--1758. The response selector and supervised oracle use the same 160-point logarithmic grid on `0.01 <= h <= 8.0`. Real datasets are loaded from scikit-learn and require no network access during execution.

## Versioned assets

The `assets/` directory contains the current manuscript figures and their underlying result tables:

- `lac_multiseed_validation.pdf`
- `annealing_bifurcation.pdf`
- `sne_density_scaling.pdf`
- `sne_neighbor_response.pdf`
- `lac_multiseed_results.csv`
- `lac_convergence_results.csv`
- `lac_real_benchmarks.csv`
- `numerical_summary.csv`
