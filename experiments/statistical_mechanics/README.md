# Statistical-mechanics experiments

This directory contains the reproducibility code for the numerical experiments supporting the manuscript **Feature and Neighbor Ensembles in Locally Adaptive Clustering and Stochastic Neighbor Embedding: Response Functions and Perplexity Tolerance**.

## Contents

- `run_experiments.py`: complete experiment driver.
- `requirements.txt`: Python dependencies required by the experiment driver.

## Experiments reproduced

The driver reproduces:

- the 30-realization synthetic Locally Adaptive Clustering (LAC) validation;
- convergence and explicit cycle diagnostics for the adaptive response rule;
- the two-pass response rule with frozen temperature;
- the common-grid supervised oracle comparison;
- plateau-width diagnostics around the optimal adjusted Rand index;
- the clusterwise response-temperature experiment;
- standardized-feature sensitivity checks;
- Iris, Wine, and Breast Cancer Wisconsin Diagnostic clustering benchmarks;
- the deterministic-annealing bifurcation control;
- the Stochastic Neighbor Embedding (SNE) density-scaling control;
- local-neighbor response and perplexity-tolerance diagnostics.

## Reproduce

```bash
python -m pip install -r experiments/statistical_mechanics/requirements.txt
python experiments/statistical_mechanics/run_experiments.py
```

The script uses deterministic seeds 1729--1758 for the 30-realization experiments. The real datasets are loaded from scikit-learn and require no network request during execution.

The response selector and supervised oracle use the same 160-point logarithmic grid on `0.01 <= h <= 8.0`. The adaptive LAC implementation records every visited partition and terminates explicitly as `converged`, `cycle`, or `max_iter`. The two-pass rule selects one global response temperature from the initial k-means partition and then freezes it for the subsequent LAC iterations.

Generated figures and CSV files are written to `experiments/statistical_mechanics/outputs/`.
