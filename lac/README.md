# Locally Adaptive Clustering (LAC)

This directory contains method-specific documentation for Locally Adaptive Clustering. The executable implementation is canonicalized at [`../src/dela_sne/lac.py`](../src/dela_sne/lac.py); there is no second LAC implementation in this directory.

The implemented local weighted squared distance is

```text
d_k^2(x,z_k) = sum_f w_kf (x_f-z_kf)^2,
```

with entropy-regularized feature weights

```text
w_kf = exp(-V_kf/h) / sum_g exp(-V_kg/h).
```

Here `V_kf` is a cluster-dependent feature dispersion. The temperature `h` has units of variance, so comparisons across values of `h` require a defined feature scale.

## Software interface

```python
from dela_sne import LAC

model = LAC(
    n_clusters=3,
    h=0.2,
    n_init=10,
    max_iter=200,
    random_state=42,
).fit(X)
```

The fitted model exposes the distance and entropy contributions separately:

```python
model.distance_term_
model.entropy_term_
model.objective_
```

and convergence diagnostics through `status_`, `cycle_period_`, `n_iter_`, and `restart_objectives_`.

Multiple restarts are selected by the lowest complete entropy-regularized objective. Empty-cluster repair uses distinct farthest observations within an iteration. An explicit `temperature_rule` and `initial_labels` can be supplied when reproducing manuscript experiments; these choices alter the experimental protocol without creating another LAC implementation.

## Shared workflow

```bash
dela-sne-run data/test/df_baseline.csv
```

The stable output is `data/result/df_baseline_result.csv`. The result CSV contains observation-level labels and distances; the cluster-level feature weights remain in `model.feature_weights_`.

## Reference

C. Domeniconi, D. Gunopulos, S. Ma, B. Yan, M. Al-Razgan, and D. Papadopoulos, “Locally Adaptive Metrics for Clustering High Dimensional Data,” *Data Mining and Knowledge Discovery* 14, 63–97 (2007), DOI `10.1007/s10618-006-0060-8`.

See [`docs/`](docs/) and the canonical bibliography [`../references/references.bib`](../references/references.bib).
