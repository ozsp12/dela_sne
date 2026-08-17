# Locally Adaptive Clustering (LAC)

This module contains the project reference implementation of **Locally Adaptive Clustering (LAC)**.

## Files

```text
lac/
├── __init__.py
├── lac.py
├── README.md
└── docs/
```

`lac.py` implements alternating hard assignment, centroid updates, cluster-dependent feature dispersions, and entropy-regularized feature weights.

For cluster C_k, feature f, dispersion V_kf, and smoothing parameter h > 0, the implemented feature update is

```text
w_kf = exp(-V_kf/h) / sum_g exp(-V_kg/h).
```

Assignments use the cluster-dependent weighted squared distance

```text
d_k^2(x,z_k) = sum_f w_kf (x_f-z_kf)^2.
```

## Software interface

```python
from lac import LAC

model = LAC(
    n_clusters=3,
    h=0.5,
    max_iter=200,
    random_state=42,
)

labels = model.fit_predict(X)
weights = model.feature_weights_
centroids = model.cluster_centers_
```

The fitted object exposes `labels_`, `cluster_centers_`, `feature_weights_`, `distances_`, `distance_to_assigned_`, `n_iter_`, and `objective_`.

## Shared workflow

LAC is normally executed through the repository-level workflow:

```bash
python run_workflow.py data/test/df_baseline.csv
```

Its row-level outputs are written into the common result CSV as `lac_cluster`, `lac_distance`, and `lac_weight_feature_*` columns. The result filename follows `<test_stem>_result_YYYYMMDD.csv`.

## Reference

C. Domeniconi, D. Gunopulos, S. Ma, B. Yan, M. Al-Razgan, and D. Papadopoulos, “Locally Adaptive Metrics for Clustering High Dimensional Data,” *Data Mining and Knowledge Discovery* 14, 63–97 (2007), DOI `10.1007/s10618-006-0060-8`.

See [`docs/`](docs/) for methodological documentation and [`../references/references.bib`](../references/references.bib) for the project bibliography.
