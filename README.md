# DELA-SNE

Research repository for **Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)** and its two algorithmic foundations: **Locally Adaptive Clustering (LAC)** and **t-distributed Stochastic Neighbor Embedding (t-SNE)**.

The repository separates clustering, embedding, data, reproducible execution, manuscript material, and references. LAC and t-SNE are implemented as independent reference modules. DELA-SNE remains a research implementation target and is not yet frozen as software.

## Algorithms

| Method | Role | Implementation | Output |
|---|---|---|---|
| LAC | clustering with cluster-dependent feature relevance | [`lac/lac.py`](lac/lac.py) | hard cluster, assigned weighted distance, local feature weights |
| t-SNE | nonlinear probabilistic embedding | [`tsne/tsne.py`](tsne/tsne.py) | two-dimensional embedding |
| DELA-SNE | proposed method | pending | pending |

The LAC implementation follows the entropy-regularized feature-weight update of Domeniconi et al. The t-SNE implementation is an explicit exact NumPy implementation intended for small reproducible datasets rather than large-scale workloads.

## Repository structure

```text
dela_sne/
├── README.md
├── run_workflow.py
├── requirements.txt
├── .github/
│   └── workflows/
│       └── algorithms.yml
├── data/
│   ├── test/
│   │   └── df_baseline.csv
│   └── result/
│       └── df_baseline_result_YYYYMMDD.csv
├── lac/
│   ├── __init__.py
│   ├── lac.py
│   ├── README.md
│   └── docs/
├── tsne/
│   ├── __init__.py
│   ├── tsne.py
│   ├── README.md
│   └── docs/
├── dela_sne/
│   ├── README.md
│   └── docs/
├── paper/
├── references/
│   └── references.bib
├── docs/
└── tests/
```

## Data convention

`data/test/` contains input datasets and `data/result/` contains generated algorithm outputs. Test CSV files must contain numeric feature columns named `feature_1`, `feature_2`, and so on. Other columns are preserved as metadata and are not used as model inputs.

The committed baseline dataset, [`data/test/df_baseline.csv`](data/test/df_baseline.csv), is a deterministic synthetic benchmark generated with random seed 42. It contains 96 observations, eight numerical features, and three known groups. Informative dimensions differ by group, while other dimensions contain larger nuisance variation. The `true_cluster` column is retained only for validation and is excluded from algorithm inputs.

## Joint workflow

Run both implemented algorithms over every CSV in `data/test/`:

```bash
python run_workflow.py
```

Run a specific file:

```bash
python run_workflow.py data/test/df_baseline.csv
```

Relevant parameters can be controlled from the command line:

```bash
python run_workflow.py --clusters 3 --h 0.5 --perplexity 30 --seed 42
```

For an input named

```text
data/test/df_baseline.csv
```

an execution on 17 August 2026 writes

```text
data/result/df_baseline_result_20260817.csv
```

The result preserves the input columns and adds the row-level outputs `lac_cluster`, `lac_distance`, `tsne_1`, and `tsne_2`. The cluster-level LAC feature weights remain available through `LAC.feature_weights_` and are not duplicated on every CSV row.

All model features are standardized once inside the shared workflow before either algorithm is executed, so the two methods receive the same numerical input representation.

## GitHub Actions

[`.github/workflows/algorithms.yml`](.github/workflows/algorithms.yml) provides the joint CI workflow. It installs the project dependency, runs unit/smoke tests, executes LAC and t-SNE on the committed test datasets, and uploads the generated `data/result/*.csv` files as a workflow artifact.

## Reproducibility

The reference workflow uses fixed seeds by default. Exact t-SNE is quadratic in the number of observations and is deliberately used here because the baseline is small and the implementation is intended to remain mathematically inspectable. Large datasets should use an accelerated implementation rather than this reference code.

## Associated manuscript

The repository accompanies the manuscript on the statistical-mechanical interpretation of LAC and SNE/t-SNE and the subsequent DELA-SNE construction. Manuscript-related material is maintained under [`paper/`](paper/).

The canonical machine-readable bibliography is [`references/references.bib`](references/references.bib). It is synchronized with the current manuscript reference list.

## Documentation

Algorithm-specific documentation is colocated with each algorithm:

- [`lac/docs/`](lac/docs/)
- [`tsne/docs/`](tsne/docs/)
- [`dela_sne/docs/`](dela_sne/docs/)

Project-wide documentation remains under [`docs/`](docs/).
