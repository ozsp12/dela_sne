# t-distributed Stochastic Neighbor Embedding (t-SNE)

This directory contains method-specific documentation for t-SNE. The executable exact NumPy implementation is canonicalized at [`../src/dela_sne/tsne.py`](../src/dela_sne/tsne.py).

The implementation performs the high-dimensional perplexity search, symmetric joint probabilities, Student-t low-dimensional affinities, and gradient descent on the Kullback-Leibler objective.

## Software interface

```python
from dela_sne import TSNE

model = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    n_iter=1000,
    random_state=42,
)
embedding = model.fit_transform(X)
```

The fitted model exposes `P_`, `P_conditional_`, `sigmas_`, `kl_history_`, `kl_divergence_`, `learning_rate_`, `n_iter_`, and `status_`.

## Optimizer details

The reference optimizer now includes:

- PCA or random initialization;
- adaptive per-coordinate gains;
- `learning_rate="auto"`, using `max(n / (4 * early_exaggeration), 50)`;
- early exaggeration and momentum;
- stopping by gradient norm after the exaggeration phase.

Strict step-by-step monotonic decrease of KL is not asserted because momentum does not guarantee it. The test suite instead verifies post-exaggeration improvement and checks the analytic KL gradient against finite differences.

The implementation remains O(n^2) in time/storage for pairwise probability matrices. It is intended as a mathematically inspectable reference, not a high-throughput replacement for Barnes-Hut or FFT implementations.

## Shared workflow

```bash
python run_workflow.py data/test/df_baseline.csv
```

The coordinates are appended to the stable output `data/result/df_baseline_result.csv` as `tsne_1` and `tsne_2`.

## Canonical references

G. E. Hinton and S. T. Roweis, “Stochastic Neighbor Embedding,” *Advances in Neural Information Processing Systems* 15 (2002).

L. van der Maaten and G. Hinton, “Visualizing Data using t-SNE,” *Journal of Machine Learning Research* 9, 2579–2605 (2008).

See [`docs/`](docs/) and [`../references/references.bib`](../references/references.bib).
