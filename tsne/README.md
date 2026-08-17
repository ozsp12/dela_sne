# t-distributed Stochastic Neighbor Embedding (t-SNE)

This module contains an explicit **exact t-SNE reference implementation in NumPy**. It is designed for small reproducible datasets and mathematical inspection rather than large-scale production use.

## Files

```text
tsne/
├── __init__.py
├── tsne.py
├── README.md
└── docs/
```

The implementation performs:

1. pairwise squared Euclidean distances in the input space;
2. pointwise binary search for beta_i = (2 sigma_i^2)^(-1) at fixed perplexity;
3. symmetrization of the high-dimensional probabilities;
4. Student-t affinities in the low-dimensional space;
5. gradient descent on the Kullback–Leibler objective with early exaggeration and momentum.

## Software interface

```python
from tsne import TSNE

model = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=100,
    n_iter=500,
    random_state=42,
)

embedding = model.fit_transform(X)
```

The fitted object exposes `embedding_`, `P_`, `P_conditional_`, `sigmas_`, `kl_history_`, and `kl_divergence_`.

The implementation is O(n^2) in storage and computation for the pairwise probability matrices. It is therefore deliberately a reference implementation.

## Shared workflow

t-SNE is normally executed together with LAC:

```bash
python run_workflow.py data/test/df_baseline.csv
```

The two embedding coordinates are appended to the common result CSV as `tsne_1` and `tsne_2`. The result filename follows `<test_stem>_result_YYYYMMDD.csv`.

## Canonical references

G. E. Hinton and S. T. Roweis, “Stochastic Neighbor Embedding,” *Advances in Neural Information Processing Systems* 15 (2002).

L. van der Maaten and G. Hinton, “Visualizing Data using t-SNE,” *Journal of Machine Learning Research* 9, 2579–2605 (2008).

See [`docs/`](docs/) and [`../references/references.bib`](../references/references.bib).
