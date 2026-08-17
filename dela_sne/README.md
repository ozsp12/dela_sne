# Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)

This module is reserved for the proposed DELA-SNE method.

## Current status

The software implementation is intentionally **not yet added**. The repository currently implements only the two source algorithms used as baselines:

- [`../lac/`](../lac/) — Locally Adaptive Clustering;
- [`../tsne/`](../tsne/) — t-distributed Stochastic Neighbor Embedding.

DELA-SNE will be implemented after its mathematical specification and evaluation protocol are frozen. This avoids allowing an experimental code path to become the de facto definition of the method.

## Planned integration

The future implementation will use the same repository conventions:

```text
dela_sne/
├── README.md
├── docs/
└── dela_sne.py            # future
```

It will consume datasets from `../data/test/`, participate in the shared workflow, and append its outputs to the corresponding dated file in `../data/result/`.

See [`docs/`](docs/) for the current methodological description and [`../paper/`](../paper/) for manuscript material.
