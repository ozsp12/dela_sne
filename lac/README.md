# Locally Adaptive Clustering (LAC)

This directory contains the LAC component of the DELA-SNE project: documentation, the planned reproducible example, and its generated artifacts.

## Structure

```text
lac/
├── README.md
├── docs/                  # Mathematical and methodological documentation
│   └── README.md
├── example.ipynb          # Planned executable example
├── figures/               # Generated figures
└── results/               # Optional machine-readable outputs
```

## Practical objective

Demonstrate how Locally Adaptive Clustering assigns cluster-dependent relevance to features in high-dimensional data. A controlled synthetic dataset with informative and nuisance dimensions is preferred so that the learned local weights can be inspected directly.

The example should report cluster assignments, learned feature weights, convergence behavior, parameter values, random seed, and a compact visualization.

See [`docs/`](docs/) for the methodological description and canonical reference.
