# Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)

This directory contains the proposed DELA-SNE method: documentation, the future implementation, the planned reproducible example, and its validation artifacts.

## Structure

```text
dela_sne/
├── README.md
├── docs/                  # Mathematical and methodological documentation
│   └── README.md
├── example.ipynb          # Planned executable example
├── figures/               # Generated figures
└── results/               # Optional machine-readable outputs
```

## Practical objective

Demonstrate the proposed locally adaptive stochastic-neighbor embedding after the mathematical specification is frozen in the manuscript. The example should reuse the benchmark data and preprocessing adopted for the LAC and t-SNE modules whenever possible.

The implementation and notebook must expose the locally adaptive metric quantities explicitly and keep embedding evaluation separate from clustering evaluation.

See [`docs/`](docs/) for the current methodological description and [`../paper/`](../paper/) for manuscript-related material.
