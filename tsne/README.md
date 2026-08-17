# t-distributed Stochastic Neighbor Embedding (t-SNE)

This directory contains the t-SNE component of the DELA-SNE project: documentation, the planned reproducible example, and its generated artifacts.

## Structure

```text
tsne/
├── README.md
├── docs/                  # Mathematical and methodological documentation
│   └── README.md
├── example.ipynb          # Planned executable example
├── figures/               # Generated figures
└── results/               # Optional machine-readable outputs
```

## Practical objective

Demonstrate stochastic-neighbor embedding on a controlled high-dimensional dataset while making explicit that the output is an embedding rather than a clustering assignment.

The example should record preprocessing, perplexity, learning rate, initialization, number of iterations, random seed, and the criteria used to interpret neighborhood preservation.

See [`docs/`](docs/) for the methodological description and canonical references.
