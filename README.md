# DELA-SNE

Research repository for the development of **Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)** and for reproducible comparisons with its two main methodological foundations: **Locally Adaptive Clustering (LAC)** and **t-distributed Stochastic Neighbor Embedding (t-SNE)**.

## Scope

The project studies how locally adaptive feature metrics can be incorporated into stochastic-neighbor embedding while retaining an explicit information-theoretic interpretation.

A central distinction is maintained throughout the project:

- **LAC** is a clustering algorithm that learns cluster-dependent feature weights.
- **t-SNE** is a nonlinear embedding and visualization algorithm, not a clustering algorithm.
- **DELA-SNE** is the proposed method investigated in the associated manuscript.

## Algorithms

| Method | Role | Main idea | Module |
|---|---|---|---|
| LAC | Methodological foundation | Locally adaptive metrics and feature weighting for high-dimensional clustering | [`lac/`](lac/) |
| t-SNE | Methodological foundation | Probabilistic neighborhood-preserving low-dimensional embedding | [`tsne/`](tsne/) |
| DELA-SNE | Proposed method | Locally adaptive stochastic-neighbor embedding with an entropy-based formulation | [`dela_sne/`](dela_sne/) |

Each algorithm is organized as an independent top-level module containing its own README, methodological documentation, practical example, figures, and results.

## Associated manuscript

**Working title:** *Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding*

**Status:** manuscript in preparation.

The manuscript develops the connection among locally adaptive metrics, stochastic neighbor embedding, generalized distance functions, and information-theoretic quantities. Repository implementations and numerical examples are intended to reproduce the computational results presented in the paper.

See [`paper/`](paper/) and the project-wide [paper documentation](docs/paper.md).

## Repository structure

```text
dela_sne/
├── README.md
├── lac/                   # LAC module
│   ├── README.md
│   └── docs/
├── tsne/                  # t-SNE module
│   ├── README.md
│   └── docs/
├── dela_sne/              # Proposed DELA-SNE module
│   ├── README.md
│   └── docs/
├── data/                  # Shared or reproducible datasets
├── docs/                  # Project-wide documentation and GitHub Pages source
│   ├── index.md
│   ├── examples.md
│   ├── paper.md
│   └── references.md
├── paper/                 # Manuscript-related material
├── references/            # Bibliography and reference material
│   └── references.bib
├── src/dela_sne/          # Future reusable DELA-SNE package implementation
└── tests/                 # Tests for reusable implementation
```

## Practical examples

Each top-level algorithm module will contain one practical example. The three examples should follow a common reproducibility standard:

1. define or load a compact dataset;
2. state preprocessing assumptions;
3. execute the method with explicit parameters;
4. visualize the result;
5. interpret the output according to the actual role of the method;
6. record random seeds and software versions.

Whenever possible, the same benchmark dataset or deliberately comparable synthetic datasets should be used across the three modules so that methodological differences are visible rather than confounded by dataset choice.

## References

Core references currently used by the project include:

1. P. C. Mahalanobis, “On the generalized distance in statistics,” *Proceedings of the National Institute of Sciences of India* **2**, 49–55 (1936).
2. C. E. Shannon, “A Mathematical Theory of Communication,” *Bell System Technical Journal* **27**, 379–423 and 623–656 (1948).
3. E. T. Jaynes, “Information Theory and Statistical Mechanics,” *Physical Review* **106**, 620–630 (1957). DOI: 10.1103/PhysRev.106.620.
4. E. T. Jaynes, “Information Theory and Statistical Mechanics. II,” *Physical Review* **108**, 171–190 (1957). DOI: 10.1103/PhysRev.108.171.
5. G. E. Hinton and S. T. Roweis, “Stochastic Neighbor Embedding,” *Advances in Neural Information Processing Systems* **15** (2002).
6. C. Domeniconi, D. Gunopulos, S. Ma, B. Yan, M. Al-Razgan, and D. Papadopoulos, “Locally adaptive metrics for clustering high dimensional data,” *Data Mining and Knowledge Discovery* **14**, 63–97 (2007). DOI: 10.1007/s10618-006-0060-8.
7. L. van der Maaten and G. Hinton, “Visualizing Data using t-SNE,” *Journal of Machine Learning Research* **9**, 2579–2605 (2008).
8. A. Rényi, “On Measures of Entropy and Information,” in *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, Vol. 1, 547–561 (1961).
9. C. Tsallis, “Possible generalization of Boltzmann-Gibbs statistics,” *Journal of Statistical Physics* **52**, 479–487 (1988). DOI: 10.1007/BF01016429.

The machine-readable bibliography is maintained in [`references/references.bib`](references/references.bib).

## Documentation

Algorithm-specific documentation is colocated with each module:

- [`lac/docs/`](lac/docs/)
- [`tsne/docs/`](tsne/docs/)
- [`dela_sne/docs/`](dela_sne/docs/)

The root [`docs/`](docs/) directory is reserved for project-wide material such as the landing page, manuscript overview, shared examples, and bibliography.

## Development status

The repository is currently in the research and prototyping phase. The mathematical definitions in the manuscript remain the source of truth for DELA-SNE until the algorithmic specification is frozen.
