# DELA-SNE

Research repository for the development of **Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)** and for reproducible comparisons with the two main methodological foundations used in the project: **Locally Adaptive Clustering (LAC)** and **t-distributed Stochastic Neighbor Embedding (t-SNE)**.

## Scope

The project studies how locally adaptive feature metrics can be incorporated into stochastic-neighbor embedding while retaining a clear information-theoretic interpretation. The repository is organized to support the mathematical development, computational implementation, practical examples, experiments, references, and the associated manuscript.

A central distinction is maintained throughout the project:

- **LAC** is a clustering algorithm that learns cluster-dependent feature weights.
- **t-SNE** is a nonlinear embedding and visualization algorithm, not a clustering algorithm.
- **DELA-SNE** is the proposed method investigated in the associated manuscript.

## Algorithms

| Method | Role in the project | Main idea | Practical example |
|---|---|---|---|
| LAC | Methodological foundation | Locally adaptive metrics and feature weighting for high-dimensional clustering | `examples/lac/` |
| t-SNE | Methodological foundation | Probabilistic neighborhood-preserving low-dimensional embedding | `examples/tsne/` |
| DELA-SNE | Proposed method | Locally adaptive stochastic-neighbor embedding with an entropy-based formulation | `examples/dela_sne/` |

## Associated manuscript

**Working title:** *Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding*

**Status:** manuscript in preparation.

The manuscript develops the connection among locally adaptive metrics, stochastic neighbor embedding, generalized distance functions, and information-theoretic quantities. The repository will contain the implementations and numerical examples required to reproduce the computational results presented in the paper.

See [`paper/README.md`](paper/README.md) and the [paper page](docs/paper.md) for the current project-level description.

## Repository structure

```text
dela_sne/
├── README.md
├── data/                  # Small or reproducible datasets used in examples
├── docs/                  # GitHub Pages source
│   ├── algorithms/        # Method pages for LAC, t-SNE, and DELA-SNE
│   ├── examples.md
│   ├── index.md
│   ├── paper.md
│   └── references.md
├── examples/              # One practical example for each algorithm
│   ├── lac/
│   ├── tsne/
│   └── dela_sne/
├── paper/                 # Manuscript-related material
├── references/            # Bibliography and reference material
│   └── references.bib
├── src/dela_sne/          # Future DELA-SNE implementation
└── tests/                 # Tests for the implementation
```

## Practical examples

Each method has a dedicated example directory. The intended common structure is:

1. define or load a compact dataset;
2. state the preprocessing assumptions;
3. execute the algorithm with explicit parameters;
4. visualize the result;
5. interpret the output without conflating clustering and embedding;
6. record the random seed and software versions required for reproduction.

The three examples should eventually use either the same benchmark dataset or deliberately comparable synthetic datasets so that the methodological differences are visible rather than hidden by dataset choice.

## References

The canonical references currently used by the project include:

1. P. C. Mahalanobis, “On the generalized distance in statistics,” *Proceedings of the National Institute of Sciences of India* **2**, 49–55 (1936).
2. C. E. Shannon, “A Mathematical Theory of Communication,” *Bell System Technical Journal* **27**, 379–423 and 623–656 (1948).
3. E. T. Jaynes, “Information Theory and Statistical Mechanics,” *Physical Review* **106**, 620–630 (1957). DOI: 10.1103/PhysRev.106.620.
4. E. T. Jaynes, “Information Theory and Statistical Mechanics. II,” *Physical Review* **108**, 171–190 (1957). DOI: 10.1103/PhysRev.108.171.
5. G. E. Hinton and S. T. Roweis, “Stochastic Neighbor Embedding,” *Advances in Neural Information Processing Systems* **15** (2002).
6. C. Domeniconi, D. Gunopulos, S. Ma, B. Yan, M. Al-Razgan, and D. Papadopoulos, “Locally adaptive metrics for clustering high dimensional data,” *Data Mining and Knowledge Discovery* **14**, 63–97 (2007). DOI: 10.1007/s10618-006-0060-8.
7. L. van der Maaten and G. Hinton, “Visualizing Data using t-SNE,” *Journal of Machine Learning Research* **9**, 2579–2605 (2008).
8. A. Rényi, “On Measures of Entropy and Information,” in *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, Vol. 1, 547–561 (1961).
9. C. Tsallis, “Possible generalization of Boltzmann-Gibbs statistics,” *Journal of Statistical Physics* **52**, 479–487 (1988). DOI: 10.1007/BF01016429.

The machine-readable bibliography is maintained in [`references/references.bib`](references/references.bib). Reference PDFs or author manuscripts may be added under `references/` when redistribution rights permit it.

## Website

The project website source is located in [`docs/`](docs/). It is designed for GitHub Pages and separates the scientific overview from the repository-level documentation.

After merging the site structure into `main`, GitHub Pages can be configured to publish from the `/docs` directory.

## Development status

The repository is currently in the research and prototyping phase. Mathematical definitions in the manuscript should remain the source of truth for the DELA-SNE implementation until the algorithmic specification is frozen.
