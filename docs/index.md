# DELA-SNE

**Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding** is a research project on locally adaptive metrics, stochastic-neighbor embeddings, and information-theoretic formulations for high-dimensional data analysis.

This project-wide documentation accompanies the research repository and the manuscript in preparation. Algorithm-specific documentation is colocated with each top-level module rather than duplicated here.

## Project map

- [LAC module](https://github.com/ozsp12/dela_sne/tree/main/lac): locally adaptive clustering and cluster-dependent feature relevance.
- [t-SNE module](https://github.com/ozsp12/dela_sne/tree/main/tsne): probabilistic neighborhood-preserving embedding.
- [DELA-SNE module](https://github.com/ozsp12/dela_sne/tree/main/dela_sne): proposed locally adaptive stochastic-neighbor embedding.
- [Practical examples](examples.md): common reproducibility and comparison strategy.
- [Associated manuscript](paper.md): project scope and manuscript information.
- [References](references.md): canonical literature used by the project.

## Methodological structure

The project starts from two established methods with different purposes. Locally Adaptive Clustering (LAC) learns feature relevance locally for clustering high-dimensional observations. t-distributed Stochastic Neighbor Embedding (t-SNE) constructs a low-dimensional embedding by matching neighborhood distributions. DELA-SNE investigates a principled combination of locally adaptive metrics and stochastic-neighbor embedding under an explicit entropy-based formulation.

The repository therefore treats `lac/`, `tsne/`, and `dela_sne/` as independent top-level modules. Each module contains its own documentation and will contain its own practical example and generated artifacts.
