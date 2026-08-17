# DELA-SNE

**Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding** is a research project on locally adaptive metrics, stochastic-neighbor embeddings, and information-theoretic formulations for high-dimensional data analysis.

This site accompanies the research repository and the manuscript in preparation.

## Project map

- [Algorithms](algorithms/): LAC, t-SNE, and DELA-SNE.
- [Practical examples](examples.md): reproducible demonstrations planned for the three algorithms.
- [Associated manuscript](paper.md): scope and current manuscript structure.
- [References](references.md): canonical literature used by the project.

## Methodological structure

The project starts from two established methods with different purposes. Locally Adaptive Clustering (LAC) learns feature relevance locally for clustering high-dimensional observations. t-distributed Stochastic Neighbor Embedding (t-SNE) constructs a low-dimensional embedding by matching neighborhood distributions. DELA-SNE investigates a principled combination of locally adaptive metrics and stochastic-neighbor embedding under an explicit entropy-based formulation.

The repository separates these components so that the proposed method can be compared against each methodological foundation independently.
