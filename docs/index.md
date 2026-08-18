# DELA-SNE

**Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)** is the project-level research program for locally adaptive metrics, stochastic-neighbor embeddings, and information-theoretic formulations for high-dimensional data analysis.

The repository currently provides tested reference implementations of Locally Adaptive Clustering (LAC) and exact t-SNE. DELA-SNE itself remains under mathematical specification and is intentionally not exposed as a stable algorithm until the freeze checklist is complete.

## Project map

- [Canonical Python package](https://github.com/ozsp12/dela_sne/tree/main/src/dela_sne): executable LAC, t-SNE, diagnostics, and shared workflow.
- [LAC documentation](https://github.com/ozsp12/dela_sne/tree/main/lac): locally adaptive clustering and cluster-dependent feature relevance.
- [t-SNE documentation](https://github.com/ozsp12/dela_sne/tree/main/tsne): probabilistic neighborhood-preserving embedding.
- [DELA-SNE specification checklist](dela_sne.md): mathematical and numerical requirements that must be frozen before implementation.
- [Practical examples](examples.md): common reproducibility and comparison strategy.
- [Current manuscript](paper.md): scope and relation between the present article and the wider DELA-SNE project.
- [References](references.md): literature and bibliography policy.

## Methodological structure

LAC and t-SNE solve different problems. LAC is a clustering method that learns cluster-dependent feature relevance. t-SNE is an embedding method that seeks low-dimensional coordinates reproducing neighborhood probabilities. The DELA-SNE project investigates how locally adaptive metrics can be coupled consistently to stochastic-neighbor embeddings under an explicit information-theoretic formulation.

Executable code has one canonical location, `src/dela_sne/`. The top-level `lac/` and `tsne/` directories are documentation layers, not competing Python packages. The current manuscript in `paper/` studies response functions, feature and neighbor ensembles, and perplexity tolerance as foundational results; it does not claim to define the final DELA-SNE algorithm.
