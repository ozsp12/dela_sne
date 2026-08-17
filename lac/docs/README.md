# Locally Adaptive Clustering (LAC)

Locally Adaptive Clustering (LAC) is the clustering foundation used in the DELA-SNE project for locally adaptive feature relevance in high-dimensional data.

## Methodological role

LAC associates each cluster with its own feature-weight vector rather than imposing one global metric on the entire feature space. The resulting local geometry allows different dimensions to carry different relevance for different clusters.

Within DELA-SNE, LAC is used specifically as the source of the locally adaptive metric mechanism. It is not treated as an embedding method.

## Project use

The LAC module is intended to contain:

- the practical and reproducible LAC example;
- mathematical and methodological notes under `docs/`;
- generated figures and machine-readable results when the example is implemented.

The practical example should expose the learned feature weights explicitly and distinguish cluster assignment from subsequent visualization.

## Canonical reference

C. Domeniconi, D. Gunopulos, S. Ma, B. Yan, M. Al-Razgan, and D. Papadopoulos, “Locally adaptive metrics for clustering high dimensional data,” *Data Mining and Knowledge Discovery* **14**, 63–97 (2007). DOI: 10.1007/s10618-006-0060-8.

Project-wide references are maintained in [`../../references/`](../../references/).
