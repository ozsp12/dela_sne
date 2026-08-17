# Locally Adaptive Clustering (LAC)

Locally Adaptive Clustering is used in this project as the canonical reference for cluster-dependent feature relevance in high-dimensional data. Instead of imposing a single global weighting of the input dimensions, LAC associates clusters with locally adapted feature weights and therefore with cluster-specific distance behavior.

## Role in DELA-SNE

The relevant idea is not to reinterpret LAC as an embedding algorithm. Its role is to supply the locally adaptive metric mechanism that motivates feature-dependent neighborhood geometry in the proposed method.

## Canonical reference

C. Domeniconi, D. Gunopulos, S. Ma, B. Yan, M. Al-Razgan, and D. Papadopoulos, “Locally adaptive metrics for clustering high dimensional data,” *Data Mining and Knowledge Discovery* **14**, 63–97 (2007). DOI: 10.1007/s10618-006-0060-8.

See the planned practical example in `examples/lac/`.
