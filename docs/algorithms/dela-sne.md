# DELA-SNE

**Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)** is the proposed method developed in the associated manuscript.

## Research objective

The method investigates how locally adaptive feature metrics can modify the high-dimensional neighborhood geometry used by stochastic-neighbor embedding while retaining a mathematically explicit information-theoretic formulation. The project also studies the relation between the metric construction, Shannon-type quantities, and generalized entropy families considered in the manuscript.

## Current status

The formal mathematical specification is under active development. For that reason, this repository treats the manuscript definition as the source of truth and does not yet present a frozen production implementation.

## Planned validation

The DELA-SNE example should be compared against the LAC and t-SNE examples using controlled data, common preprocessing, fixed random seeds, and explicit evaluation criteria. Comparisons should distinguish clustering quality from embedding quality.

See `examples/dela_sne/` and the [paper page](../paper.md).
