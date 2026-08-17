# Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)

Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE) is the proposed method developed in the associated manuscript.

## Research objective

The method investigates how locally adaptive feature metrics can modify the high-dimensional neighborhood geometry used by stochastic-neighbor embedding while retaining an explicit information-theoretic formulation. The project studies the relation among local metric adaptation, neighborhood probabilities, Shannon-type quantities, and generalized entropy families considered in the manuscript.

## Current status

The formal specification is under active development. The manuscript remains the source of truth until the mathematical definition and optimization procedure are frozen.

## Project use

The DELA-SNE module is intended to contain:

- the implementation of the proposed method;
- the practical and reproducible DELA-SNE example;
- mathematical and methodological notes under `docs/`;
- generated figures and machine-readable results used for validation and manuscript reproduction.

The DELA-SNE example should be compared with the LAC and t-SNE modules using controlled data, common preprocessing, fixed random seeds, and explicit evaluation criteria. Embedding quality and clustering quality must be evaluated separately.

See also [`../../paper/`](../../paper/) and the project-wide [`../../references/`](../../references/).
