# DELA-SNE implementation status

Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE) is the proposed method developed in the associated manuscript. The repository deliberately does not expose a production DELA-SNE class yet; `src/dela_sne/` currently contains only the reference LAC and t-SNE implementations used to validate the mathematical ingredients.

## Freeze criteria

The DELA-SNE implementation is considered frozen only when all of the following are satisfied:

- [ ] the high-dimensional locally adaptive metric is stated unambiguously in the manuscript;
- [ ] the location and update rule of the local weights are fixed;
- [ ] asymmetric local distances have an explicit symmetrization rule;
- [ ] the relation between feature entropy and neighbor entropy is fixed;
- [ ] the optimization objective and gradient are derived and tested by finite differences;
- [ ] initialization, stopping, and numerical-stability rules are specified;
- [ ] baseline comparisons against the canonical LAC and t-SNE package pass on synthetic and real datasets;
- [ ] random seeds and hyperparameter search spaces used by the manuscript are versioned;
- [ ] the manuscript equations referenced by the implementation have stable labels;
- [ ] a versioned release is archived before the manuscript cites the software.

Until this checklist is complete, changes to DELA-SNE belong to the manuscript and experiment layer rather than the stable package API.
