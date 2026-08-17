# Practical examples

The repository reserves one reproducible example for each of the three central algorithms.

## 1. LAC

Target directory: `examples/lac/`.

The example should illustrate cluster-specific feature relevance. A synthetic dataset with informative and nuisance dimensions is preferable because the learned local weights can be inspected directly.

## 2. t-SNE

Target directory: `examples/tsne/`.

The example should illustrate probabilistic neighborhood preservation and the distinction between an embedding and a clustering assignment. Hyperparameters, random seed, input scaling, and initialization must be recorded.

## 3. DELA-SNE

Target directory: `examples/dela_sne/`.

The example should reuse the same data whenever possible and expose the locally adaptive metric quantities introduced by the method. Any comparison with t-SNE must keep preprocessing and evaluation conditions controlled.

## Reproducibility standard

Each example should contain the dataset provenance, environment information, random seed, parameter values, generated figures, and a short interpretation of the result. Generated outputs should be reproducible from code rather than stored as unexplained figures.
