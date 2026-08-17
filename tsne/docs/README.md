# t-distributed Stochastic Neighbor Embedding (t-SNE)

t-distributed Stochastic Neighbor Embedding (t-SNE) is the probabilistic embedding foundation used in the DELA-SNE project.

## Methodological role

t-SNE constructs a low-dimensional representation by matching neighborhood probability distributions between the original high-dimensional space and the embedding space. Its heavy-tailed low-dimensional distribution alleviates the crowding problem of the original Stochastic Neighbor Embedding formulation.

Within DELA-SNE, SNE/t-SNE supplies the neighborhood-based probabilistic embedding framework. t-SNE is an embedding algorithm, not a clustering algorithm.

## Project use

The t-SNE module is intended to contain:

- the practical and reproducible t-SNE example;
- mathematical and methodological notes under `docs/`;
- generated figures and machine-readable results when the example is implemented.

The example should record preprocessing, perplexity, learning rate, initialization, number of iterations, random seed, and any evaluation metrics used.

## Canonical references

G. E. Hinton and S. T. Roweis, “Stochastic Neighbor Embedding,” *Advances in Neural Information Processing Systems* **15** (2002).

L. van der Maaten and G. Hinton, “Visualizing Data using t-SNE,” *Journal of Machine Learning Research* **9**, 2579–2605 (2008).

Project-wide references are maintained in [`../../references/`](../../references/).
