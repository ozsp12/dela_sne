# t-distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a nonlinear dimensionality-reduction method designed to construct low-dimensional embeddings that preserve neighborhood structure probabilistically. It extends the Stochastic Neighbor Embedding framework by using a heavy-tailed distribution in the embedding space, reducing the crowding behavior of the original formulation.

## Role in DELA-SNE

The project uses SNE/t-SNE as the canonical probabilistic embedding framework. The distinction from clustering is maintained explicitly: t-SNE returns an embedding, not a partition of the dataset.

## Canonical references

G. E. Hinton and S. T. Roweis, “Stochastic Neighbor Embedding,” *Advances in Neural Information Processing Systems* **15** (2002).

L. van der Maaten and G. Hinton, “Visualizing Data using t-SNE,” *Journal of Machine Learning Research* **9**, 2579–2605 (2008).

See the planned practical example in `examples/tsne/`.
