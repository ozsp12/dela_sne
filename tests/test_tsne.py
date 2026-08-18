import numpy as np

from dela_sne import TSNE, joint_probabilities, kl_divergence, kl_gradient


def sample_data(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal(-2.0, 0.25, size=(12, 4)),
            rng.normal(0.0, 0.25, size=(12, 4)),
            rng.normal(2.0, 0.25, size=(12, 4)),
        ]
    )


def test_tsne_is_deterministic_for_fixed_seed() -> None:
    X = sample_data()
    kwargs = dict(
        perplexity=8,
        n_iter=80,
        exaggeration_iter=20,
        init="random",
        random_state=11,
    )
    first = TSNE(**kwargs).fit_transform(X)
    second = TSNE(**kwargs).fit_transform(X)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_tsne_kl_improves_after_early_exaggeration() -> None:
    X = sample_data()
    model = TSNE(
        perplexity=8,
        n_iter=120,
        exaggeration_iter=30,
        init="pca",
        random_state=11,
    )
    model.fit_transform(X)
    post = [value for iteration, value in model.kl_history_ if iteration >= 30]
    assert len(post) >= 2
    assert post[-1] < post[0]


def test_tsne_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(7, 3))
    P, _, _ = joint_probabilities(X, perplexity=3)
    Y = rng.normal(scale=0.05, size=(7, 2))
    analytic = kl_gradient(P, Y)

    eps = 1e-6
    numeric = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            plus = Y.copy()
            minus = Y.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            numeric[i, j] = (
                kl_divergence(P, plus) - kl_divergence(P, minus)
            ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-5)
