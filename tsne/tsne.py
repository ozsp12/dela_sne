"""Reference NumPy implementation of exact t-SNE.

This implementation is intentionally explicit and quadratic in the number of
samples. It is intended for reproducible small test datasets, not large-scale
production workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


EPS = 1e-12


def _validate_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("X must have shape (n_samples >= 3, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    return X


def _squared_distances(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X * X, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * X @ X.T
    return np.maximum(distances, 0.0)


def _conditional_row(
    distances: np.ndarray,
    self_index: int,
    perplexity: float,
    tolerance: float = 1e-5,
    max_iter: int = 80,
) -> tuple[np.ndarray, float]:
    n = len(distances)
    if not 1.0 < perplexity < n:
        raise ValueError("perplexity must satisfy 1 < perplexity < n_samples.")

    mask = np.ones(n, dtype=bool)
    mask[self_index] = False
    d = np.asarray(distances[mask], dtype=float)
    d -= np.min(d)

    target_entropy = np.log(perplexity)
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf
    p = np.full(n - 1, 1.0 / (n - 1))

    for _ in range(max_iter):
        values = np.exp(-beta * d)
        total = float(np.sum(values))
        if total <= EPS:
            values = np.ones_like(d)
            total = float(len(d))

        p = values / total
        entropy = np.log(total) + beta * float(np.sum(d * values) / total)
        error = entropy - target_entropy

        if abs(error) <= tolerance:
            break
        if error > 0:
            beta_min = beta
            beta = 2.0 * beta if np.isinf(beta_max) else 0.5 * (beta + beta_max)
        else:
            beta_max = beta
            beta = 0.5 * beta if np.isinf(beta_min) else 0.5 * (beta + beta_min)

    row = np.zeros(n)
    row[mask] = p
    sigma = np.sqrt(1.0 / (2.0 * max(beta, EPS)))
    return row, float(sigma)


def _joint_probabilities(
    X: np.ndarray, perplexity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = _squared_distances(X)
    n = len(X)
    conditional = np.zeros((n, n))
    sigmas = np.zeros(n)

    for i in range(n):
        conditional[i], sigmas[i] = _conditional_row(
            distances[i], i, perplexity
        )

    P = (conditional + conditional.T) / (2.0 * n)
    P = np.maximum(P, EPS)
    np.fill_diagonal(P, 0.0)
    P /= np.sum(P)
    return P, conditional, sigmas


@dataclass
class TSNE:
    """Exact t-distributed Stochastic Neighbor Embedding."""

    n_components: int = 2
    perplexity: float = 30.0
    learning_rate: float = 100.0
    n_iter: int = 500
    early_exaggeration: float = 4.0
    exaggeration_iter: int = 100
    random_state: int | None = 42

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = _validate_array(X)
        if self.n_components < 1:
            raise ValueError("n_components must be positive.")
        if self.learning_rate <= 0 or self.n_iter < 1:
            raise ValueError("learning_rate and n_iter must be positive.")

        n = len(X)
        P, conditional, sigmas = _joint_probabilities(X, self.perplexity)
        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(0.0, 1e-4, size=(n, self.n_components))
        velocity = np.zeros_like(Y)
        history: list[tuple[int, float]] = []

        for iteration in range(self.n_iter):
            low_distances = _squared_distances(Y)
            numerator = 1.0 / (1.0 + low_distances)
            np.fill_diagonal(numerator, 0.0)
            Q = numerator / max(float(np.sum(numerator)), EPS)
            Q = np.maximum(Q, EPS)
            np.fill_diagonal(Q, 0.0)
            Q /= np.sum(Q)

            if iteration < self.exaggeration_iter:
                target = self.early_exaggeration * P
                momentum = 0.5
            else:
                target = P
                momentum = 0.8

            A = (target - Q) * numerator
            gradient = 4.0 * (
                np.sum(A, axis=1)[:, None] * Y - A @ Y
            )
            velocity = momentum * velocity - self.learning_rate * gradient
            Y += velocity
            Y -= np.mean(Y, axis=0, keepdims=True)

            if iteration % 10 == 0 or iteration == self.n_iter - 1:
                kl = float(np.sum(P * np.log((P + EPS) / (Q + EPS))))
                history.append((iteration, kl))

        self.embedding_ = Y
        self.P_ = P
        self.P_conditional_ = conditional
        self.sigmas_ = sigmas
        self.kl_history_ = history
        self.kl_divergence_ = history[-1][1]
        return Y.copy()
