"""Exact NumPy t-SNE reference implementation.

The implementation is intentionally O(n^2) and favors inspectability over
large-scale performance. It includes PCA initialization, adaptive gains, an
automatic sample-size-aware learning rate, and a gradient-norm stopping rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

EPS = 1e-12


def validate_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("X must have shape (n_samples >= 3, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    return X


def squared_distances(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X * X, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * X @ X.T
    return np.maximum(distances, 0.0)


def conditional_row(
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
    # Subtracting a row-wise constant changes the partition function but not
    # normalized probabilities or Shannon entropy.
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


def joint_probabilities(
    X: np.ndarray, perplexity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = squared_distances(X)
    n = len(X)
    conditional = np.zeros((n, n))
    sigmas = np.zeros(n)
    for i in range(n):
        conditional[i], sigmas[i] = conditional_row(distances[i], i, perplexity)

    P = (conditional + conditional.T) / (2.0 * n)
    np.fill_diagonal(P, 0.0)
    P = np.maximum(P, EPS)
    np.fill_diagonal(P, 0.0)
    P /= np.sum(P)
    return P, conditional, sigmas


def low_dimensional_probabilities(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = squared_distances(Y)
    numerator = 1.0 / (1.0 + distances)
    np.fill_diagonal(numerator, 0.0)
    Q = numerator / max(float(np.sum(numerator)), EPS)
    Q = np.maximum(Q, EPS)
    np.fill_diagonal(Q, 0.0)
    Q /= np.sum(Q)
    return Q, numerator


def kl_divergence(P: np.ndarray, Y: np.ndarray) -> float:
    Q, _ = low_dimensional_probabilities(Y)
    return float(np.sum(P * np.log((P + EPS) / (Q + EPS))))


def kl_gradient(P: np.ndarray, Y: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Analytic gradient of KL(scale*P || Q) with respect to Y."""
    Q, numerator = low_dimensional_probabilities(Y)
    A = (scale * P - Q) * numerator
    return 4.0 * (np.sum(A, axis=1)[:, None] * Y - A @ Y)


def _pca_initialization(X: np.ndarray, n_components: int) -> np.ndarray:
    centered = X - np.mean(X, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    Y = centered @ components
    std = float(np.std(Y))
    if std > 0:
        Y = Y / std * 1e-4
    return Y


@dataclass
class TSNE:
    """Exact t-distributed Stochastic Neighbor Embedding.

    ``learning_rate='auto'`` uses ``max(n / (4 * early_exaggeration), 50)``.
    The optimizer follows the classic momentum scheme and adaptive gain update.
    KL is expected to improve after early exaggeration, but strict monotonicity
    at every iteration is not a mathematical guarantee when momentum is used.
    """

    n_components: int = 2
    perplexity: float = 30.0
    learning_rate: float | Literal["auto"] = "auto"
    n_iter: int = 1000
    early_exaggeration: float = 12.0
    exaggeration_iter: int = 250
    init: Literal["random", "pca"] = "pca"
    adaptive_gains: bool = True
    min_grad_norm: float = 1e-7
    random_state: int | None = 42

    def _validate_parameters(self, n: int) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be positive.")
        if not 1.0 < self.perplexity < n:
            raise ValueError("perplexity must satisfy 1 < perplexity < n_samples.")
        if self.n_iter < 1:
            raise ValueError("n_iter must be positive.")
        if self.exaggeration_iter < 0 or self.exaggeration_iter > self.n_iter:
            raise ValueError("exaggeration_iter must lie between 0 and n_iter.")
        if self.early_exaggeration <= 0:
            raise ValueError("early_exaggeration must be positive.")
        if self.learning_rate != "auto" and float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive or 'auto'.")
        if self.min_grad_norm < 0:
            raise ValueError("min_grad_norm must be non-negative.")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = validate_array(X)
        n = len(X)
        self._validate_parameters(n)
        P, conditional, sigmas = joint_probabilities(X, self.perplexity)

        rng = np.random.default_rng(self.random_state)
        if self.init == "pca":
            Y = _pca_initialization(X, self.n_components)
        elif self.init == "random":
            Y = rng.normal(0.0, 1e-4, size=(n, self.n_components))
        else:
            raise ValueError("init must be 'random' or 'pca'.")

        if self.learning_rate == "auto":
            learning_rate = max(n / (4.0 * self.early_exaggeration), 50.0)
        else:
            learning_rate = float(self.learning_rate)

        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)
        previous_gradient = np.zeros_like(Y)
        history: list[tuple[int, float]] = []
        status = "max_iter"

        for iteration in range(self.n_iter):
            scale = (
                self.early_exaggeration
                if iteration < self.exaggeration_iter
                else 1.0
            )
            momentum = 0.5 if iteration < self.exaggeration_iter else 0.8
            gradient = kl_gradient(P, Y, scale=scale)
            grad_norm = float(np.linalg.norm(gradient))

            if self.adaptive_gains:
                changed_sign = np.sign(gradient) != np.sign(previous_gradient)
                gains = np.where(changed_sign, gains + 0.2, gains * 0.8)
                gains = np.maximum(gains, 0.01)
            else:
                gains.fill(1.0)

            velocity = momentum * velocity - learning_rate * gains * gradient
            Y += velocity
            Y -= np.mean(Y, axis=0, keepdims=True)
            previous_gradient = gradient

            if iteration % 10 == 0 or iteration == self.n_iter - 1:
                history.append((iteration, kl_divergence(P, Y)))

            if iteration >= self.exaggeration_iter and grad_norm <= self.min_grad_norm:
                status = "gradient_tolerance"
                break

        self.embedding_ = Y.copy()
        self.P_ = P
        self.P_conditional_ = conditional
        self.sigmas_ = sigmas
        self.kl_history_ = history
        self.kl_divergence_ = history[-1][1]
        self.learning_rate_ = learning_rate
        self.n_iter_ = iteration + 1
        self.status_ = status
        return self.embedding_.copy()
