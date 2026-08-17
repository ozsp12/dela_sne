"""Reference implementation of Locally Adaptive Clustering (LAC).

The implementation follows the cluster-dependent weighted squared distance and
entropy-regularized exponential feature-weight update of Domeniconi et al.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


EPS = 1e-12


def _validate_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must have shape (n_samples, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    return X


def _softmax_negative(values: np.ndarray, h: float) -> np.ndarray:
    if h <= 0:
        raise ValueError("h must be positive.")
    z = -np.asarray(values, dtype=float) / h
    z -= np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass
class LAC:
    """Locally Adaptive Clustering with entropy-regularized feature weights."""

    n_clusters: int = 3
    h: float = 0.5
    max_iter: int = 200
    tol: float = 1e-7
    random_state: int | None = 42

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        n, _ = X.shape
        if not 1 <= self.n_clusters <= n:
            raise ValueError("n_clusters must be between 1 and n_samples.")

        rng = np.random.default_rng(self.random_state)
        centroids = [X[rng.integers(n)].copy()]
        closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)

        for _ in range(1, self.n_clusters):
            total = float(np.sum(closest_sq))
            if total <= EPS:
                candidates = [
                    i for i in range(n)
                    if not any(np.array_equal(X[i], c) for c in centroids)
                ]
                idx = candidates[0] if candidates else int(rng.integers(n))
            else:
                idx = int(rng.choice(n, p=closest_sq / total))
            centroids.append(X[idx].copy())
            new_sq = np.sum((X - centroids[-1]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, new_sq)

        return np.vstack(centroids)

    @staticmethod
    def _distance_matrix(
        X: np.ndarray, centroids: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        diff = X[:, None, :] - centroids[None, :, :]
        return np.sum(weights[None, :, :] * diff**2, axis=2)

    def fit(self, X: np.ndarray) -> "LAC":
        X = _validate_array(X)
        if self.h <= 0:
            raise ValueError("h must be positive.")

        n, d = X.shape
        centroids = self._initialize_centroids(X)
        weights = np.full((self.n_clusters, d), 1.0 / d)
        previous_labels = None

        for iteration in range(self.max_iter):
            distances = self._distance_matrix(X, centroids, weights)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            new_weights = weights.copy()

            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    nearest = np.min(distances, axis=1)
                    idx = int(np.argmax(nearest))
                    new_centroids[k] = X[idx]
                    new_weights[k] = 1.0 / d
                    continue

                center = np.mean(members, axis=0)
                dispersion = np.mean((members - center) ** 2, axis=0)
                new_centroids[k] = center
                new_weights[k] = _softmax_negative(dispersion, self.h)

            centroid_change = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            weights = new_weights

            if previous_labels is not None and (
                np.array_equal(labels, previous_labels) or centroid_change < self.tol
            ):
                break
            previous_labels = labels.copy()

        final_distances = self._distance_matrix(X, centroids, weights)
        labels = np.argmin(final_distances, axis=1)

        self.cluster_centers_ = centroids
        self.feature_weights_ = weights
        self.labels_ = labels
        self.distances_ = final_distances
        self.distance_to_assigned_ = final_distances[np.arange(n), labels]
        self.n_iter_ = iteration + 1
        self.objective_ = float(np.sum(self.distance_to_assigned_))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_.copy()
