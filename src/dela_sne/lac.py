"""Canonical Locally Adaptive Clustering implementation.

The module implements the entropy-regularized local feature weighting described
by Domeniconi et al. and exposes the instrumentation required by the paper
experiments through a single implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

EPS = 1e-12
TemperatureRule = Callable[[np.ndarray, int], float | np.ndarray]


class _LACRun(TypedDict):
    cluster_centers: np.ndarray
    feature_weights: np.ndarray
    labels: np.ndarray
    distances: np.ndarray
    distance_to_assigned: np.ndarray
    n_iter: int
    status: str
    cycle_period: int
    h: np.ndarray
    distance_term: float
    entropy_term: float
    objective: float


def validate_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must have shape (n_samples, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    return X


def dispersions_from_labels(
    X: np.ndarray, labels: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return centroids and mean squared feature dispersions for a partition."""
    X = validate_array(X)
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (len(X),):
        raise ValueError("labels must have shape (n_samples,).")

    d = X.shape[1]
    centroids = np.empty((n_clusters, d), dtype=float)
    dispersions = np.empty((n_clusters, d), dtype=float)
    for k in range(n_clusters):
        members = X[labels == k]
        if len(members) == 0:
            raise ValueError("labels contain an empty cluster.")
        centroids[k] = np.mean(members, axis=0)
        dispersions[k] = np.mean((members - centroids[k]) ** 2, axis=0)
    return centroids, dispersions


def softmax_feature_weights(
    dispersions: np.ndarray, h: float | np.ndarray
) -> np.ndarray:
    """Compute entropy-regularized feature weights for every cluster."""
    V = np.asarray(dispersions, dtype=float)
    if V.ndim != 2:
        raise ValueError("dispersions must have shape (n_clusters, n_features).")

    temperature = np.asarray(h, dtype=float)
    if temperature.ndim == 0:
        temperature = np.full(V.shape[0], float(temperature))
    if temperature.shape != (V.shape[0],) or np.any(temperature <= 0):
        raise ValueError("h must be positive and scalar or length n_clusters.")

    logits = -V / temperature[:, None]
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights


def distance_matrix(
    X: np.ndarray, centroids: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    diff = X[:, None, :] - centroids[None, :, :]
    return np.sum(weights[None, :, :] * diff**2, axis=2)


def assign_weighted(
    X: np.ndarray, centroids: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return np.argmin(distance_matrix(X, centroids, weights), axis=1)


def _objective_components(
    distances: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    h: float | np.ndarray,
) -> tuple[float, float, float]:
    temperature = np.asarray(h, dtype=float)
    if temperature.ndim == 0:
        temperature = np.full(weights.shape[0], float(temperature))
    distance_term = float(np.sum(distances[np.arange(len(labels)), labels]))
    entropy_by_cluster = np.sum(weights * np.log(np.maximum(weights, EPS)), axis=1)
    entropy_term = float(np.sum(temperature * entropy_by_cluster))
    return distance_term, entropy_term, distance_term + entropy_term


@dataclass
class LAC:
    """Locally Adaptive Clustering with restart and convergence diagnostics.

    ``h`` has the physical dimension of a feature variance. Consequently, LAC
    is not scale-invariant with respect to ``h``; callers should standardize or
    otherwise define the scale of the features before comparing temperatures.
    """

    n_clusters: int = 3
    h: float = 0.5
    max_iter: int = 200
    tol: float = 1e-7
    n_init: int = 10
    random_state: int | None = 42

    def _validate_parameters(self, n_samples: int) -> None:
        if not 1 <= self.n_clusters <= n_samples:
            raise ValueError("n_clusters must be between 1 and n_samples.")
        if self.h <= 0:
            raise ValueError("h must be positive.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

    def _kmeanspp_centroids(self, X: np.ndarray, seed: int) -> np.ndarray:
        n = len(X)
        rng = np.random.default_rng(seed)
        chosen: set[int] = set()
        first = int(rng.integers(n))
        chosen.add(first)
        centroids = [X[first].copy()]
        closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)

        for _ in range(1, self.n_clusters):
            probabilities = closest_sq.copy()
            if chosen:
                probabilities[list(chosen)] = 0.0
            total = float(np.sum(probabilities))
            if total <= EPS:
                candidates = [i for i in range(n) if i not in chosen]
                idx = candidates[0] if candidates else int(rng.integers(n))
            else:
                idx = int(rng.choice(n, p=probabilities / total))
            chosen.add(idx)
            centroids.append(X[idx].copy())
            closest_sq = np.minimum(
                closest_sq, np.sum((X - centroids[-1]) ** 2, axis=1)
            )
        return np.vstack(centroids)

    def _fit_single(
        self,
        X: np.ndarray,
        seed: int,
        initial_labels: np.ndarray | None,
        temperature_rule: TemperatureRule | None,
    ) -> _LACRun:
        n, d = X.shape
        if initial_labels is None:
            centroids = self._kmeanspp_centroids(X, seed)
            weights = np.full((self.n_clusters, d), 1.0 / d)
            labels = np.argmin(distance_matrix(X, centroids, weights), axis=1)
        else:
            labels = np.asarray(initial_labels, dtype=int).copy()
            if labels.shape != (n,):
                raise ValueError("initial_labels must have shape (n_samples,).")
            if set(np.unique(labels)) != set(range(self.n_clusters)):
                raise ValueError("initial_labels must contain every cluster label.")
            centroids, V = dispersions_from_labels(X, labels, self.n_clusters)
            weights = softmax_feature_weights(V, self.h)

        seen: dict[bytes, int] = {labels.tobytes(): 0}
        status = "max_iter"
        cycle_period = 0
        h_current = np.full(self.n_clusters, self.h, dtype=float)

        for iteration in range(1, self.max_iter + 1):
            distances = distance_matrix(X, centroids, weights)
            new_labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            new_weights = weights.copy()
            used_reseeds: set[int] = set()

            # Compute non-empty clusters first so that their local temperatures
            # and weights are defined from the current partition.
            V = np.zeros((self.n_clusters, d), dtype=float)
            nonempty = np.zeros(self.n_clusters, dtype=bool)
            for k in range(self.n_clusters):
                members = X[new_labels == k]
                if len(members):
                    nonempty[k] = True
                    new_centroids[k] = np.mean(members, axis=0)
                    V[k] = np.mean((members - new_centroids[k]) ** 2, axis=0)

            h_value = (
                self.h
                if temperature_rule is None
                else temperature_rule(V.copy(), iteration)
            )
            h_current = np.asarray(h_value, dtype=float)
            if h_current.ndim == 0:
                h_current = np.full(self.n_clusters, float(h_current))
            if h_current.shape != (self.n_clusters,) or np.any(h_current <= 0):
                raise ValueError(
                    "temperature_rule must return a positive scalar or "
                    "length n_clusters."
                )

            if np.any(nonempty):
                new_weights[nonempty] = softmax_feature_weights(
                    V[nonempty], h_current[nonempty]
                )

            if np.any(~nonempty):
                nearest = np.min(distances, axis=1)
                order = np.argsort(nearest)[::-1]
                for k in np.where(~nonempty)[0]:
                    idx = next(
                        (int(i) for i in order if int(i) not in used_reseeds),
                        None,
                    )
                    if idx is None:
                        raise RuntimeError(
                            "Unable to choose a unique empty-cluster reseed."
                        )
                    used_reseeds.add(idx)
                    new_centroids[k] = X[idx]
                    new_weights[k] = 1.0 / d

            centroid_change = float(np.linalg.norm(new_centroids - centroids))
            labels_changed = not np.array_equal(new_labels, labels)
            centroids, weights, labels = new_centroids, new_weights, new_labels

            if not labels_changed or centroid_change <= self.tol:
                status = "converged"
                break

            key = labels.tobytes()
            if key in seen:
                status = "cycle"
                cycle_period = iteration - seen[key]
                break
            seen[key] = iteration

        final_distances = distance_matrix(X, centroids, weights)
        final_labels = np.argmin(final_distances, axis=1)
        distance_term, entropy_term, objective = _objective_components(
            final_distances, final_labels, weights, h_current
        )
        return {
            "cluster_centers": centroids,
            "feature_weights": weights,
            "labels": final_labels,
            "distances": final_distances,
            "distance_to_assigned": final_distances[np.arange(n), final_labels],
            "n_iter": iteration,
            "status": status,
            "cycle_period": cycle_period,
            "h": h_current,
            "distance_term": distance_term,
            "entropy_term": entropy_term,
            "objective": objective,
        }

    def fit(
        self,
        X: np.ndarray,
        *,
        initial_labels: np.ndarray | None = None,
        temperature_rule: TemperatureRule | None = None,
    ) -> LAC:
        X = validate_array(X)
        self._validate_parameters(len(X))
        if initial_labels is not None and self.n_init != 1:
            raise ValueError("Use n_init=1 when initial_labels are supplied.")

        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, np.iinfo(np.int32).max, size=self.n_init)
        runs = [
            self._fit_single(X, int(seed), initial_labels, temperature_rule)
            for seed in seeds
        ]
        best = min(runs, key=lambda run: run["objective"])

        self.cluster_centers_ = best["cluster_centers"].copy()
        self.feature_weights_ = best["feature_weights"].copy()
        self.labels_ = best["labels"].copy()
        self.distances_ = best["distances"].copy()
        self.distance_to_assigned_ = best["distance_to_assigned"].copy()
        self.n_iter_ = best["n_iter"]
        self.status_ = best["status"]
        self.cycle_period_ = best["cycle_period"]
        self.h_ = best["h"].copy()
        self.distance_term_ = best["distance_term"]
        self.entropy_term_ = best["entropy_term"]
        self.objective_ = best["objective"]
        self.restart_objectives_ = np.asarray(
            [run["objective"] for run in runs], dtype=float
        )
        return self

    def fit_predict(
        self,
        X: np.ndarray,
        *,
        initial_labels: np.ndarray | None = None,
        temperature_rule: TemperatureRule | None = None,
    ) -> np.ndarray:
        fitted = self.fit(
            X,
            initial_labels=initial_labels,
            temperature_rule=temperature_rule,
        )
        return fitted.labels_.copy()
