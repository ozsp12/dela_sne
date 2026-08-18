"""LAC iteration hooks used by manuscript experiments.

The paper may choose an external initial partition or an adaptive temperature
rule, but all LAC updates are executed here rather than reimplemented by the
experiment driver.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .lac import assign_weighted, dispersions_from_labels, softmax_feature_weights

TemperatureRule = Callable[[np.ndarray, int], float | np.ndarray]


def iterate_with_temperature_rule(
    X: np.ndarray,
    n_clusters: int,
    init_labels: np.ndarray,
    temperature_rule: TemperatureRule,
    max_iter: int = 100,
) -> tuple[np.ndarray, float | np.ndarray, int, str, int, list[np.ndarray]]:
    """Iterate LAC from an explicit partition with cycle diagnostics.

    This function intentionally mirrors the manuscript experiment contract.
    Initialization is external to the LAC update and is therefore an explicit
    experimental choice rather than a second algorithm implementation.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    labels = np.asarray(init_labels, dtype=int).copy()
    if labels.shape != (len(X),):
        raise ValueError("init_labels must have shape (n_samples,).")
    if set(np.unique(labels)) != set(range(n_clusters)):
        raise ValueError("init_labels must contain every cluster label.")

    seen: dict[bytes, int] = {labels.tobytes(): 0}
    h_path: list[np.ndarray] = []
    h: float | np.ndarray = np.nan

    for iteration in range(1, max_iter + 1):
        centroids, dispersions = dispersions_from_labels(X, labels, n_clusters)
        h = temperature_rule(dispersions.copy(), iteration)
        h_array = np.asarray(h, dtype=float)
        h_path.append(h_array.copy())
        weights = softmax_feature_weights(dispersions, h_array)
        new_labels = assign_weighted(X, centroids, weights)

        if np.array_equal(new_labels, labels):
            return new_labels, h, iteration, "converged", 0, h_path

        key = new_labels.tobytes()
        if key in seen:
            return (
                new_labels,
                h,
                iteration,
                "cycle",
                iteration - seen[key],
                h_path,
            )
        seen[key] = iteration
        labels = new_labels

    return labels, h, max_iter, "max_iter", 0, h_path
