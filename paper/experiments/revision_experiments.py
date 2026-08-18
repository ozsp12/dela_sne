"""Scientific revision experiments for the manuscript.

This module adds the tests requested by the major-revision assessment without
reimplementing the LAC algorithm. All clustering updates are executed by the
canonical ``dela_sne`` package. The module adds four diagnostics:

1. two-level (Schottky) theory versus prescribed, true-partition, and estimated
   feature-dispersion spectra;
2. unsupervised temperature selection baselines based on silhouette and
   perturbation stability;
3. controlled robustness tests for overlap, imbalance, added dimensions, and
   rotation away from axis-aligned relevance;
4. direct numerical validation of the SNE first-order perplexity-tolerance rule
   over several perplexities.
"""

from __future__ import annotations

import csv
from itertools import combinations
from math import exp, log

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler

import _experiment_body as base
from dela_sne import dispersions_from_labels
from dela_sne.lac_diagnostics import iterate_with_temperature_rule

SEED = base.SEED
H_BASE = np.geomspace(0.02, 6.0, 36)
STRESS_SEEDS = tuple(SEED + i for i in range(8))
BASELINE_SEEDS = tuple(SEED + i for i in range(10))
REAL_BASELINE_SEEDS = tuple(SEED + i for i in range(5))


def _write_rows(filename: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (base.ASSET_DIR / filename).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: base.csv_value(row.get(k, "")) for k in fieldnames})


def _fixed_lac(X: np.ndarray, n_clusters: int, h: float, init: np.ndarray) -> np.ndarray:
    labels, *_ = iterate_with_temperature_rule(
        X,
        n_clusters,
        init,
        lambda V, iteration: float(h),
        max_iter=base.MAX_ITER,
    )
    return labels


def _response_h(X: np.ndarray, n_clusters: int, init: np.ndarray, grid=H_BASE) -> float:
    _, V = dispersions_from_labels(X, init, n_clusters)
    return base.select_global_response_h(V, np.asarray(grid, dtype=float))


def _align_partition(reference: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    table = contingency_matrix(reference, labels, sparse=False)
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(col): int(row) for row, col in zip(rows, cols, strict=True)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def _schottky_xstar(g0: int, g1: int) -> float:
    ratio = g1 / g0

    def f(x: float) -> float:
        u = ratio * exp(-x)
        return x - 2.0 * (1.0 + u) / (1.0 - u)

    lower = max(2.0000001, log(ratio) + 1.0e-7) if ratio > 1 else 2.0000001
    return float(brentq(f, lower, 20.0))


def _schottky_response(h: np.ndarray, g0: int, g1: int, delta: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    x = delta / h
    u = (g1 / g0) * np.exp(-x)
    return x**2 * u / (1.0 + u) ** 2


def _band_statistics(values: np.ndarray, relevant: list[int]) -> dict[str, float]:
    mask = np.zeros(len(values), dtype=bool)
    mask[relevant] = True
    low = np.asarray(values[mask], dtype=float)
    high = np.asarray(values[~mask], dtype=float)
    mean_low = float(np.mean(low))
    mean_high = float(np.mean(high))
    pooled = float(np.sqrt(0.5 * (np.var(low) + np.var(high))))
    return {
        "mean_relevant": mean_low,
        "mean_irrelevant": mean_high,
        "variance_ratio": mean_high / max(mean_low, np.finfo(float).tiny),
        "gap": float(np.min(high) - np.max(low)),
        "band_separation": (mean_high - mean_low) / max(pooled, 1.0e-12),
    }


def schottky_validation() -> list[dict]:
    X, y, relevant = base.make_structured_synthetic(SEED)
    n_clusters = base.K_SYN
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED).fit_predict(X)
    km = _align_partition(y, km, n_clusters)
    _, V_true = dispersions_from_labels(X, y, n_clusters)
    _, V_km = dispersions_from_labels(X, km, n_clusters)

    rel_sd = np.asarray([0.18, 0.38, 0.65])
    irr_sd = np.asarray([1.45, 1.65, 1.90])
    V_prescribed = np.empty_like(V_true)
    for ell in range(n_clusters):
        V_prescribed[ell] = irr_sd[ell] ** 2
        V_prescribed[ell, relevant[ell]] = rel_sd[ell] ** 2

    h_dense = np.geomspace(0.015, 8.0, 1200)
    rows: list[dict] = []
    fig, axes = plt.subplots(n_clusters, 2, figsize=(8.0, 9.5))

    for ell in range(n_clusters):
        g0 = len(relevant[ell])
        g1 = X.shape[1] - g0
        e0 = rel_sd[ell] ** 2
        e1 = irr_sd[ell] ** 2
        delta = e1 - e0
        xstar = _schottky_xstar(g0, g1)
        hstar_analytic = delta / xstar

        spectra = {
            "prescribed": V_prescribed[ell],
            "true_partition": V_true[ell],
            "kmeans_partition": V_km[ell],
        }
        curves = {
            name: base.response_curve(values, h_dense) for name, values in spectra.items()
        }
        hstars_global = {
            name: float(h_dense[int(np.argmax(curve))]) for name, curve in curves.items()
        }
        window = (h_dense >= hstar_analytic / 2.0) & (h_dense <= 2.0 * hstar_analytic)
        window_idx = np.flatnonzero(window)
        hstars_gap = {
            name: float(h_dense[window_idx[int(np.argmax(curve[window]))]])
            for name, curve in curves.items()
        }

        for spectrum_name, values in spectra.items():
            stats = _band_statistics(values, relevant[ell])
            rows.append(
                {
                    "cluster": ell + 1,
                    "spectrum": spectrum_name,
                    "g_relevant": g0,
                    "g_irrelevant": g1,
                    "delta_prescribed": delta,
                    "xstar_analytic": xstar,
                    "hstar_analytic": hstar_analytic,
                    "hstar_global_numeric": hstars_global[spectrum_name],
                    "hstar_gap_numeric": hstars_gap[spectrum_name],
                    "relative_gap_hstar_error": (
                        hstars_gap[spectrum_name] - hstar_analytic
                    ) / hstar_analytic,
                    # Temporary compatibility aliases used only while the already
                    # materialized manuscript helper is being retired.
                    "hstar_numeric": hstars_global[spectrum_name],
                    "relative_hstar_error": (
                        hstars_global[spectrum_name] - hstar_analytic
                    ) / hstar_analytic,
                    "low_temperature_peak": int(
                        hstars_global[spectrum_name] < 0.5 * hstar_analytic
                    ),
                    **stats,
                }
            )

        ax_e = axes[ell, 0]
        features = np.arange(1, X.shape[1] + 1)
        ax_e.plot(features, V_prescribed[ell], marker="o", linewidth=1.0, label="prescribed")
        ax_e.plot(features, V_true[ell], marker="s", linewidth=1.0, label="true partition")
        ax_e.plot(features, V_km[ell], marker="^", linewidth=1.0, label="$k$-means partition")
        ax_e.set_ylabel(rf"cluster {ell + 1}: $V_{{\ell f}}$")
        ax_e.set_xlabel("feature index")
        if ell == 0:
            ax_e.legend(frameon=False, fontsize=8)

        ax_c = axes[ell, 1]
        ax_c.plot(
            h_dense,
            _schottky_response(h_dense, g0, g1, delta),
            linewidth=1.5,
            label="two-level Schottky",
        )
        ax_c.plot(h_dense, curves["true_partition"], linestyle="--", label="true partition")
        ax_c.plot(h_dense, curves["kmeans_partition"], linestyle=":", label="$k$-means partition")
        ax_c.axvline(hstar_analytic, linewidth=0.9, linestyle="-.")
        ax_c.set_xscale("log")
        ax_c.set_ylabel(rf"$C_{{{ell + 1}}}^{{(F)}}(h)$")
        ax_c.set_xlabel(r"feature temperature $h$")
        if ell == 0:
            ax_c.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    base.save_figure(fig, "lac_schottky_validation")
    plt.close(fig)

    kmeans_curves = []
    for ell in range(n_clusters):
        kmeans_curves.append(base.response_curve(V_km[ell], h_dense))
    global_mean_curve = np.mean(np.vstack(kmeans_curves), axis=0)
    global_mean_hstar = float(h_dense[int(np.argmax(global_mean_curve))])
    for row in rows:
        row["kmeans_global_mean_hstar"] = global_mean_hstar

    _write_rows(
        "lac_schottky_diagnostics.csv",
        [
            "cluster",
            "spectrum",
            "g_relevant",
            "g_irrelevant",
            "delta_prescribed",
            "xstar_analytic",
            "hstar_analytic",
            "hstar_global_numeric",
            "hstar_gap_numeric",
            "relative_gap_hstar_error",
            "hstar_numeric",
            "relative_hstar_error",
            "low_temperature_peak",
            "kmeans_global_mean_hstar",
            "mean_relevant",
            "mean_irrelevant",
            "variance_ratio",
            "gap",
            "band_separation",
        ],
        rows,
    )
    return rows


def _silhouette_and_oracle_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    silhouettes = np.full(len(H_BASE), -np.inf)
    ari = np.empty(len(H_BASE))
    partitions: list[np.ndarray] = []
    for q, h in enumerate(H_BASE):
        labels = _fixed_lac(X, n_clusters, float(h), init)
        partitions.append(labels)
        ari[q] = adjusted_rand_score(y, labels)
        if len(np.unique(labels)) > 1:
            silhouettes[q] = silhouette_score(X, labels, metric="euclidean")
    return silhouettes, ari, partitions


def _stability_curve(X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    scale = np.std(X, axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    rng = np.random.default_rng(seed)
    perturbations = [
        X + rng.normal(scale=0.02 * scale, size=X.shape) for _ in range(4)
    ]
    scores = np.empty(len(H_BASE))
    for q, h in enumerate(H_BASE):
        labels_set = []
        for r, Xp in enumerate(perturbations):
            init = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=seed + 97 * r,
            ).fit_predict(Xp)
            labels_set.append(_fixed_lac(Xp, n_clusters, float(h), init))
        pair_scores = [
            adjusted_rand_score(a, b) for a, b in combinations(labels_set, 2)
        ]
        scores[q] = float(np.mean(pair_scores))
    return scores


def _baseline_rows_for_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    seeds: tuple[int, ...],
) -> list[dict]:
    n_clusters = len(np.unique(y))
    rows: list[dict] = []
    for seed in seeds:
        init = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)
        sil, ari_grid, partitions = _silhouette_and_oracle_grid(X, y, n_clusters, init)
        stability = _stability_curve(X, n_clusters, seed)
        _, V0 = dispersions_from_labels(X, init, n_clusters)
        response = np.mean(
            np.vstack([base.response_curve(V0[k], H_BASE) for k in range(n_clusters)]),
            axis=0,
        )
        selectors = {
            "response": int(np.argmax(response)),
            "silhouette": int(np.argmax(sil)),
            "stability": int(np.argmax(stability)),
            "oracle": int(np.argmax(ari_grid)),
        }
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "selector": "kmeans",
                "h": np.nan,
                "selection_score": np.nan,
                "ari": adjusted_rand_score(y, init),
            }
        )
        for selector, idx in selectors.items():
            score = {
                "response": response[idx],
                "silhouette": sil[idx],
                "stability": stability[idx],
                "oracle": ari_grid[idx],
            }[selector]
            rows.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "selector": selector,
                    "h": H_BASE[idx],
                    "selection_score": score,
                    "ari": ari_grid[idx],
                }
            )
    return rows


def unsupervised_selector_validation() -> list[dict]:
    rows: list[dict] = []
    for seed in BASELINE_SEEDS:
        X, y, _ = base.make_structured_synthetic(seed)
        rows.extend(_baseline_rows_for_dataset("Synthetic", X, y, (seed,)))

    real = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("WDBC", *load_breast_cancer(return_X_y=True)),
    ]
    for name, X, y in real:
        X = StandardScaler().fit_transform(X)
        rows.extend(_baseline_rows_for_dataset(name, X, y, REAL_BASELINE_SEEDS))

    _write_rows(
        "lac_unsupervised_baselines.csv",
        ["dataset", "seed", "selector", "h", "selection_score", "ari"],
        rows,
    )

    datasets = ["Synthetic", "Iris", "Wine", "WDBC"]
    selectors = ["kmeans", "response", "silhouette", "stability", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.6))
    for ax, dataset in zip(axes.flat, datasets, strict=True):
        means = []
        sds = []
        active = []
        for selector in selectors:
            vals = np.asarray(
                [r["ari"] for r in rows if r["dataset"] == dataset and r["selector"] == selector],
                dtype=float,
            )
            if len(vals):
                active.append(selector)
                means.append(float(np.mean(vals)))
                sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        x = np.arange(len(active))
        ax.errorbar(x, means, yerr=sds, marker="o", linestyle="none", capsize=3)
        ax.set_xticks(x, active, rotation=28, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("ARI")
        ax.set_title(dataset)
    fig.tight_layout()
    base.save_figure(fig, "lac_selector_baselines")
    plt.close(fig)
    return rows


def _general_synthetic(
    seed: int,
    *,
    center_factor: float = 1.0,
    counts: tuple[int, int, int] = (240, 240, 240),
    extra_dimensions: int = 0,
    rotation_degrees: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    relevant = [list(range(0, 2)), list(range(2, 6)), list(range(6, 12))]
    relevant_scales = [0.18, 0.38, 0.65]
    irrelevant_scales = [1.45, 1.65, 1.90]
    center_amplitudes = [2.40, 1.80, 1.40]
    patterns = [
        np.array([-1.0, 1.0]),
        np.array([1.0, -1.0, 1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
    ]
    D = 12 + extra_dimensions
    Xs = []
    ys = []
    for ell in range(3):
        Z = rng.normal(size=(counts[ell], D))
        Z -= Z.mean(axis=0)
        Z /= np.sqrt(np.mean(Z**2, axis=0))
        scales = np.full(D, irrelevant_scales[ell])
        scales[relevant[ell]] = relevant_scales[ell]
        center = np.zeros(D)
        center[relevant[ell]] = center_factor * center_amplitudes[ell] * patterns[ell]
        Xs.append(center + Z * scales)
        ys.append(np.full(counts[ell], ell, dtype=int))
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    theta = np.deg2rad(rotation_degrees)
    if theta != 0.0:
        c, s = np.cos(theta), np.sin(theta)
        for a, b in ((0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)):
            xa = X[:, a].copy()
            xb = X[:, b].copy()
            X[:, a] = c * xa - s * xb
            X[:, b] = s * xa + c * xb
    return X, y


def _evaluate_response_condition(X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    K = len(np.unique(y))
    init = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(X)
    h_response = _response_h(X, K, init, base.H_GRID)
    labels_response = _fixed_lac(X, K, h_response, init)
    labels_h1 = _fixed_lac(X, K, 1.0, init)
    return {
        "kmeans": adjusted_rand_score(y, init),
        "fixed_h1": adjusted_rand_score(y, labels_h1),
        "response": adjusted_rand_score(y, labels_response),
        "response_h": h_response,
    }


def stress_tests() -> list[dict]:
    configurations = {
        "center_factor": [0.50, 0.75, 1.00, 1.25],
        "imbalance_ratio": [1.0, 1.5, 2.0, 4.0],
        "dimensions": [12, 24, 48, 96],
        "rotation_degrees": [0.0, 15.0, 30.0, 45.0],
    }
    rows: list[dict] = []
    for factor, values in configurations.items():
        for value in values:
            for seed in STRESS_SEEDS:
                kwargs = {}
                if factor == "center_factor":
                    kwargs["center_factor"] = float(value)
                elif factor == "imbalance_ratio":
                    ratio = float(value)
                    kwargs["counts"] = (240, int(round(240 / np.sqrt(ratio))), int(round(240 / ratio)))
                elif factor == "dimensions":
                    kwargs["extra_dimensions"] = int(value) - 12
                elif factor == "rotation_degrees":
                    kwargs["rotation_degrees"] = float(value)
                X, y = _general_synthetic(seed, **kwargs)
                result = _evaluate_response_condition(X, y, seed)
                rows.append({"factor": factor, "value": value, "seed": seed, **result})

    _write_rows(
        "lac_stress_tests.csv",
        ["factor", "value", "seed", "kmeans", "fixed_h1", "response", "response_h"],
        rows,
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.8))
    for ax, (factor, values) in zip(axes.flat, configurations.items(), strict=True):
        for method in ("kmeans", "fixed_h1", "response"):
            means = []
            sds = []
            for value in values:
                vals = np.asarray(
                    [
                        r[method]
                        for r in rows
                        if r["factor"] == factor and float(r["value"]) == float(value)
                    ],
                    dtype=float,
                )
                means.append(float(np.mean(vals)))
                sds.append(float(np.std(vals, ddof=1)))
            ax.errorbar(values, means, yerr=sds, marker="o", label=method)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(factor.replace("_", " "))
        ax.set_ylabel("ARI")
        if factor == "dimensions":
            ax.set_xscale("log", base=2)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    base.save_figure(fig, "lac_stress_tests")
    plt.close(fig)
    return rows


def _perplexity_from_beta(energies: np.ndarray, beta: float) -> float:
    entropy, _ = base.entropy_for_beta(energies, beta)
    return float(np.exp(entropy))


def sne_perplexity_validation() -> list[dict]:
    X, _ = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    rows: list[dict] = []
    for perplexity in (5.0, 20.0, 30.0, 50.0):
        T, C, _ = base.local_temperatures(X, perplexity)
        for i in range(len(X)):
            energies = D2[i, np.arange(len(X)) != i]
            beta = 1.0 / T[i]
            tolerance = 0.01 / (2.0 * C[i])
            beta_perturbed = beta / (1.0 + tolerance) ** 2
            actual_perp = _perplexity_from_beta(energies, beta_perturbed)
            relative_error = abs(actual_perp / perplexity - 1.0)
            rows.append(
                {
                    "perplexity": perplexity,
                    "anchor": i,
                    "response": C[i],
                    "predicted_sigma_tolerance": tolerance,
                    "actual_perplexity_relative_error": relative_error,
                    "linearization_ratio": relative_error / 0.01,
                }
            )

    _write_rows(
        "sne_perplexity_validation.csv",
        [
            "perplexity",
            "anchor",
            "response",
            "predicted_sigma_tolerance",
            "actual_perplexity_relative_error",
            "linearization_ratio",
        ],
        rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    perps = [5.0, 20.0, 30.0, 50.0]
    response_median = []
    response_q95 = []
    tolerance_median = []
    actual_error_median = []
    for perplexity in perps:
        subset = [r for r in rows if r["perplexity"] == perplexity]
        response = np.asarray([r["response"] for r in subset])
        tolerance = np.asarray([r["predicted_sigma_tolerance"] for r in subset])
        actual = np.asarray([r["actual_perplexity_relative_error"] for r in subset])
        response_median.append(float(np.median(response)))
        response_q95.append(float(np.quantile(response, 0.95)))
        tolerance_median.append(float(np.median(tolerance)))
        actual_error_median.append(float(np.median(actual)))
    axes[0].plot(perps, response_median, marker="o", label="median response")
    axes[0].plot(perps, response_q95, marker="s", linestyle="--", label="95th percentile")
    axes[0].set_xlabel("perplexity")
    axes[0].set_ylabel(r"neighbor response $C_i^{(N)}$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(perps, 100 * np.asarray(tolerance_median), marker="o", label="predicted $\sigma$ tolerance")
    axes[1].plot(perps, 100 * np.asarray(actual_error_median), marker="s", linestyle="--", label="actual perplexity error")
    axes[1].axhline(1.0, linewidth=0.8, linestyle=":", label="1% target error")
    axes[1].set_xlabel("perplexity")
    axes[1].set_ylabel("percent")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    base.save_figure(fig, "sne_perplexity_validation")
    plt.close(fig)
    return rows


def main() -> None:
    schottky = schottky_validation()
    selectors = unsupervised_selector_validation()
    stress = stress_tests()
    sne = sne_perplexity_validation()
    print("Revision diagnostics")
    print("Schottky rows", len(schottky))
    print("Selector rows", len(selectors))
    print("Stress rows", len(stress))
    print("SNE tolerance rows", len(sne))
