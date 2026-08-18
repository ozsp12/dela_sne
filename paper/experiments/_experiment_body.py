"""Reproduce the numerical experiments used by the manuscript.

The experiment directory is self-contained inside ``paper/``. Each execution
overwrites stable output names:

- CSV results -> ``paper/experiments/assets/``;
- PNG companion/source exports -> ``paper/experiments/assets/source_images/``;
- PDF figures consumed by LaTeX -> ``paper/figures/``.

The script uses datasets bundled with scikit-learn and requires no network
request. All stochastic experiments use explicitly recorded seeds.
"""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

SEED = 1729
N_SEEDS = 30
K_SYN = 3
H_GRID = np.geomspace(0.01, 8.0, 160)  # common response/oracle grid and domain
MAX_ITER = 100


def csv_value(value):
    """Return a deterministic, platform-stable representation for CSV output."""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{value:.12g}"
    return value
EXPERIMENT_DIR = Path(__file__).resolve().parent
PAPER_DIR = EXPERIMENT_DIR.parent
ASSET_DIR = EXPERIMENT_DIR / "assets"
SOURCE_IMAGE_DIR = ASSET_DIR / "source_images"
FIGURE_DIR = PAPER_DIR / "figures"

for directory in (ASSET_DIR, SOURCE_IMAGE_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem: str) -> None:
    """Overwrite deterministic PDF and PNG exports for a manuscript figure."""
    fig.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        SOURCE_IMAGE_DIR / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "DELA-SNE paper experiment"},
    )


def canonical_stats(energies: np.ndarray, temperature: float):
    e = np.asarray(energies, dtype=float)
    logits = -(e - e.min()) / temperature
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    mean = np.dot(p, e)
    var = np.dot(p, (e - mean) ** 2)
    entropy = -np.dot(p, np.log(np.maximum(p, np.finfo(float).tiny)))
    heat_capacity = var / temperature**2
    return p, entropy, mean, var, heat_capacity


def response_curve(energies: np.ndarray, hs: np.ndarray) -> np.ndarray:
    e = np.asarray(energies, dtype=float)
    H = np.asarray(hs, dtype=float)[:, None]
    logits = -e[None, :] / H
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    mean = p @ e
    var = np.sum(p * (e[None, :] - mean[:, None]) ** 2, axis=1)
    return var / hs**2


def select_global_response_h(V: np.ndarray, h_grid: np.ndarray = H_GRID) -> float:
    Cmat = np.vstack([response_curve(V[ell], h_grid) for ell in range(V.shape[0])])
    return float(h_grid[np.argmax(Cmat.mean(axis=0))])


def select_cluster_response_h(V: np.ndarray, h_grid: np.ndarray = H_GRID) -> np.ndarray:
    return np.asarray([h_grid[np.argmax(response_curve(V[ell], h_grid))] for ell in range(V.shape[0])])


def softmax_feature_weights(V: np.ndarray, h) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    h = np.asarray(h, dtype=float)
    if h.ndim == 0:
        h = np.full(V.shape[0], float(h))
    logits = -V / h[:, None]
    logits -= logits.max(axis=1, keepdims=True)
    W = np.exp(logits)
    W /= W.sum(axis=1, keepdims=True)
    return W


def make_structured_synthetic(seed: int = SEED):
    """Three clusters with 2, 4, 6 relevant coordinates and distinct contrasts."""
    rng = np.random.default_rng(seed)
    n_per, D, K = 240, 12, 3
    relevant = [list(range(0, 2)), list(range(2, 6)), list(range(6, 12))]
    relevant_scales = [0.18, 0.38, 0.65]
    irrelevant_scales = [1.45, 1.65, 1.90]
    center_amplitudes = [2.40, 1.80, 1.40]
    patterns = [
        np.array([-1.0, 1.0]),
        np.array([1.0, -1.0, 1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
    ]
    Xs, ys = [], []
    for ell in range(K):
        Z = rng.normal(size=(n_per, D))
        Z -= Z.mean(axis=0)
        Z /= np.sqrt(np.mean(Z**2, axis=0))
        scales = np.full(D, irrelevant_scales[ell])
        scales[relevant[ell]] = relevant_scales[ell]
        center = np.zeros(D)
        center[relevant[ell]] = center_amplitudes[ell] * patterns[ell]
        Xs.append(center + Z * scales)
        ys.append(np.full(n_per, ell, dtype=int))
    return np.vstack(Xs), np.concatenate(ys), relevant


def dispersions_from_labels(X: np.ndarray, labels: np.ndarray, K: int):
    D = X.shape[1]
    centroids = np.zeros((K, D))
    V = np.zeros((K, D))
    for ell in range(K):
        Z = X[labels == ell]
        if len(Z) == 0:
            raise RuntimeError("Empty cluster encountered in LAC iteration.")
        centroids[ell] = Z.mean(axis=0)
        V[ell] = np.mean((Z - centroids[ell]) ** 2, axis=0)
    return centroids, V


def assign_weighted(X: np.ndarray, centroids: np.ndarray, W: np.ndarray) -> np.ndarray:
    d = np.stack([
        np.sum((X - centroids[ell]) ** 2 * W[ell], axis=1)
        for ell in range(len(centroids))
    ], axis=1)
    return np.argmin(d, axis=1)


def kmeans_init(X: np.ndarray, K: int, seed: int):
    return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(X)


def _iterate_with_temperature_rule(X, K, init_labels, temperature_rule, max_iter=MAX_ITER):
    """Generic LAC loop with explicit cycle detection.

    temperature_rule(V, iteration) returns a scalar or length-K vector.
    """
    labels = init_labels.copy()
    seen = {labels.tobytes(): 0}
    h_path = []
    for iteration in range(1, max_iter + 1):
        centroids, V = dispersions_from_labels(X, labels, K)
        h = temperature_rule(V, iteration)
        h_path.append(np.asarray(h, dtype=float).copy())
        W = softmax_feature_weights(V, h)
        new_labels = assign_weighted(X, centroids, W)
        if np.array_equal(new_labels, labels):
            return new_labels, h, iteration, "converged", 0, h_path
        key = new_labels.tobytes()
        if key in seen:
            period = iteration - seen[key]
            return new_labels, h, iteration, "cycle", period, h_path
        seen[key] = iteration
        labels = new_labels
    return labels, h, max_iter, "max_iter", 0, h_path


def alternating_lac_fixed_h(X, K, h, init_labels, max_iter=MAX_ITER):
    return _iterate_with_temperature_rule(X, K, init_labels, lambda V, it: float(h), max_iter)


def alternating_lac_meanV(X, K, init_labels, max_iter=MAX_ITER):
    return _iterate_with_temperature_rule(X, K, init_labels, lambda V, it: float(np.mean(V)), max_iter)


def alternating_lac_response_adaptive(X, K, init_labels, h_grid=H_GRID, max_iter=MAX_ITER):
    return _iterate_with_temperature_rule(X, K, init_labels, lambda V, it: select_global_response_h(V, h_grid), max_iter)


def alternating_lac_response_clusterwise(X, K, init_labels, h_grid=H_GRID, max_iter=MAX_ITER):
    return _iterate_with_temperature_rule(X, K, init_labels, lambda V, it: select_cluster_response_h(V, h_grid), max_iter)


def two_pass_response_lac(X, K, init_labels, h_grid=H_GRID, max_iter=MAX_ITER):
    """Select one global h from the initial partition, then freeze it."""
    _, V0 = dispersions_from_labels(X, init_labels, K)
    h = select_global_response_h(V0, h_grid)
    labels, _, iterations, status, cycle_period, h_path = alternating_lac_fixed_h(
        X, K, h, init_labels, max_iter=max_iter
    )
    return labels, h, iterations, status, cycle_period, h_path


def oracle_fixed_h(X, y, K, init_labels, h_grid=H_GRID):
    scores = np.empty(len(h_grid))
    for q, h in enumerate(h_grid):
        labels, *_ = alternating_lac_fixed_h(X, K, float(h), init_labels)
        scores[q] = adjusted_rand_score(y, labels)
    idx = int(np.argmax(scores))
    return float(h_grid[idx]), float(scores[idx]), scores


def plateau_component(h_grid, scores, delta=0.005):
    """Connected grid component containing the oracle maximum above max-delta."""
    scores = np.asarray(scores)
    q = int(np.argmax(scores))
    mask = scores >= scores[q] - delta
    lo = q
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = q
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1
    return float(h_grid[lo]), float(h_grid[hi])


def evaluate_seed(seed: int, standardize: bool = False):
    X, y, relevant = make_structured_synthetic(seed)
    if standardize:
        X = StandardScaler().fit_transform(X)
    K = K_SYN
    init = kmeans_init(X, K, seed)
    out = {"seed": seed, "standardized": int(standardize)}
    out["kmeans"] = adjusted_rand_score(y, init)

    labels, _, it, status, period, _ = alternating_lac_fixed_h(X, K, 1.0, init)
    out["fixed_h1"] = adjusted_rand_score(y, labels)
    out["fixed_h1_iter"] = it
    out["fixed_h1_status"] = status

    labels, hmean, it, status, period, _ = alternating_lac_meanV(X, K, init)
    out["meanV"] = adjusted_rand_score(y, labels)
    out["meanV_h"] = float(hmean)

    labels, hadapt, it, status, period, hpath = alternating_lac_response_adaptive(X, K, init)
    out["adaptive_response"] = adjusted_rand_score(y, labels)
    out["adaptive_h_final"] = float(hadapt)
    out["adaptive_iter"] = it
    out["adaptive_status"] = status
    out["adaptive_cycle_period"] = period
    out["adaptive_h_unique"] = len(np.unique(np.asarray([float(np.asarray(h)) for h in hpath])))

    labels, hfrozen, it, status, period, _ = two_pass_response_lac(X, K, init)
    out["frozen_response"] = adjusted_rand_score(y, labels)
    out["frozen_h"] = float(hfrozen)
    out["frozen_iter"] = it
    out["frozen_status"] = status
    out["frozen_cycle_period"] = period

    labels, hvec, it, status, period, _ = alternating_lac_response_clusterwise(X, K, init)
    out["clusterwise_response"] = adjusted_rand_score(y, labels)
    out["clusterwise_h"] = ";".join(f"{x:.12g}" for x in np.asarray(hvec))
    out["clusterwise_iter"] = it
    out["clusterwise_status"] = status
    out["clusterwise_cycle_period"] = period

    h_oracle, ari_oracle, curve = oracle_fixed_h(X, y, K, init)
    out["oracle_grid"] = ari_oracle
    out["oracle_h"] = h_oracle
    lo, hi = plateau_component(H_GRID, curve, delta=0.005)
    out["plateau_h_low"] = lo
    out["plateau_h_high"] = hi
    out["plateau_factor"] = hi / lo
    out["adaptive_in_plateau"] = int(lo <= out["adaptive_h_final"] <= hi)
    out["frozen_in_plateau"] = int(lo <= out["frozen_h"] <= hi)
    if hi > lo:
        out["frozen_plateau_log_position"] = (np.log(out["frozen_h"]) - np.log(lo)) / (np.log(hi) - np.log(lo))
    else:
        out["frozen_plateau_log_position"] = np.nan
    out["curve"] = curve
    return out


def _evaluate_seed_job(args):
    return evaluate_seed(*args)


def multiseed_lac_validation():
    jobs = [(SEED + s, st) for st in (False, True) for s in range(N_SEEDS)]
    rows = [_evaluate_seed_job(job) for job in jobs]
    raw = [r for r in rows if not r["standardized"]]
    std = [r for r in rows if r["standardized"]]

    # Representative fixed-spectrum curves are analytical from the prescribed data design.
    X0, y0, relevant0 = make_structured_synthetic(SEED)
    _, V0 = dispersions_from_labels(X0, y0, K_SYN)
    h_dense = np.geomspace(0.015, 8.0, 1000)
    C0 = np.vstack([response_curve(V0[ell], h_dense) for ell in range(K_SYN)])
    D0 = np.zeros_like(C0)
    for ell in range(K_SYN):
        for q, h in enumerate(h_dense):
            _, S, _, _, _ = canonical_stats(V0[ell], h)
            D0[ell, q] = np.exp(S)
    hstar0 = np.asarray([h_dense[np.argmax(C0[ell])] for ell in range(K_SYN)])

    curves = np.vstack([r["curve"] for r in raw])
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0, ddof=1)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.8), sharex=False)
    for ell in range(K_SYN):
        axes[0].plot(h_dense, D0[ell], label=f"cluster {ell+1}: {len(relevant0[ell])} relevant")
    axes[0].set_xscale("log")
    axes[0].set_ylabel(r"effective features $D_{\mathrm{eff},\ell}$")
    axes[0].legend(frameon=False)

    for ell in range(K_SYN):
        axes[1].plot(h_dense, C0[ell], label=rf"cluster {ell+1}, $h^*={hstar0[ell]:.3f}$")
    axes[1].set_xscale("log")
    axes[1].set_ylabel(r"response $C_\ell^{(F)}(h)$")
    axes[1].legend(frameon=False)

    axes[2].plot(H_GRID, mean_curve, label="fixed-$h$ LAC: mean ARI")
    axes[2].fill_between(H_GRID, mean_curve - std_curve, mean_curve + std_curve, alpha=0.18,
                         label=r"$\pm1$ SD across 30 realizations")
    for key, ls, label in [
        ("adaptive_response", ":", "adaptive response"),
        ("frozen_response", "--", "two-pass frozen response"),
    ]:
        vals = np.asarray([r[key] for r in raw])
        axes[2].axhline(vals.mean(), linestyle=ls, linewidth=1.0,
                        label=rf"{label}: {vals.mean():.3f}$\pm${vals.std(ddof=1):.3f}")
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"global feature temperature $h$")
    axes[2].set_ylabel("adjusted Rand index")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "lac_multiseed_validation")
    plt.close(fig)

    # Save compact rows without full curves.
    fields = [k for k in raw[0].keys() if k != "curve"]
    with open(ASSET_DIR / "lac_multiseed_results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in raw + std:
            wr.writerow({k: csv_value(r[k]) for k in fields})

    conv_fields = ["seed", "standardized", "adaptive_iter", "adaptive_status", "adaptive_cycle_period",
                   "adaptive_h_unique", "frozen_iter", "frozen_status", "frozen_cycle_period"]
    with open(ASSET_DIR / "lac_convergence_results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=conv_fields)
        wr.writeheader()
        for r in raw + std:
            wr.writerow({k: csv_value(r[k]) for k in conv_fields})

    return raw, std, hstar0


def real_dataset_benchmarks():
    datasets = [
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
        ("Breast Cancer Wisconsin Diagnostic", *load_breast_cancer(return_X_y=True)),
    ]
    rows = []
    for name, X, y in datasets:
        X = StandardScaler().fit_transform(X)
        K = len(np.unique(y))
        for s in range(N_SEEDS):
            seed = SEED + s
            init = kmeans_init(X, K, seed)
            row = {"dataset": name, "seed": seed, "n": len(X), "D": X.shape[1], "K": K}
            row["kmeans"] = adjusted_rand_score(y, init)
            labels, *_ = alternating_lac_fixed_h(X, K, 1.0, init)
            row["fixed_h1"] = adjusted_rand_score(y, labels)
            labels, *_ = alternating_lac_meanV(X, K, init)
            row["meanV"] = adjusted_rand_score(y, labels)
            labels, *_ = alternating_lac_response_adaptive(X, K, init)
            row["adaptive_response"] = adjusted_rand_score(y, labels)
            labels, *_ = two_pass_response_lac(X, K, init)
            row["frozen_response"] = adjusted_rand_score(y, labels)
            labels, *_ = alternating_lac_response_clusterwise(X, K, init)
            row["clusterwise_response"] = adjusted_rand_score(y, labels)
            rows.append(row)
    with open(ASSET_DIR / "lac_real_benchmarks.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows({k: csv_value(v) for k, v in r.items()} for r in rows)
    return rows


def deterministic_annealing_diagnostic(X: np.ndarray, seed: int = SEED):
    rng = np.random.default_rng(seed)
    K = 3
    mu = X.mean(axis=0)
    scale = np.std(X, axis=0).mean()
    Z = np.tile(mu, (K, 1))
    temperatures = np.geomspace(80.0, 0.05, 400)
    order_parameter = np.zeros(len(temperatures))
    for q, T in enumerate(temperatures):
        Z = Z + 1e-4 * scale * rng.normal(size=Z.shape)
        for _ in range(1000):
            d2 = np.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=2)
            logits = -d2 / T
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
            Z_new = (P.T @ X) / P.sum(axis=0)[:, None]
            if np.max(np.linalg.norm(Z_new - Z, axis=1)) < 1e-10:
                Z = Z_new; break
            Z = Z_new
        order_parameter[q] = np.sqrt(np.mean(np.sum((Z - Z.mean(axis=0)) ** 2, axis=1)))
    covariance = np.cov(X, rowvar=False, bias=True)
    critical_temperature = 2.0 * np.linalg.eigvalsh(covariance).max()
    split_idx = np.flatnonzero(order_parameter > 1e-3)
    observed_split = temperatures[split_idx[0]] if len(split_idx) else np.nan
    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    ax.plot(temperatures, order_parameter, label="codevector-separation order parameter")
    ax.axvline(critical_temperature, linestyle="--", linewidth=1.0,
               label=rf"$T_c=2\lambda_{{\max}}={critical_temperature:.3f}$")
    ax.invert_xaxis(); ax.set_xscale("log")
    ax.set_ylabel("assignment order parameter"); ax.set_xlabel("deterministic-annealing temperature")
    ax.legend(frameon=False); fig.tight_layout()
    save_figure(fig, "annealing_bifurcation"); plt.close(fig)
    return critical_temperature, observed_split


def entropy_for_beta(energies: np.ndarray, beta: float):
    e = np.asarray(energies, dtype=float)
    logits = -beta * e; logits -= logits.max()
    p = np.exp(logits); p /= p.sum()
    entropy = -np.dot(p, np.log(np.maximum(p, np.finfo(float).tiny)))
    return entropy, p


def beta_for_perplexity(energies: np.ndarray, perplexity: float, tol: float = 1e-12):
    target = np.log(perplexity); lo, hi = 1e-12, 1e12; beta = 1.0
    for _ in range(200):
        entropy, p = entropy_for_beta(energies, beta)
        if abs(entropy - target) < tol:
            return beta, p
        if entropy > target:
            lo = beta; beta = np.sqrt(beta * hi) if hi < 1e11 else 2.0 * beta
        else:
            hi = beta; beta = np.sqrt(beta * lo) if lo > 1e-11 else 0.5 * beta
    return beta, p


def local_temperatures(X: np.ndarray, perplexity: float):
    D2 = pairwise_distances(X, metric="sqeuclidean")
    D = np.sqrt(np.maximum(D2, 0.0)); n = len(X)
    T = np.zeros(n); C = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i; energies = D2[i, mask]
        beta, p = beta_for_perplexity(energies, perplexity)
        T[i] = 1.0 / beta
        mean = np.dot(p, energies); var = np.dot(p, (energies - mean) ** 2)
        C[i] = var / T[i] ** 2
    return T, C, D


def density_proxy(D: np.ndarray, k: int):
    n = len(D); rho = np.zeros(n)
    for i in range(n):
        d = np.sort(D[i, np.arange(n) != i]); rho[i] = 1.0 / np.mean(d[:k])
    return rho


def tls_slope(x: np.ndarray, y: np.ndarray):
    centered = np.column_stack([x - x.mean(), y - y.mean()])
    cov = centered.T @ centered / (len(x) - 1)
    _, vecs = np.linalg.eigh(cov); v = vecs[:, -1]
    return v[1] / v[0]


def bootstrap_slope_ci(x, y, slope_fn, seed=SEED, B=3000):
    rng = np.random.default_rng(seed); n = len(x); slopes = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n); slopes[b] = slope_fn(x[idx], y[idx])
    return np.quantile(slopes, [0.025, 0.975])


def controlled_scaling_diagnostic(perplexity: float = 20.0):
    rng = np.random.default_rng(2026); base = rng.uniform(-0.5, 0.5, size=(240, 5))
    scales = np.array([0.6, 0.8, 1.0, 1.3, 1.7, 2.2]); med_T, med_rho = [], []
    for lam in scales:
        T, _, D = local_temperatures(lam * base, perplexity); rho = density_proxy(D, 20)
        med_T.append(np.median(T)); med_rho.append(np.median(rho))
    med_T = np.asarray(med_T); med_rho = np.asarray(med_rho)
    x = np.log(med_rho); y = np.log(med_T); exact_slope = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(314159); rep = 80
    xrep = np.repeat(x, rep); yrep = np.repeat(y, rep); noise_sd = 0.18
    xn = xrep + rng.normal(scale=noise_sd, size=len(xrep)); yn = yrep + rng.normal(scale=noise_sd, size=len(yrep))
    return med_rho, med_T, exact_slope, xn, yn, np.polyfit(xn, yn, 1)[0], tls_slope(xn, yn)


def wine_neighbor_diagnostics(perplexity: float = 20.0):
    X, _ = load_wine(return_X_y=True); X = StandardScaler().fit_transform(X)
    T, C, D = local_temperatures(X, perplexity)
    rho10, rho20, rho30 = density_proxy(D, 10), density_proxy(D, 20), density_proxy(D, 30)
    x10, y = np.log(rho10), np.log(T)
    rho_s, p_s = spearmanr(x10, y)
    ols10 = np.polyfit(x10, y, 1)[0]
    ci10 = bootstrap_slope_ci(x10, y, lambda a, b: np.polyfit(a, b, 1)[0])
    tls10 = tls_slope(x10, y); tls_ci10 = bootstrap_slope_ci(x10, y, tls_slope, seed=SEED+1)
    ols20 = np.polyfit(np.log(rho20), y, 1)[0]; tls20 = tls_slope(np.log(rho20), y)
    ols30 = np.polyfit(np.log(rho30), y, 1)[0]; tls30 = tls_slope(np.log(rho30), y)
    control_rho, control_T, control_slope, xn, yn, noisy_ols, noisy_tls = controlled_scaling_diagnostic(perplexity)
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 9.0))
    cfit = np.polyfit(np.log(control_rho), np.log(control_T), 1); xx = np.linspace(np.log(control_rho).min(), np.log(control_rho).max(), 200)
    axes[0].scatter(control_rho, control_T, s=30); axes[0].plot(np.exp(xx), np.exp(cfit[0]*xx+cfit[1]), label=rf"exact-copy slope $={control_slope:.3f}$")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_ylabel(r"median $T_i^{(N)}$"); axes[0].set_xlabel(r"median density proxy $\rho_i^{(20)}$"); axes[0].legend(frameon=False)
    axes[1].scatter(np.exp(xn), np.exp(yn), s=8, alpha=0.35); axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("controlled noisy density proxy"); axes[1].set_ylabel("controlled noisy temperature"); axes[1].text(0.03,0.05,rf"OLS={noisy_ols:.3f}; TLS={noisy_tls:.3f}",transform=axes[1].transAxes)
    axes[2].scatter(rho10, T, s=18, alpha=0.75); fit=np.polyfit(x10,y,1); xx=np.linspace(x10.min(),x10.max(),200); axes[2].plot(np.exp(xx),np.exp(fit[0]*xx+fit[1]),label=rf"Wine OLS slope $={ols10:.3f}$")
    axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlabel(r"Wine density proxy $\rho_i^{(10)}$"); axes[2].set_ylabel(r"local temperature $T_i^{(N)}$"); axes[2].legend(frameon=False)
    fig.tight_layout(); save_figure(fig, "sne_density_scaling"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.1,4.4)); ax.hist(C,bins=22,edgecolor="black",linewidth=0.45)
    q50,q90,q95=np.quantile(C,[0.5,0.9,0.95]); ax.axvline(q50,linewidth=1.0,label=rf"median $={q50:.3f}$"); ax.axvline(q95,linewidth=1.0,linestyle="--",label=rf"95th percentile $={q95:.3f}$")
    ax.set_xlabel(r"neighbor response $C_i^{(N)}$"); ax.set_ylabel("anchors"); ax.legend(frameon=False); fig.tight_layout(); save_figure(fig, "sne_neighbor_response"); plt.close(fig)
    return dict(spearman_rho=rho_s,spearman_p=p_s,ols10=ols10,ols10_ci_low=ci10[0],ols10_ci_high=ci10[1],tls10=tls10,tls10_ci_low=tls_ci10[0],tls10_ci_high=tls_ci10[1],ols20=ols20,tls20=tls20,ols30=ols30,tls30=tls30,control_slope=control_slope,noisy_ols=noisy_ols,noisy_tls=noisy_tls,response_median=q50,response_q90=q90,response_q95=q95)


def paired_bootstrap_mean_ci(a, b, seed=SEED, B=10000):
    a=np.asarray(a); b=np.asarray(b); d=a-b; rng=np.random.default_rng(seed); n=len(d); means=np.empty(B)
    for q in range(B):
        idx=rng.integers(0,n,n); means[q]=d[idx].mean()
    return d.mean(), np.quantile(means,[0.025,0.975])


def summarize(raw, std, real_rows, hstar0, critical_T, observed_split, wine):
    methods=["kmeans","fixed_h1","meanV","adaptive_response","frozen_response","clusterwise_response","oracle_grid"]
    def arr(rows,key): return np.asarray([r[key] for r in rows],dtype=float)
    summary={}
    for prefix, rows in [("raw",raw),("std",std)]:
        for m in methods:
            a=arr(rows,m); summary[f"lac_{prefix}_{m}_mean"]=a.mean(); summary[f"lac_{prefix}_{m}_sd"]=a.std(ddof=1)
        summary[f"lac_{prefix}_adaptive_iter_mean"]=arr(rows,"adaptive_iter").mean()
        summary[f"lac_{prefix}_adaptive_iter_max"]=arr(rows,"adaptive_iter").max()
        summary[f"lac_{prefix}_adaptive_cycle_fraction"]=np.mean([r["adaptive_status"]=="cycle" for r in rows])
        summary[f"lac_{prefix}_adaptive_maxiter_fraction"]=np.mean([r["adaptive_status"]=="max_iter" for r in rows])
        summary[f"lac_{prefix}_frozen_iter_mean"]=arr(rows,"frozen_iter").mean()
        summary[f"lac_{prefix}_frozen_cycle_fraction"]=np.mean([r["frozen_status"]=="cycle" for r in rows])
        summary[f"lac_{prefix}_plateau_factor_median"]=np.median(arr(rows,"plateau_factor"))
        summary[f"lac_{prefix}_adaptive_in_plateau_fraction"]=arr(rows,"adaptive_in_plateau").mean()
        summary[f"lac_{prefix}_frozen_in_plateau_fraction"]=arr(rows,"frozen_in_plateau").mean()
    for a,b,name in [(arr(raw,"adaptive_response"),arr(raw,"frozen_response"),"adaptive_minus_frozen"),(arr(raw,"adaptive_response"),arr(raw,"oracle_grid"),"adaptive_minus_oracle"),(arr(raw,"frozen_response"),arr(raw,"oracle_grid"),"frozen_minus_oracle"),(arr(raw,"adaptive_response"),arr(raw,"fixed_h1"),"adaptive_minus_h1")]:
        d,ci=paired_bootstrap_mean_ci(a,b); summary[f"lac_{name}_mean"]=d; summary[f"lac_{name}_ci_low"]=ci[0]; summary[f"lac_{name}_ci_high"]=ci[1]
    for ell,h in enumerate(hstar0,1): summary[f"representative_hstar_cluster_{ell}"]=h
    summary["deterministic_annealing_Tc_theory"]=critical_T; summary["deterministic_annealing_first_observed_split"]=observed_split
    for k,v in wine.items(): summary[f"wine_{k}"]=v
    # Representative dispersion ratios for the explanatory text.
    X0,y0,rel=make_structured_synthetic(SEED); _,V0=dispersions_from_labels(X0,y0,K_SYN)
    ratios=[]
    for ell in range(K_SYN):
        mask=np.zeros(X0.shape[1],dtype=bool); mask[rel[ell]]=True
        ratios.append(V0[ell,~mask].mean()/V0[ell,mask].mean())
    summary["synthetic_irrelevant_to_relevant_variance_ratio_min"]=min(ratios); summary["synthetic_irrelevant_to_relevant_variance_ratio_max"]=max(ratios)
    # Real benchmarks summary.
    for ds in sorted(set(r["dataset"] for r in real_rows)):
        rs=[r for r in real_rows if r["dataset"]==ds]
        safe=ds.lower().replace(" ","_")
        for m in ["kmeans","fixed_h1","meanV","adaptive_response","frozen_response","clusterwise_response"]:
            a=np.asarray([r[m] for r in rs]); summary[f"real_{safe}_{m}_mean"]=a.mean(); summary[f"real_{safe}_{m}_sd"]=a.std(ddof=1)
    with open(ASSET_DIR / "numerical_summary.csv","w",newline="") as fh:
        wr=csv.writer(fh); wr.writerow(["metric","value"])
        for k in sorted(summary): wr.writerow([k,f"{summary[k]:.12g}"])
    return summary


def main():
    raw,std,hstar0=multiseed_lac_validation()
    real_rows=real_dataset_benchmarks()
    X0,_,_=make_structured_synthetic(SEED)
    critical_T,observed_split=deterministic_annealing_diagnostic(X0)
    wine=wine_neighbor_diagnostics(20.0)
    summary=summarize(raw,std,real_rows,hstar0,critical_T,observed_split,wine)
    print("Synthetic raw ARI means:")
    for m in ["kmeans","fixed_h1","meanV","adaptive_response","frozen_response","clusterwise_response","oracle_grid"]:
        a=np.asarray([r[m] for r in raw]); print(m,a.mean(),a.std(ddof=1))
    print("Adaptive statuses", {s:sum(r['adaptive_status']==s for r in raw) for s in ['converged','cycle','max_iter']})
    print("Frozen statuses", {s:sum(r['frozen_status']==s for r in raw) for s in ['converged','cycle','max_iter']})
    print("Median plateau factor", summary["lac_raw_plateau_factor_median"])
    print("Real benchmarks")
    for ds in sorted(set(r["dataset"] for r in real_rows)):
        rs=[r for r in real_rows if r["dataset"]==ds]
        print(ds,{m:np.mean([r[m] for r in rs]) for m in ['kmeans','fixed_h1','meanV','adaptive_response','frozen_response','clusterwise_response']})
    print("Assets", ASSET_DIR)
    print("Figures", FIGURE_DIR)


if __name__ == "__main__":
    main()
