"""Patch the already-materialized Schottky paragraph/table from regenerated CSV."""
from __future__ import annotations

import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "main.tex"
CSV_PATH = HERE / "assets" / "lac_schottky_diagnostics.csv"

with CSV_PATH.open(encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

by_cluster: dict[int, dict[str, dict[str, str]]] = {}
for row in rows:
    by_cluster.setdefault(int(row["cluster"]), {})[row["spectrum"]] = row

true_errors = [
    abs(float(by_cluster[k]["true_partition"]["relative_gap_hstar_error"]))
    for k in (1, 2, 3)
]
km_errors = [
    abs(float(by_cluster[k]["kmeans_partition"]["relative_gap_hstar_error"]))
    for k in (1, 2, 3)
]
max_true = 100.0 * max(true_errors)
max_km = 100.0 * max(km_errors)
global_h = float(by_cluster[1]["kmeans_partition"]["kmeans_global_mean_hstar"])
low_count = sum(
    int(float(by_cluster[k]["kmeans_partition"]["low_temperature_peak"]))
    for k in (1, 2, 3)
)

text = MAIN.read_text(encoding="utf-8")

text = re.sub(
    r"On the structured synthetic design, the resulting analytic peak predicts the true-partition numerical response maxima with at most [0-9.]+\\% relative error; after replacing the true partition by the initial \$k\$-means partition, the maximum relative displacement is [0-9.]+\\%\.",
    (
        "On the structured synthetic design, the resulting analytic peak predicts "
        f"the true-partition response peak with at most {max_true:.2f}\\% relative "
        "error. In the initial $k$-means spectra, the gap-associated local peak "
        f"remains within {max_km:.2f}\\% of the analytic scale, although within-band "
        f"dispersion produces additional low-temperature maxima in {low_count} clusters."
    ),
    text,
    count=1,
)

paragraph = (
    "The analytic peak agrees with the gap-associated true-partition maximum to "
    f"within {max_true:.2f}\\% in all three clusters. After the true labels are "
    "replaced by the initial $k$-means partition, the high-temperature local maximum "
    "associated with the prescribed relevant--irrelevant gap remains within "
    f"{max_km:.2f}\\% of the analytic scale. The broadened spectra of clusters 2 and 3 "
    "also develop stronger low-temperature maxima caused by dispersion within the "
    "nominal bands. Thus the estimated cluster spectrum is genuinely multiscale rather "
    "than an exactly degenerate two-level system. The operational rule averages cluster "
    "responses before maximization; for seed 1729 that global mean response selects "
    f"$h={global_h:.3f}$, on the same scale as the gap-associated branch rather than "
    "the individual low-temperature anomalies."
)

text = re.sub(
    r"The analytic peak agrees with the true-partition numerical maximum.*?exactly degenerate two-level system\.",
    lambda _m: paragraph,
    text,
    count=1,
    flags=re.S,
)

row_lines = []
for k in (1, 2, 3):
    p = by_cluster[k]["prescribed"]
    t = by_cluster[k]["true_partition"]
    m = by_cluster[k]["kmeans_partition"]
    row_lines.append(
        f"{k} & {int(float(p['g_relevant']))}/{int(float(p['g_irrelevant']))} & "
        f"{float(p['delta_prescribed']):.4f} & {float(p['hstar_analytic']):.4f} & "
        f"{float(t['hstar_gap_numeric']):.4f} & {float(m['hstar_gap_numeric']):.4f} & "
        f"{float(m['hstar_global_numeric']):.4f}\\\\"
    )

table = r"""
\begin{table}[htbp]
\centering
\small
\caption{Two-level Schottky prediction and numerical LAC response scales for seed 1729. $g_0/g_1$ denotes the number of prescribed low/high-dispersion feature states. The $k$-means gap column reports the local maximum on the analytic gap scale; the final column reports the unrestricted clusterwise maximum and exposes additional low-temperature structure.}
\label{tab:schottky_validation}
\begin{tabular}{ccccccc}
\toprule
\textbf{Cluster} & $g_0/g_1$ & $\Delta$ & $h^*_{\rm Sch}$ & $h^*_{\rm true}$ & $h^*_{k\text{-means},\,gap}$ & $h^*_{k\text{-means},\,global}$\\
\midrule
""" + "\n".join(row_lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

text = re.sub(
    r"\\begin\{table\}\[htbp\]\s*\\centering\s*\\caption\{Two-level Schottky prediction.*?\\label\{tab:schottky_validation\}.*?\\end\{table\}",
    lambda _m: table.strip(),
    text,
    count=1,
    flags=re.S,
)

MAIN.write_text(text, encoding="utf-8")
