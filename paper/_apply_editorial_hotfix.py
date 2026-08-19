"""Materialize the final SNE validation and remove duplicated revision text."""
from __future__ import annotations

import csv
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent
MAIN = PAPER / "main.tex"
CSV_PATH = PAPER / "experiments" / "assets" / "sne_perplexity_validation.csv"

with CSV_PATH.open(encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

summaries = {}
for perplexity in (5.0, 20.0, 30.0, 50.0):
    subset = [r for r in rows if float(r["perplexity"]) == perplexity]
    response = sorted(float(r["response"]) for r in subset)
    tolerance = sorted(float(r["predicted_sigma_tolerance"]) for r in subset)
    actual = sorted(float(r["actual_perplexity_relative_error"]) for r in subset)
    n = len(subset)
    def median(values):
        mid = n // 2
        return values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])
    summaries[perplexity] = (
        median(response),
        median(tolerance),
        median(actual),
    )

text = MAIN.read_text(encoding="utf-8")

# Correct the t-SNE row: its Gibbs representation has unit temperature after the
# scale has been absorbed into the logarithmic pair energy.
text = text.replace(
    "t-SNE map & Embedded pair $(i,j)$ & Logarithmic or generalized Student pair energy & Kernel parameterization & Joint pair probability $q_{ij}$\\\\",
    "t-SNE map & Embedded pair $(i,j)$ & Logarithmic or generalized Student pair energy & Unit scale by construction & Joint pair probability $q_{ij}$\\\\",
)

# Remove accidental repeated copies of the same Schottky conclusion paragraph.
paragraph = (
    "That distinction leads to a specific spectral mechanism. For a two-level LAC "
    "dispersion spectrum, the response is a finite Schottky anomaly whose peak is fixed "
    "by the dispersion gap and the degeneracy ratio. The synthetic experiment confirms "
    "the analytic peak on the true partition and quantifies its displacement when the "
    "spectrum is estimated from the initial $k$-means partition. Broad or rotated spectra "
    "provide corresponding failure modes rather than unexplained exceptions."
)
pattern = re.escape(paragraph) + r"(?:\s+" + re.escape(paragraph) + r")+"
text = re.sub(pattern, paragraph, text, count=1)

if "tab:sne_tolerance_validation" not in text:
    table_rows = []
    for perplexity in (5.0, 20.0, 30.0, 50.0):
        response, tolerance, actual = summaries[perplexity]
        table_rows.append(
            f"{int(perplexity)} & {response:.3f} & {100*tolerance:.3f} & {100*actual:.3f}\\\\"
        )
    block = r'''

The first-order rule can also be tested directly instead of being inferred only from the response distribution. For every Wine anchor and each target perplexity in $\{5,20,30,50\}$, the bandwidth is perturbed by the anchor-specific magnitude $\varepsilon/(2C_i^{(N)})$ with $\varepsilon=0.01$, after which the perplexity is recomputed from the unchanged neighbor-energy spectrum. Tab.\,\ref{tab:sne_tolerance_validation} reports the median predicted bandwidth tolerance and the median perplexity error actually produced by that perturbation. Across the four perplexities, the achieved error remains close to the prescribed 1\% level. This experiment therefore validates Eq.\,\eqref{eq:perplexity_sigma_tolerance} as a local first-order sensitivity estimate over the tested range; it does not imply that the tolerance materially accelerates a modern root solver.

\begin{table}[htbp]
\centering
\caption{Direct validation of the first-order SNE bandwidth-tolerance rule on standardized Wine data. Tolerances and achieved errors are medians across the 178 anchors.}
\label{tab:sne_tolerance_validation}
\begin{tabular}{cccc}
\toprule
\textbf{Perplexity} & \textbf{Median $C_i^{(N)}$} & \textbf{Predicted $|\delta\sigma|/\sigma$ (\%)} & \textbf{Actual perplexity error (\%)}\\
\midrule
''' + "\n".join(table_rows) + r'''
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.88\linewidth]{sne_perplexity_validation.pdf}
\caption{Neighbor response and direct tolerance validation across target perplexities on Wine. Left: median and 95th-percentile response. Right: median bandwidth tolerance predicted for a 1\% perplexity error and the actual median error obtained after applying that perturbation.}
\label{fig:sne_perplexity_validation}
\end{figure}
'''
    anchor = "\\label{fig:sne_neighbor_response}\n\\end{figure}"
    if anchor not in text:
        raise RuntimeError("Could not locate SNE response figure insertion point")
    text = text.replace(anchor, anchor + block, 1)

text = text.replace(
    "In SNE, the response yields a directly usable bandwidth tolerance, and the density-scaling experiments distinguish the exact self-similar law from regression and geometric effects that are present in real data.",
    "In SNE, the response yields an anchor-specific bandwidth tolerance; direct perturbation tests at perplexities 5, 20, 30, and 50 reproduce the prescribed 1\\% perplexity error to first order, while the density-scaling experiments distinguish the exact self-similar law from regression and geometric effects that are present in real data.",
)

MAIN.write_text(text, encoding="utf-8")
