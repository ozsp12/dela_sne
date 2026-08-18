"""Apply the scientific major revision after regenerated experiment assets exist."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ASSETS = PAPER / "experiments" / "assets"
MAIN = PAPER / "main.tex"
BIBS = [PAPER / "references.bib", ROOT / "references" / "references.bib"]


def read_csv(name: str) -> list[dict[str, str]]:
    with (ASSETS / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def mean_sd(rows: list[dict[str, str]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def fmt_pm(mean: float, sd: float) -> str:
    return rf"${mean:.3f}\pm{sd:.3f}$"


def append_bibliography() -> None:
    block = r'''

@article{jing2007entropy,
  author  = {Jing, Liping and Ng, Michael K. and Huang, Joshua Zhexue},
  title   = {An Entropy Weighting k-Means Algorithm for Subspace Clustering of High-Dimensional Sparse Data},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {19},
  number  = {8},
  pages   = {1026--1041},
  year    = {2007},
  doi     = {10.1109/TKDE.2007.1048}
}

@inproceedings{tishby1999ib,
  author    = {Tishby, Naftali and Pereira, Fernando C. and Bialek, William},
  title     = {The Information Bottleneck Method},
  booktitle = {Proceedings of the 37th Annual Allerton Conference on Communication, Control, and Computing},
  pages     = {368--377},
  year      = {1999}
}

@article{lee2015multiscale,
  author  = {Lee, John A. and Peluffo-Ord{\'o}{\~n}ez, Diego H. and Verleysen, Michel},
  title   = {Multi-scale Similarities in Stochastic Neighbour Embedding: Reducing Dimensionality while Preserving Both Local and Global Structure},
  journal = {Neurocomputing},
  volume  = {169},
  pages   = {246--261},
  year    = {2015},
  doi     = {10.1016/j.neucom.2014.12.095}
}

@inproceedings{debodt2018perplexityfree,
  author    = {de Bodt, Cyril and Mulders, Dounia and Verleysen, Michel and Lee, John A.},
  title     = {Perplexity-free t-SNE and Twice Student tt-SNE},
  booktitle = {Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  year      = {2018}
}

@misc{cao2017perplexity,
  author       = {Cao, Yanshuai and Wang, Luyu},
  title        = {Automatic Selection of t-SNE Perplexity},
  year         = {2017},
  eprint       = {1708.03229},
  archivePrefix = {arXiv},
  primaryClass = {stat.ML}
}

@article{belkina2019optsne,
  author  = {Belkina, Anna C. and Ciccolella, Christopher O. and Anno, Rina and Halpert, Richard and Spidlen, Josef and Snyder-Cappione, Jennifer E.},
  title   = {Automated Optimized Parameters for T-Distributed Stochastic Neighbor Embedding Improve Visualization and Analysis of Large Datasets},
  journal = {Nature Communications},
  volume  = {10},
  pages   = {5415},
  year    = {2019},
  doi     = {10.1038/s41467-019-13055-y}
}

@article{rousseeuw1987silhouette,
  author  = {Rousseeuw, Peter J.},
  title   = {Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis},
  journal = {Journal of Computational and Applied Mathematics},
  volume  = {20},
  pages   = {53--65},
  year    = {1987},
  doi     = {10.1016/0377-0427(87)90125-7}
}

@inproceedings{benhur2002stability,
  author    = {Ben-Hur, Asa and Elisseeff, Andr{\'e} and Guyon, Isabelle},
  title     = {A Stability Based Method for Discovering Structure in Clustered Data},
  booktitle = {Pacific Symposium on Biocomputing},
  pages     = {6--17},
  year      = {2002},
  doi       = {10.1142/9789812799623_0002}
}
'''
    for path in BIBS:
        text = path.read_text(encoding="utf-8")
        if "jing2007entropy" not in text:
            path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def build_numeric_context() -> dict[str, object]:
    schottky = read_csv("lac_schottky_diagnostics.csv")
    baselines = read_csv("lac_unsupervised_baselines.csv")
    stress = read_csv("lac_stress_tests.csv")
    sne = read_csv("sne_perplexity_validation.csv")

    sch_by_cluster: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in schottky:
        sch_by_cluster[int(row["cluster"])][row["spectrum"]] = row

    baseline_summary: dict[tuple[str, str], tuple[float, float]] = {}
    for dataset in ("Synthetic", "Iris", "Wine", "WDBC"):
        for selector in ("kmeans", "response", "silhouette", "stability", "oracle"):
            subset = [r for r in baselines if r["dataset"] == dataset and r["selector"] == selector]
            if subset:
                baseline_summary[(dataset, selector)] = mean_sd(subset, "ari")

    stress_summary: dict[tuple[str, float, str], tuple[float, float]] = {}
    for factor in {r["factor"] for r in stress}:
        values = sorted({float(r["value"]) for r in stress if r["factor"] == factor})
        for value in values:
            subset = [r for r in stress if r["factor"] == factor and float(r["value"]) == value]
            for method in ("kmeans", "fixed_h1", "response"):
                stress_summary[(factor, value, method)] = mean_sd(subset, method)

    sne_summary: dict[float, dict[str, float]] = {}
    for perplexity in (5.0, 20.0, 30.0, 50.0):
        subset = [r for r in sne if float(r["perplexity"]) == perplexity]
        response = np.asarray([float(r["response"]) for r in subset])
        tolerance = np.asarray([float(r["predicted_sigma_tolerance"]) for r in subset])
        actual = np.asarray([float(r["actual_perplexity_relative_error"]) for r in subset])
        sne_summary[perplexity] = {
            "response_median": float(np.median(response)),
            "response_q95": float(np.quantile(response, 0.95)),
            "tolerance_median": float(np.median(tolerance)),
            "actual_error_median": float(np.median(actual)),
        }

    return {
        "sch": sch_by_cluster,
        "baseline": baseline_summary,
        "stress": stress_summary,
        "sne": sne_summary,
    }


def replace_proposition(text: str) -> str:
    start = text.find(r"\begin{proposition}[State-space specialization of the classical Gibbs variational principle]")
    if start == -1:
        return text
    end = text.find(r"\end{proof}", start)
    if end == -1:
        raise RuntimeError("Could not locate end of classical Gibbs proposition proof")
    end += len(r"\end{proof}")
    replacement = r'''
The common variational specialization needed for the comparison is a direct consequence of the classical Gibbs--Jaynes principle rather than an independent theorem. For a finite state space with energies $E_a$ and $T>0$, let $p_a^\star=\exp(-E_a/T)/Z_T$, where $Z_T=\sum_b\exp(-E_b/T)$. Then
\begin{equation}
    \sum_a p_aE_a+T\sum_a p_a\ln\left(p_a\right)
    =-T\ln\left(Z_T\right)+T\KL\left(p\middle\|p^\star\right),
    \label{eq:common_variational_kl_decomposition}
\end{equation}
so nonnegativity of the Kullback--Leibler divergence gives the Gibbs minimizer Refs.\,\cite{gibbs1902,jaynes1957,kullback1951}. LAC follows by taking the states to be features, $E_f=V_{\ell f}$ and $T=h$; SNE follows by taking the states to be candidate neighbors, $E_j=\|x_i-x_j\|_2^2$ and $T=2\sigma_i^2$. The contribution of the present comparison is therefore the state-space distinction and the response diagnostics, not this classical minimization identity.
'''
    return text[:start] + replacement + text[end:]


def apply_main_revision(ctx: dict[str, object]) -> None:
    sch = ctx["sch"]
    baseline = ctx["baseline"]
    stress = ctx["stress"]
    sne = ctx["sne"]

    true_errors = [abs(float(sch[k]["true_partition"]["relative_hstar_error"])) for k in (1, 2, 3)]
    km_errors = [abs(float(sch[k]["kmeans_partition"]["relative_hstar_error"])) for k in (1, 2, 3)]
    max_true = 100 * max(true_errors)
    max_km = 100 * max(km_errors)

    title = "Schottky Response and Perplexity Tolerance in Locally Adaptive Clustering and Stochastic Neighbor Embedding"
    abstract = rf'''\begin{{abstract}}
The entropy-regularized feature weights of Locally Adaptive Clustering (LAC) define a finite feature ensemble whose response is controlled by gaps in the within-cluster dispersion spectrum. For an ideal two-level spectrum with $g_0$ low-dispersion and $g_1$ high-dispersion features, we derive the Schottky form $C=x^2u/(1+u)^2$, where $x=\Delta/h$ and $u=(g_1/g_0)e^{{-x}}$, and the peak condition $x=2(1+u)/(1-u)$. On the structured synthetic design, the resulting analytic peak predicts the true-partition numerical response maxima with at most {max_true:.2f}\% relative error; after replacing the true partition by the initial $k$-means partition, the maximum relative displacement is {max_km:.2f}\%. Across 30 realizations, a two-pass global response rule retains an adjusted Rand index (ARI) of $0.983\pm0.008$, close to the retrospective fixed-$h$ oracle at $0.988\pm0.006$, while independently selected cluster temperatures remain substantially worse. A separate matched-grid comparison evaluates response selection against silhouette and perturbation-stability selection, and controlled stress tests vary overlap, imbalance, dimensionality, and rotations that destroy axis-aligned feature relevance. For Stochastic Neighbor Embedding (SNE), the canonical response gives a first-order conversion between bandwidth and perplexity errors; direct perturbation tests on Wine at perplexities 5, 20, 30, and 50 quantify the accuracy of that linearization. The normalized-exponential identities are treated as classical prior structure; the new result is the spectral mechanism linking a LAC response peak to a feature-dispersion gap and its empirically testable failure conditions.
\end{{abstract}}'''

    text = MAIN.read_text(encoding="utf-8")
    text = re.sub(r"\\title\{\\textbf\{.*?\}\}", rf"\\title{{\\textbf{{{title}}}}}", text, count=1)
    text = re.sub(r"pdftitle=\{.*?\},", f"pdftitle={{{title}}},", text, count=1)
    text = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", abstract, text, count=1, flags=re.S)

    if "Schottky response" not in text.split(r"\bottomrule", 1)[0]:
        text = text.replace(
            r"$C(T)$ & Fixed-spectrum response $\partial U/\partial T=\operatorname{Var}(E)/T^2$; equal to dimensionless heat capacity in the canonical physical model.\\",
            r"$C(T)$ & Fixed-spectrum response $\partial U/\partial T=\operatorname{Var}(E)/T^2$; equal to dimensionless heat capacity in the canonical physical model.\\" + "\n" +
            r"Schottky response & Finite-system response peak generated by occupation transfer across a separated energy or dispersion gap.\\",
        )

    jing_anchor = "Locally Adaptive Clustering (LAC), proposed by Domeniconi et al., Ref.\\,\\cite{domeniconi2007lac}, belongs naturally to this historical sequence: instead of selecting a single global subspace, it assigns cluster-dependent feature weights and thereby learns a local metric adapted to the dispersion structure of each group."
    jing_para = r'''

A particularly close precedent is the entropy-weighting $k$-means method of Jing, Ng, and Huang, Ref.\,\cite{jing2007entropy}, which also assigns cluster-dependent weights to dimensions and includes the entropy of those weights in the clustering objective. This prior work makes it essential to distinguish the present claim from entropy-weighted feature selection itself. The contribution developed below is instead the finite-spectrum response of the LAC weight distribution, the analytic two-level mechanism for its maximum, and tests of that response as a temperature-selection diagnostic.
'''
    if "jing2007entropy" not in text:
        text = text.replace(jing_anchor, jing_anchor + jing_para)

    ib_anchor = "The resulting connection between Shannon's information measure, Gibbsian statistical mechanics, and constrained probabilistic inference supplies the conceptual basis for the entropy-optimization viewpoint adopted in this article."
    ib_para = r'''

The information-bottleneck construction of Tishby, Pereira, and Bialek, Ref.\,\cite{tishby1999ib}, provides another important variational precedent: it trades compression of one random variable against preservation of information about another. Its state variables and objective differ from LAC and SNE, but it reinforces the broader point that entropy-regularized learning objectives and temperature-like Lagrange multipliers predate the present analysis.
'''
    if "tishby1999ib" not in text:
        text = text.replace(ib_anchor, ib_anchor + ib_para)

    tsne_anchor = "These developments do not alter the standard t-SNE equations studied here, but they clarify the methodological environment in which the algorithm is now interpreted."
    tsne_para = r'''

Perplexity itself has also been targeted directly by alternative formulations. Multi-scale SNE averages neighborhoods across several bandwidth scales rather than committing to one local scale Ref.\,\cite{lee2015multiscale}; perplexity-free t-SNE replaces a single Gaussian-neighborhood scale by heavy-tailed or multi-scale constructions Ref.\,\cite{debodt2018perplexityfree}; Cao and Wang proposed an explicit model-selection criterion for t-SNE perplexity Ref.\,\cite{cao2017perplexity}; and opt-SNE automates optimization parameters from the evolution of the Kullback--Leibler divergence Ref.\,\cite{belkina2019optsne}. The first-order tolerance developed in Sec.\,\ref{subsec:perplexity} is therefore not proposed as a replacement for perplexity selection or perplexity-free embeddings. It is a local sensitivity diagnostic conditional on a target perplexity that has already been chosen.
'''
    if "lee2015multiscale" not in text:
        text = text.replace(tsne_anchor, tsne_anchor + tsne_para)

    text = replace_proposition(text)

    schottky_anchor = "The scale behavior of this diagnostic should also be stated explicitly."
    schottky_theory = r'''
\subsection{Two-level feature spectra and the Schottky response}
\label{subsec:lac_schottky}

The response maximum in Eq.\,\eqref{eq:lac_crossover} has a closed analytic mechanism when the feature-dispersion spectrum separates into two levels. Suppose that $g_0$ feature states have dispersion $V_0$ and $g_1$ states have dispersion $V_1>V_0$, and define the gap $\Delta:=V_1-V_0$. Shifting all energies by $V_0$ leaves the probabilities unchanged. With
\begin{equation}
    x:=\frac{\Delta}{h},
    \qquad
    u:=\frac{g_1}{g_0}e^{-x},
    \label{eq:schottky_xu}
\end{equation}
the total probability assigned to the high-dispersion level is $u/(1+u)$. The expected dispersion and its response are therefore
\begin{subequations}
\label{eq:schottky_response}
\begin{empheq}[left=\empheqlbrace]{align}
    U^{(F)}(h)&=V_0+\Delta\frac{u}{1+u},
    \label{eq:schottky_response:U}\\
    C^{(F)}(h)&=x^2\frac{u}{(1+u)^2}.
    \label{eq:schottky_response:C}
\end{empheq}
\end{subequations}
Eq.\,\eqref{eq:schottky_response:C} is the finite two-level Schottky response with a degeneracy ratio $g_1/g_0$. Differentiating $\ln C^{(F)}$ with respect to $x$ gives the peak equation
\begin{equation}
    x^*=2\frac{1+u^*}{1-u^*},
    \qquad
    u^*=\frac{g_1}{g_0}e^{-x^*},
    \qquad
    h^*=\frac{\Delta}{x^*}.
    \label{eq:schottky_peak}
\end{equation}
Thus the peak is not an arbitrary temperature scale: it is set by the dispersion gap $\Delta$ together with the number of feature states on each side of that gap. At $h\ll h^*$ the high-dispersion level is exponentially suppressed; at $h\gg h^*$ the two levels become insufficiently distinguished by the Boltzmann factor; the response is maximal where occupation probability is being transferred most rapidly between the two groups of feature states. This establishes a mechanism linking the response diagnostic to feature discrimination while preserving the finite-system interpretation: no thermodynamic singularity is present.

Real LAC spectra need not have exactly two levels. Finite samples, assignment error, anisotropy, and several groups of feature relevance broaden or split the levels. Eq.\,\eqref{eq:schottky_peak} therefore supplies both a reference model and a failure condition: a clean dispersion gap should produce a localized response maximum, whereas a broad, gapless, or multi-band spectrum can produce a wide or competing set of response scales. Sec.\,\ref{subsec:numerical_lac} tests this distinction by comparing prescribed spectra, spectra computed from the true partition, and spectra computed from the initial $k$-means partition.

'''
    if r"\label{subsec:lac_schottky}" not in text:
        text = text.replace(schottky_anchor, schottky_theory + schottky_anchor)

    text = text.replace(
        "so the two procedures are statistically indistinguishable at this sample size.",
        "so the interval includes zero and no difference is detected at this sample size. This statement is not an equivalence claim.",
    )
    text = text.replace(
        "Its ARI is statistically indistinguishable from the adaptive rule on the controlled design while avoiding repeated changes of the objective parameter.",
        "Its paired bootstrap interval relative to the adaptive rule includes zero on the controlled design, so no difference is detected at this sample size; this is not treated as evidence of formal equivalence. The frozen rule nevertheless avoids repeated changes of the objective parameter.",
    )

    hstar_rows = []
    for k in (1, 2, 3):
        p = sch[k]["prescribed"]
        t = sch[k]["true_partition"]
        m = sch[k]["kmeans_partition"]
        hstar_rows.append(
            f"{k} & {int(float(p['g_relevant']))}/{int(float(p['g_irrelevant']))} & "
            f"{float(p['delta_prescribed']):.4f} & {float(p['hstar_analytic']):.4f} & "
            f"{float(t['hstar_numeric']):.4f} & {float(m['hstar_numeric']):.4f}\\\\"
        )
    schottky_numeric = rf'''

The two-level mechanism can be checked before interpreting clustering accuracy. Because each synthetic column is centered and rescaled within its generating cluster before the prescribed standard deviation is applied, the true-partition dispersion spectrum is an exact two-level reference up to floating-point arithmetic. The operational response selector, however, starts from an estimated $k$-means partition, for which assignment errors broaden the two levels. Tab.\,\ref{{tab:schottky_validation}} and Fig.\,\ref{{fig:schottky_validation}} compare these three objects explicitly. The analytic peak agrees with the true-partition numerical maximum to within {max_true:.2f}\% in all three clusters; after the true labels are replaced by the initial $k$-means partition, the largest relative peak displacement is {max_km:.2f}\%. The response scale therefore survives the finite-sample and initialization perturbation in this controlled design, while the broadened spectrum makes clear that the operational problem is not literally an exactly degenerate two-level system.

\begin{{table}}[htbp]
\centering
\caption{{Two-level Schottky prediction and numerical LAC response maxima for seed 1729. $g_0/g_1$ denotes the number of prescribed low/high-dispersion feature states. The true-partition column uses dispersions recomputed from the generated observations; the $k$-means column uses the aligned initial estimated partition.}}
\label{{tab:schottky_validation}}
\begin{{tabular}}{{cccccc}}
\toprule
\textbf{{Cluster}} & $g_0/g_1$ & $\Delta$ & $h^*_{{\rm Sch}}$ & $h^*_{{\rm true}}$ & $h^*_{{k\text{{-means}}}}$\\
\midrule
{chr(10).join(hstar_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.94\linewidth]{{lac_schottky_validation.pdf}}
\caption{{Feature-dispersion spectra and response curves for the structured synthetic design. Left: prescribed, true-partition, and aligned initial $k$-means dispersions. Right: the analytic two-level Schottky response, the response from the true partition, and the response from the estimated initial partition. The analytic scale is fixed by the dispersion gap and degeneracy ratio rather than fitted to the numerical response.}}
\label{{fig:schottky_validation}}
\end{{figure}}
'''
    numeric_anchor = "The upper two panels of Fig.\\,\\ref{fig:lac_multiseed} show that the three prescribed dispersion spectra produce distinct effective-dimensionality trajectories and separated response maxima."
    if r"\label{fig:schottky_validation}" not in text:
        text = text.replace(numeric_anchor, numeric_anchor + schottky_numeric)

    baseline_rows = []
    for dataset in ("Synthetic", "Iris", "Wine", "WDBC"):
        cells = []
        for selector in ("kmeans", "response", "silhouette", "stability", "oracle"):
            if (dataset, selector) in baseline:
                cells.append(fmt_pm(*baseline[(dataset, selector)]))
            else:
                cells.append("--")
        baseline_rows.append(dataset + " & " + " & ".join(cells) + r"\\")

    baseline_text = rf'''

\subsubsection{{Matched-grid unsupervised temperature-selection baselines}}
\label{{subsubsec:h_baselines}}

The response rule is also compared with two genuinely unsupervised selectors rather than only with fixed numerical choices of $h$. Average silhouette width, Ref.\,\cite{{rousseeuw1987silhouette}}, is evaluated on the Euclidean geometry of the data for each candidate fixed-$h$ partition. A perturbation-stability score, following the general stability principle of Ben-Hur, Elisseeff, and Guyon, Ref.\,\cite{{benhur2002stability}}, is computed from pairwise ARI among four small-noise perturbations of the same dataset. To keep the computational comparison matched, response, silhouette, stability, and retrospective oracle selection use the same 36-point logarithmic grid $0.02\le h\le6$. The synthetic comparison uses ten seeds and each real dataset uses five common seeds; these runs are separate from the 30-seed estimates in Tabs.\,\ref{{tab:lac_multiseed}} and \ref{{tab:lac_real_benchmarks}}.

\begin{{table}}[htbp]
\centering
\small
\caption{{Matched-grid comparison of temperature selectors. Values are mean $\pm$ sample standard deviation of retrospective ARI. Response, silhouette, and stability are unsupervised; the oracle uses labels only to show the best fixed-$h$ value on the same grid.}}
\label{{tab:h_selector_baselines}}
\begin{{tabular}}{{lccccc}}
\toprule
\textbf{{Dataset}} & \textbf{{$k$-means}} & \textbf{{Response}} & \textbf{{Silhouette}} & \textbf{{Stability}} & \textbf{{Oracle}}\\
\midrule
{chr(10).join(baseline_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.90\linewidth]{{lac_selector_baselines.pdf}}
\caption{{Matched-grid comparison of unsupervised temperature selectors. Error bars show sample standard deviations across the dedicated baseline runs. The oracle is retrospective and is included only as a reference scale.}}
\label{{fig:h_selector_baselines}}
\end{{figure}}
'''

    real_anchor = "The absence of a universal win is consistent with the intended scope of the paper and prevents the controlled synthetic result from being generalized beyond its evidence."
    if r"\label{fig:h_selector_baselines}" not in text:
        text = text.replace(real_anchor, real_anchor + baseline_text)

    def sm(factor: str, value: float, method: str) -> float:
        return stress[(factor, value, method)][0]

    stress_text = rf'''

\subsubsection{{Controlled robustness beyond the reference generator}}
\label{{subsubsec:lac_stress}}

A one-factor-at-a-time stress study tests limitations that are not represented by changing the random seed alone. Eight seeds are evaluated per setting while varying center separation, cluster-size imbalance, the number of dimensions through additional irrelevant coordinates, and planar rotations that mix coordinates that are relevant for one cluster with coordinates that are irrelevant for another. Fig.\,\ref{{fig:lac_stress}} reports $k$-means, fixed $h=1$, and the two-pass global response rule. On the unmodified reference setting the response ARI is approximately {sm('center_factor', 1.0, 'response'):.3f}. Reducing the center factor to $0.5$ changes it to {sm('center_factor', 0.5, 'response'):.3f}; a four-to-one cluster-size ratio gives {sm('imbalance_ratio', 4.0, 'response'):.3f}; increasing the ambient dimension to 96 gives {sm('dimensions', 96.0, 'response'):.3f}; and a $45^\circ$ rotation gives {sm('rotation_degrees', 45.0, 'response'):.3f}. The rotation experiment is especially diagnostic because it directly attacks the axis-aligned local-feature assumption: any deterioration there is a structural limitation of diagonal feature weighting rather than a failure of numerical temperature optimization.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.92\linewidth]{{lac_stress_tests.pdf}}
\caption{{Controlled LAC stress tests. One factor is varied at a time around the reference synthetic generator: center separation, cluster-size imbalance, added irrelevant dimensions, and rotation away from axis-aligned feature relevance. Curves show mean ARI across eight seeds with one-sample-standard-deviation error bars.}}
\label{{fig:lac_stress}}
\end{{figure}}
'''
    if r"\label{fig:lac_stress}" not in text:
        text = text.replace(baseline_text, baseline_text + stress_text)

    sne_rows = []
    for p in (5.0, 20.0, 30.0, 50.0):
        s = sne[p]
        sne_rows.append(
            f"{int(p)} & {s['response_median']:.3f} & {100*s['tolerance_median']:.3f} & {100*s['actual_error_median']:.3f}\\\\"
        )
    sne_text = rf'''

The first-order rule can be tested directly rather than inferred only from the response distribution. For each Wine anchor and each target perplexity in $\{{5,20,30,50\}}$, the bandwidth is perturbed by the anchor-specific value $\varepsilon/(2C_i^{{(N)}})$ with $\varepsilon=0.01$, and the resulting perplexity is recomputed from the fixed distance spectrum. Tab.\,\ref{{tab:sne_tolerance_validation}} shows that the achieved median perplexity error remains close to the 1\% target across the four perplexities, while the required median bandwidth tolerance changes with the response distribution. This experiment validates the stated use of Eq.\,\eqref{{eq:perplexity_sigma_tolerance}} as a local first-order sensitivity estimate; it does not imply that such a tolerance materially accelerates a modern root solver.

\begin{{table}}[htbp]
\centering
\caption{{Direct validation of the first-order SNE bandwidth-tolerance rule on standardized Wine data. Tolerances and achieved errors are medians across anchors.}}
\label{{tab:sne_tolerance_validation}}
\begin{{tabular}}{{cccc}}
\toprule
\textbf{{Perplexity}} & \textbf{{Median $C_i^{{(N)}}$}} & \textbf{{Predicted $|\delta\sigma|/\sigma$ (\%)}} & \textbf{{Actual perplexity error (\%)}}\\
\midrule
{chr(10).join(sne_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.88\linewidth]{{sne_perplexity_validation.pdf}}
\caption{{Neighbor response and direct tolerance validation across target perplexities on Wine. Left: median and 95th-percentile response. Right: median bandwidth tolerance predicted for a 1\% perplexity error and the actual median error obtained after applying that perturbation.}}
\label{{fig:sne_perplexity_validation}}
\end{{figure}}
'''
    response_fig_end = r"\label{fig:sne_neighbor_response}\n\end{figure}"
    if r"\label{fig:sne_perplexity_validation}" not in text:
        text = text.replace(response_fig_end, response_fig_end + sne_text)

    old_lim = "The synthetic experiment varies random realization while holding the structural design fixed. Its 30-seed repetition therefore quantifies stochastic stability, not robustness over arbitrary dimensionality, overlap, imbalance, correlated relevant features, or non-Gaussian cluster geometry. The real-data benchmarks partly address this limitation and show that the response rules can improve LAC on Iris and Breast Cancer Wisconsin Diagnostic but do not dominate $k$-means on Wine. Broader benchmark suites remain necessary before the response criterion can be treated as a general model-selection procedure."
    new_lim = "The original 30-seed synthetic experiment varies random realization while holding the structural design fixed. The added one-factor stress tests extend this evidence to changes in overlap, imbalance, dimensionality, and rotation, but they still do not constitute a broad benchmark suite: the clusters remain Gaussian and the response study remains centered on diagonal LAC feature weighting. The rotation experiment is particularly relevant because it exposes the expected limitation of axis-aligned relevance. The real-data benchmarks likewise remain small in number. Broader datasets and non-Gaussian geometries are therefore still required before the response criterion can be treated as a general model-selection procedure."
    text = text.replace(old_lim, new_lim)

    concl_anchor = "That distinction leads to testable consequences."
    if "Schottky mechanism" not in text[text.find(r"\section{Conclusion}"):]:
        text = text.replace(
            concl_anchor,
            "That distinction leads to a specific spectral mechanism. For a two-level LAC dispersion spectrum, the response is a finite Schottky anomaly whose peak is fixed by the dispersion gap and the degeneracy ratio. The synthetic experiment confirms the analytic peak on the true partition and quantifies its displacement when the spectrum is estimated from the initial $k$-means partition. Broad or rotated spectra provide corresponding failure modes rather than unexplained exceptions.\n\n" + concl_anchor,
        )

    text = text.replace(
        "For SNE, the response has a direct numerical use rather than only a thermodynamic interpretation. It converts a target perplexity accuracy into an anchor-specific first-order bandwidth tolerance, while controlled scaling experiments distinguish an exact self-similar law from regression and geometric effects present in Wine.",
        "For SNE, the response has a direct numerical use rather than only a thermodynamic interpretation. It converts a target perplexity accuracy into an anchor-specific first-order bandwidth tolerance, and direct perturbation experiments across four perplexities show where that linearization is quantitatively accurate. The result is explicitly conditional on a chosen perplexity and is complementary to multiscale, perplexity-free, and automatic-perplexity methods rather than a replacement for them. Controlled scaling experiments remain a consistency check that separates the exact self-similar law from regression and geometric effects present in Wine.",
    )

    text = text.replace(
        "The Python dependencies are listed in \\path{paper/experiments/requirements.txt}",
        "The fully pinned execution environment is listed in \\path{requirements.lock}",
    )
    text = text.replace(
        "Claude Opus was used as an auxiliary tool for technical discussion of the subject matter, critical review of the manuscript, bibliographic research, validation of mathematical passages, and checks of textual coherence.",
        "Claude Opus was used as an auxiliary tool for technical discussion of the subject matter, critical review of the manuscript, bibliographic research, auxiliary checking of mathematical passages, and checks of textual coherence; all such checks were independently reviewed by the author.",
    )
    text = text.replace(
        "The most capable GPT model available to the author during preparation of the repository and manuscript was used",
        "GPT-5.6 Sol (OpenAI, used during the August 2026 revision) was used",
    )

    MAIN.write_text(text, encoding="utf-8")


def update_manifest_and_readme() -> None:
    manifest = PAPER / "experiments" / "MANIFEST.txt"
    text = manifest.read_text(encoding="utf-8")
    additions = [
        "revision_experiments.py",
        "assets/lac_schottky_diagnostics.csv",
        "assets/lac_unsupervised_baselines.csv",
        "assets/lac_stress_tests.csv",
        "assets/sne_perplexity_validation.csv",
        "assets/source_images/lac_schottky_validation.png",
        "assets/source_images/lac_selector_baselines.png",
        "assets/source_images/lac_stress_tests.png",
        "assets/source_images/sne_perplexity_validation.png",
        "../figures/lac_schottky_validation.pdf",
        "../figures/lac_selector_baselines.pdf",
        "../figures/lac_stress_tests.pdf",
        "../figures/sne_perplexity_validation.pdf",
    ]
    existing = set(text.splitlines())
    for item in additions:
        if item not in existing:
            text += "\n" + item
    manifest.write_text(text.strip() + "\n", encoding="utf-8")

    readme = PAPER / "experiments" / "README.md"
    rtext = readme.read_text(encoding="utf-8")
    if "Schottky" not in rtext:
        rtext += "\n## Scientific revision diagnostics\n\nThe paper driver also runs `revision_experiments.py`, which adds the two-level Schottky validation, matched-grid response/silhouette/stability temperature selectors, controlled LAC stress tests, and direct multi-perplexity validation of the SNE first-order tolerance rule. These diagnostics use the canonical installed LAC implementation and write stable CSV/figure names under the same asset contract.\n"
        readme.write_text(rtext, encoding="utf-8")


def main() -> None:
    append_bibliography()
    ctx = build_numeric_context()
    apply_main_revision(ctx)
    update_manifest_and_readme()


if __name__ == "__main__":
    main()
