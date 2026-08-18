from pathlib import Path

path = Path(__file__).with_name("main.tex")
text = path.read_text(encoding="utf-8")

text = text.replace(
    r"\usepackage{booktabs,tabularx,array}",
    r"\usepackage{booktabs,tabularx,array,longtable}",
)

keywords = r"\noindent\textbf{Keywords:} statistical mechanics; Gibbs distribution; Shannon entropy; clustering; locally adaptive clustering; stochastic neighbor embedding; t-SNE; perplexity; dimensionality reduction; local metric learning."

glossary = r'''

\section*{Glossary}

\begin{longtable}{>{\raggedright\arraybackslash}p{0.22\textwidth} >{\raggedright\arraybackslash}p{0.72\textwidth}}
\toprule
\textbf{Term or symbol} & \textbf{Meaning}\\
\midrule
\endfirsthead
\toprule
\textbf{Term or symbol} & \textbf{Meaning}\\
\midrule
\endhead
LAC & Locally Adaptive Clustering.\\
SNE & Stochastic Neighbor Embedding.\\
t-SNE & t-distributed Stochastic Neighbor Embedding.\\
ARI & Adjusted Rand index.\\
KL divergence & Kullback--Leibler divergence.\\
DBSCAN & Density-Based Spatial Clustering of Applications with Noise.\\
LLE & Locally Linear Embedding.\\
UMAP & Uniform Manifold Approximation and Projection.\\
OLS & Ordinary least squares.\\
TLS & Total least squares.\\
WDBC & Breast Cancer Wisconsin Diagnostic dataset.\\
$X=(x_1,\ldots,x_n)$ & Dataset of $n$ observations $x_i\in\mathbb R^D$.\\
$\mathcal S$ & Generic finite state space; specialized to features $\mathbb F$ or candidate neighbors $\mathcal N_i$.\\
$\mathbb F$ & Finite LAC feature-state space.\\
$\mathcal N_i$ & Candidate-neighbor state space for SNE anchor $i$.\\
$p$ & Probability vector on a finite state space, with $p\in\Delta_{m-1}$.\\
$E_a$ & Dimensionless effective energy attached to state $a$.\\
$T$ & Positive canonical scale, termed effective temperature in the algorithmic analogy.\\
$S[p]$ & Shannon entropy of $p$ in nats.\\
$U[p]$ & Expected effective energy $\sum_a p_aE_a$.\\
$C(T)$ & Fixed-spectrum response $\partial U/\partial T=\operatorname{Var}(E)/T^2$; equal to dimensionless heat capacity in the canonical physical model.\\
$V_{\ell f}$ & Within-cluster dispersion of feature $f$ in LAC.\\
$w_{\ell f}$ & LAC feature-state probability for cluster $\ell$.\\
$h$ & LAC smoothing parameter and effective feature temperature.\\
$\sigma_i$ & Local Gaussian bandwidth for SNE anchor $i$.\\
$T_i^{(N)}=2\sigma_i^2$ & Local SNE neighbor temperature for anchor $i$.\\
$\Perp(P_i)$ & Perplexity of the local SNE neighbor distribution $P_i$.\\
$q_{ij}$ & Low-dimensional t-SNE pair probability.\\
\bottomrule
\end{longtable}

\clearpage
\tableofcontents
\clearpage
'''

if "\\section*{Glossary}" not in text:
    text = text.replace(keywords, keywords + glossary)

symbol_caption = r"\caption{Principal symbols used in the statistical-mechanical formulation. Algorithm-specific quantities are introduced again at their first detailed use.}"
pos = text.find(symbol_caption)
if pos != -1:
    start = text.rfind(r"\begin{table}[htbp]", 0, pos)
    end = text.find(r"\end{table}", pos)
    if start == -1 or end == -1:
        raise RuntimeError("Could not isolate the principal-symbol table.")
    end += len(r"\end{table}")
    text = text[:start] + text[end:]

availability_heading = r"\subsection*{Data and code availability}"
pos = text.find(availability_heading)
if pos != -1:
    end = text.find(r"\FloatBarrier", pos)
    if end == -1:
        raise RuntimeError("Could not isolate the existing data availability statement.")
    text = text[:pos] + text[end:]

closing = r"Broader clustering benchmarks and a separately validated feature--neighbor embedding remain the next substantive steps."
endmatter = r'''

\section*{Data and code availability}
All numerical figures, CSV summaries, random seeds, and experiment drivers used in Sec.\,\ref{sec:numerical} are versioned with the manuscript under \path{paper/experiments/}. The executable driver \path{paper/experiments/run_experiments.py} regenerates the tabular outputs in \path{paper/experiments/assets/}, the companion raster figure exports in \path{paper/experiments/assets/source_images/}, and the PDF figures consumed directly by the LaTeX source in \path{paper/figures/}. File names are stable and each workflow execution overwrites the current derived artifacts; historical versions are retained by Git. The Python dependencies are listed in \path{paper/experiments/requirements.txt}, and the three real datasets are loaded from scikit-learn without a network request at execution time Ref.\,\cite{pedregosa2011scikit}.

\section*{Artificial intelligence use disclosure}
The most capable GPT model available to the author during preparation of the repository and manuscript was used to assist with implementation and review of repository software code, software tests, and preparation of the BibTeX bibliography file. Claude Opus was used as an auxiliary tool for technical discussion of the subject matter, critical review of the manuscript, bibliographic research, validation of mathematical passages, and checks of textual coherence. Grammarly was used for grammatical review. All mathematical claims, interpretations, references, code, numerical results, and final manuscript content were reviewed by the author, who retains full responsibility for the published work.
'''

if "\\section*{Artificial intelligence use disclosure}" not in text:
    text = text.replace(closing, closing + endmatter)

path.write_text(text, encoding="utf-8")
Path(__file__).unlink()
