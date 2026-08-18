# Associated manuscript

## Current manuscript

*Feature and Neighbor Ensembles in Locally Adaptive Clustering and Stochastic Neighbor Embedding: Response Functions and Perplexity Tolerance*

The source is maintained in [`paper/main.tex`](https://github.com/ozsp12/dela_sne/blob/main/paper/main.tex), with reproducible numerical experiments under `paper/experiments/`.

## Scientific scope

The current article studies the normalized-exponential structure shared by LAC and SNE/t-SNE, with emphasis on feature-response diagnostics, temperature selection in LAC, neighbor-response diagnostics, perplexity tolerance, deterministic annealing, and numerical validation. It is a foundational article within the wider DELA-SNE research program.

The manuscript does **not** currently define the final DELA-SNE algorithm. DELA-SNE remains under mathematical specification; its metric definition, entropy coupling, asymmetry handling, symmetrization, objective, gradient, initialization, stopping rules, scale conventions, and validation protocol must satisfy the project freeze checklist before a stable implementation or manuscript claim is made.

## Repository relation

The canonical executable implementations of LAC and exact t-SNE live under `src/dela_sne/`. The manuscript experiment entrypoint uses the installed canonical LAC core, while retaining paper-specific datasets, response rules, diagnostics, and plotting logic.

This separation is intentional: repository engineering and reproducibility work may proceed without changing the scientific text, figures, equations, or conclusions of the manuscript. Changes to those scientific outputs should follow the dedicated manuscript analysis and validation workflow.

## DELA-SNE development

The project-level DELA-SNE specification and freeze criteria are tracked in [`docs/dela_sne.md`](dela_sne.md). A future DELA-SNE-specific manuscript should be treated as a distinct scientific deliverable unless the final research results justify a different editorial structure.
