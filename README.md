# DELA-SNE

Research software and manuscript repository for **Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)** and its two reference foundations: **Locally Adaptive Clustering (LAC)** and **t-distributed Stochastic Neighbor Embedding (t-SNE)**.

The executable reference code has one canonical location: [`src/dela_sne/`](src/dela_sne/). The top-level [`lac/`](lac/) and [`tsne/`](tsne/) directories contain method-specific documentation, while [`paper/`](paper/) contains the self-contained manuscript and its experiment layer.

## Status

| Method | Status | Canonical code |
|---|---|---|
| LAC | implemented and tested | [`src/dela_sne/lac.py`](src/dela_sne/lac.py) |
| t-SNE | implemented and tested | [`src/dela_sne/tsne.py`](src/dela_sne/tsne.py) |
| DELA-SNE | mathematical specification in progress | [freeze checklist](docs/dela_sne.md) |

DELA-SNE is intentionally not exposed as a stable class until the metric, entropy coupling, symmetrization, objective, gradient, initialization, stopping rules, and validation protocol satisfy the explicit freeze checklist.

## Installation

For a reproducible development environment:

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

The package can then be imported independently of the current working directory:

```python
from dela_sne import LAC, TSNE
```

## Repository structure

```text
dela_sne/
├── pyproject.toml
├── requirements.lock
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── run_workflow.py
├── src/dela_sne/           # canonical executable package
├── lac/                    # LAC documentation
├── tsne/                   # t-SNE documentation
├── data/
│   ├── test/               # versioned deterministic inputs
│   └── result/             # generated stable-name outputs
├── paper/                  # manuscript and numerical experiment layer
├── references/
│   └── references.bib      # canonical bibliography
├── scripts/                # reproducibility checks
├── tests/                  # numerical and workflow tests
└── .github/workflows/      # CI
```

## LAC

The canonical LAC implementation includes cluster-dependent feature weights, a complete entropy-regularized objective, multiple restarts, deterministic seeding, cycle diagnostics, validation of stopping parameters, and unique reseeding for multiple empty clusters.

The fitted object exposes, among other quantities:

- `labels_` and `cluster_centers_`;
- `feature_weights_`;
- `distance_term_`;
- `entropy_term_`;
- `objective_ = distance_term_ + entropy_term_`;
- `status_`, `cycle_period_`, and `restart_objectives_`.

The temperature `h` has the dimension of a feature variance. It is therefore not invariant to rescaling of the input features. The shared workflow standardizes features before LAC and t-SNE are run.

## t-SNE

The exact NumPy reference implementation retains the O(n^2) probability matrices for transparency. It supports PCA or random initialization, adaptive gains, a sample-size-aware `learning_rate="auto"`, early exaggeration, momentum, and stopping by gradient norm. The analytic KL gradient is tested against finite differences.

The implementation is intended as an inspectable reference for small datasets, not as a replacement for Barnes-Hut or FFT-accelerated production implementations.

## Shared data workflow

Run all CSV files in `data/test/`:

```bash
python run_workflow.py
```

Or run one file:

```bash
python run_workflow.py data/test/df_baseline.csv
```

Outputs use stable names. For example:

```text
data/test/df_baseline.csv
        -> data/result/df_baseline_result.csv
```

Git records historical changes; timestamps are not encoded in filenames. Generated result CSVs are ignored by default and are uploaded by CI as artifacts.

The baseline dataset contains eight numeric feature columns and a `true_cluster` validation label that is excluded from model input.

## Manuscript reproducibility

The manuscript experiment entrypoint is:

```bash
python paper/experiments/run_experiments.py
```

The entrypoint binds its LAC operations to the installed canonical package before executing the paper-specific datasets, temperature rules, diagnostics, and plots. Thus the article does not maintain an independent LAC update implementation.

[`references/references.bib`](references/references.bib) is the canonical bibliography. [`paper/references.bib`](paper/references.bib) is a synchronized copy so the `paper/` tree remains self-contained:

```bash
python scripts/sync_bibliography.py --check
```

Numerical paper assets are checked with floating-point tolerances. Figure files are regenerated and checked for existence rather than compared byte-for-byte, because renderer metadata and font metrics are not a scientifically meaningful equality criterion.

## Continuous integration

CI runs Python 3.11 and 3.12 quality jobs with:

- Ruff static analysis;
- mypy type checking;
- pytest with coverage;
- deterministic/numerical LAC tests;
- deterministic and finite-difference t-SNE tests;
- workflow contract tests.

Separate jobs execute the shared data workflow and reproduce/compile the manuscript. The workflow has `contents: read`; it does not push generated PDFs back to the repository. Result datasets, experiment assets, and the compiled manuscript are published as CI artifacts.

## References and redistribution

The public repository stores bibliographic metadata rather than publisher PDFs. See [`references/README.md`](references/README.md).

## Citation and releases

Software citation metadata are provided in [`CITATION.cff`](CITATION.cff). The manuscript should cite an immutable tagged and archived release, not `main`. A release checklist is maintained in [`docs/RELEASING.md`](docs/RELEASING.md); a Zenodo DOI must be added only after an actual archive has been created.

## License

The software is released under the [MIT License](LICENSE). Third-party papers, datasets, and other external materials retain their own licenses.
