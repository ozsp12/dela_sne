# DELA-SNE

Research repository for **Dual-Entropy Locally Adaptive Stochastic Neighbor Embedding (DELA-SNE)** and its two algorithmic foundations: **Locally Adaptive Clustering (LAC)** and **t-distributed Stochastic Neighbor Embedding (t-SNE)**.

The repository separates reference algorithms, reusable data workflows, and a self-contained manuscript tree. LAC and t-SNE are implemented as independent reference modules. DELA-SNE remains a research implementation target and is not yet frozen as software.

## Algorithms

| Method | Role | Implementation | Output |
|---|---|---|---|
| LAC | clustering with cluster-dependent feature relevance | [`lac/lac.py`](lac/lac.py) | hard cluster, assigned weighted distance, local feature weights |
| t-SNE | nonlinear probabilistic embedding | [`tsne/tsne.py`](tsne/tsne.py) | two-dimensional embedding |
| DELA-SNE | proposed method | pending | pending |

## Repository structure

```text
dela_sne/
├── README.md
├── run_workflow.py
├── requirements.txt
├── .github/
│   └── workflows/
│       └── algorithms.yml
├── data/
│   ├── test/
│   └── result/
├── lac/
├── tsne/
├── dela_sne/
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   │   └── *.pdf
│   └── experiments/
│       ├── run_experiments.py
│       ├── requirements.txt
│       ├── README.md
│       ├── MANIFEST.txt
│       └── assets/
│           ├── source_images/
│           │   └── *.png
│           └── *.csv
├── references/
├── docs/
└── tests/
```

## Data convention

`data/test/` contains input datasets and `data/result/` contains generated algorithm outputs. Test CSV files must contain numeric feature columns named `feature_1`, `feature_2`, and so on. Other columns are preserved as metadata and are not used as model inputs.

The committed baseline dataset, [`data/test/df_baseline.csv`](data/test/df_baseline.csv), is a deterministic synthetic benchmark generated with random seed 42. It contains 96 observations, eight numerical features, and three known groups. Informative dimensions differ by group, while other dimensions contain larger nuisance variation. The `true_cluster` column is retained only for validation and is excluded from algorithm inputs.

## Joint algorithm workflow

Run both implemented reference algorithms over every CSV in `data/test/`:

```bash
python run_workflow.py
```

For an input such as `data/test/df_baseline.csv`, outputs are written under `data/result/` with the execution date in the result file name.

## Manuscript and paper experiments

[`paper/`](paper/) is self-contained. It contains the current LaTeX source, its bibliography, final PDF figures, the complete experiment driver, numerical CSV outputs, and PNG companion/source exports.

Regenerate the paper experiment assets with:

```bash
python -m pip install -r paper/experiments/requirements.txt
python paper/experiments/run_experiments.py
```

The experiment uses stable file names and overwrites the current assets. Historical versions are provided by Git rather than timestamped duplicate files.

Compile the manuscript from `paper/` with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## GitHub Actions

[`.github/workflows/algorithms.yml`](.github/workflows/algorithms.yml) is the repository CI workflow. It runs unit tests and the joint LAC/t-SNE workflow, executes the complete paper experiment, verifies that all expected manuscript figures and numerical assets are generated, checks that committed paper assets are reproducible, compiles `paper/main.tex`, and uploads the algorithm results, paper assets, and compiled manuscript as workflow artifacts.

## Reproducibility

The reference algorithm workflow and the paper experiment use fixed seeds. The paper experiment stores numerical outputs in `paper/experiments/assets/`, PNG companion exports in `paper/experiments/assets/source_images/`, and final vector figures in `paper/figures/`. The LaTeX source references only files inside `paper/`, so that directory can be archived or compiled independently after installing the documented dependencies.

## Documentation

Algorithm-specific documentation is colocated with each algorithm:

- [`lac/docs/`](lac/docs/)
- [`tsne/docs/`](tsne/docs/)
- [`dela_sne/docs/`](dela_sne/docs/)

Project-wide documentation remains under [`docs/`](docs/).
