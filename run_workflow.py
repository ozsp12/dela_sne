"""Joint reproducible workflow for the LAC and t-SNE reference implementations."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

from lac.lac import LAC
from tsne.tsne import TSNE


ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_DIR = ROOT / "data" / "test"
DEFAULT_RESULT_DIR = ROOT / "data" / "result"


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header.")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    feature_names = [name for name in fieldnames if name.startswith("feature_")]
    if not feature_names:
        raise ValueError(
            f"{path} must contain numeric columns named feature_1, feature_2, ..."
        )

    try:
        X = np.asarray(
            [[float(row[name]) for name in feature_names] for row in rows],
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains non-numeric feature values.") from exc

    if not np.all(np.isfinite(X)):
        raise ValueError(f"{path} contains NaN or infinite feature values.")
    return rows, feature_names, X


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (X - mean) / scale, mean, scale


def _write_result(
    source_rows: list[dict[str, str]],
    lac: LAC,
    embedding: np.ndarray,
    output_path: Path,
) -> None:
    """Write observation-level outputs.

    LAC feature weights are cluster-level parameters and remain available through
    ``LAC.feature_weights_`` rather than being duplicated on every CSV row.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_fields = list(source_rows[0].keys())
    result_fields = original_fields + ["lac_cluster", "lac_distance"] + [
        f"tsne_{i + 1}" for i in range(embedding.shape[1])
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()

        for index, source in enumerate(source_rows):
            result = dict(source)
            result["lac_cluster"] = int(lac.labels_[index])
            result["lac_distance"] = f"{lac.distance_to_assigned_[index]:.8g}"
            for j in range(embedding.shape[1]):
                result[f"tsne_{j + 1}"] = f"{embedding[index, j]:.8g}"
            writer.writerow(result)


def run_dataset(
    input_path: Path,
    output_dir: Path = DEFAULT_RESULT_DIR,
    date_stamp: str | None = None,
    n_clusters: int = 3,
    h: float = 0.5,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Path:
    rows, _, X = _read_csv(input_path)
    X_scaled, _, _ = _standardize(X)

    lac = LAC(
        n_clusters=n_clusters,
        h=h,
        random_state=random_state,
    ).fit(X_scaled)

    effective_perplexity = min(
        float(perplexity), max(2.0, (len(X_scaled) - 1) / 3.0)
    )
    tsne = TSNE(
        perplexity=effective_perplexity,
        random_state=random_state,
    )
    embedding = tsne.fit_transform(X_scaled)

    stamp = date_stamp or datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"{input_path.stem}_result_{stamp}.csv"
    _write_result(rows, lac, embedding, output_path)

    print(
        f"{input_path.name} -> {output_path.name} | "
        f"LAC iterations={lac.n_iter_}, "
        f"t-SNE KL={tsne.kl_divergence_:.6f}"
    )
    return output_path


def _resolve_inputs(values: list[str] | None) -> list[Path]:
    if values:
        return [Path(value).resolve() for value in values]
    return sorted(DEFAULT_TEST_DIR.glob("*.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LAC and t-SNE over one or more test CSV datasets."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input CSV files. Defaults to every CSV under data/test/.",
    )
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date", default=None, help="Optional YYYYMMDD output stamp.")
    args = parser.parse_args()

    inputs = _resolve_inputs(args.inputs)
    if not inputs:
        print("No CSV test datasets found.", file=sys.stderr)
        return 2

    for input_path in inputs:
        run_dataset(
            input_path=input_path,
            date_stamp=args.date,
            n_clusters=args.clusters,
            h=args.h,
            perplexity=args.perplexity,
            random_state=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
