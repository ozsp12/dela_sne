import csv
from pathlib import Path

import numpy as np
import pytest

from dela_sne.workflow import read_csv, run_dataset


def test_workflow_uses_stable_result_name(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    X = np.vstack(
        [
            rng.normal(-2.0, 0.2, size=(8, 3)),
            rng.normal(0.0, 0.2, size=(8, 3)),
            rng.normal(2.0, 0.2, size=(8, 3)),
        ]
    )
    input_path = tmp_path / "sample.csv"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "feature_1", "feature_2", "feature_3"])
        for i, row in enumerate(X):
            writer.writerow([f"S{i:03d}", *row])

    output = run_dataset(
        input_path,
        output_dir=tmp_path / "result",
        n_clusters=3,
        h=0.2,
        perplexity=5,
        random_state=9,
        n_init=3,
    )

    assert output.name == "sample_result.csv"
    assert b"\r\n" not in output.read_bytes()
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(X)
    assert {"lac_cluster", "lac_distance", "tsne_1", "tsne_2"}.issubset(rows[0])


def test_read_csv_rejects_missing_feature_columns(tmp_path: Path) -> None:
    path = tmp_path / "missing_features.csv"
    path.write_text("sample_id,value\nA,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain numeric columns"):
        read_csv(path)


def test_read_csv_rejects_non_numeric_features(tmp_path: Path) -> None:
    path = tmp_path / "non_numeric.csv"
    path.write_text("feature_1,feature_2\n1,not-a-number\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric feature values"):
        read_csv(path)


def test_read_csv_rejects_non_finite_features(tmp_path: Path) -> None:
    path = tmp_path / "non_finite.csv"
    path.write_text("feature_1,feature_2\n1,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NaN or infinite feature values"):
        read_csv(path)


def test_read_csv_rejects_empty_datasets(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("feature_1,feature_2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no observations"):
        read_csv(path)
