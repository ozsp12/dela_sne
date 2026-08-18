import csv
from pathlib import Path

import numpy as np

from dela_sne.workflow import run_dataset


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
