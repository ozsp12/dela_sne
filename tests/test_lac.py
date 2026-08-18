import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from dela_sne import LAC

ROOT = Path(__file__).resolve().parents[1]


def load_baseline() -> tuple[np.ndarray, np.ndarray]:
    with (ROOT / "data" / "test" / "df_baseline.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    feature_names = [name for name in rows[0] if name.startswith("feature_")]
    X = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=float
    )
    y = np.asarray([int(row["true_cluster"]) for row in rows], dtype=int)
    return StandardScaler().fit_transform(X), y


def test_lac_recovers_reference_structure() -> None:
    X, y = load_baseline()
    model = LAC(n_clusters=3, h=0.2, n_init=10, random_state=42).fit(X)
    assert adjusted_rand_score(y, model.labels_) > 0.80


def test_lac_is_deterministic_for_fixed_seed() -> None:
    X, _ = load_baseline()
    first = LAC(n_clusters=3, h=0.2, n_init=5, random_state=17).fit(X)
    second = LAC(n_clusters=3, h=0.2, n_init=5, random_state=17).fit(X)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_allclose(first.feature_weights_, second.feature_weights_)
    assert first.objective_ == pytest.approx(second.objective_)


def test_lac_objective_contains_entropy_term() -> None:
    X, _ = load_baseline()
    model = LAC(n_clusters=3, h=0.2, n_init=2, random_state=5).fit(X)
    assert model.entropy_term_ <= 0.0
    assert model.objective_ == pytest.approx(
        model.distance_term_ + model.entropy_term_
    )
    np.testing.assert_allclose(model.feature_weights_.sum(axis=1), 1.0)


def test_lac_validates_iteration_count() -> None:
    X, _ = load_baseline()
    with pytest.raises(ValueError, match="max_iter"):
        LAC(n_clusters=3, max_iter=0).fit(X)
