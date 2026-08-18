import numpy as np
import pytest

from dela_sne.lac_diagnostics import iterate_with_temperature_rule


def test_temperature_rule_iteration_converges_and_records_path() -> None:
    X = np.asarray(
        [
            [-2.2, -2.0],
            [-1.8, -2.1],
            [-2.0, -1.7],
            [1.8, 2.0],
            [2.1, 1.9],
            [2.0, 2.2],
        ]
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])

    def fixed_temperature(V: np.ndarray, iteration: int) -> float:
        assert V.shape == (2, 2)
        assert iteration >= 1
        return 0.2

    result = iterate_with_temperature_rule(
        X,
        n_clusters=2,
        init_labels=labels,
        temperature_rule=fixed_temperature,
        max_iter=20,
    )
    final_labels, h, n_iter, status, cycle_period, h_path = result

    np.testing.assert_array_equal(final_labels, labels)
    assert float(h) == pytest.approx(0.2)
    assert n_iter >= 1
    assert status == "converged"
    assert cycle_period == 0
    assert len(h_path) == n_iter


def test_temperature_rule_iteration_validates_inputs() -> None:
    X = np.asarray([[-1.0], [-0.9], [1.0], [0.9]])
    labels = np.asarray([0, 0, 1, 1])

    with pytest.raises(ValueError, match="max_iter"):
        iterate_with_temperature_rule(
            X,
            n_clusters=2,
            init_labels=labels,
            temperature_rule=lambda V, i: 0.2,
            max_iter=0,
        )

    with pytest.raises(ValueError, match="every cluster"):
        iterate_with_temperature_rule(
            X,
            n_clusters=2,
            init_labels=np.zeros(len(X), dtype=int),
            temperature_rule=lambda V, i: 0.2,
            max_iter=5,
        )
