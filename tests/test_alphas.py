import numpy as np

from wqsim.alphas import ALPHAS, AlphaContext
from wqsim.regression import (
    SyntheticPanelSpec,
    alpha_regression_snapshot,
    benchmark_alphas,
    make_synthetic_panel,
)


def test_registered_alphas_evaluate_to_input_shape():
    rows = 80
    cols = 5
    base = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) + 100.0
    context = AlphaContext(
        open=base,
        high=base + 2.0,
        low=base - 2.0,
        close=base + 1.0,
        volume=(base * 1000.0) + 1.0,
        returns=np.sin(base / 10.0),
        vwap=base + 0.5,
    )

    expected = {
        "eg_alpha",
        "eg_alpha2",
        "eg_alpha3",
        "alpha001",
        "alpha002",
        "alpha003",
        "alpha004",
        "alpha005",
        "alpha006",
        "alpha007",
        "alpha008",
        "alpha009",
        "alpha010",
    }

    assert expected == set(ALPHAS)
    for alpha in ALPHAS.values():
        actual = alpha(context)
        assert actual.shape == (rows, cols)


def test_synthetic_alpha_regression_snapshot_is_deterministic():
    panel = make_synthetic_panel(SyntheticPanelSpec(rows=72, cols=8, seed=1234, nan_rate=0.04))

    first = alpha_regression_snapshot(panel)
    second = alpha_regression_snapshot(panel)

    assert first == second
    assert set(first) == set(ALPHAS)
    assert first["eg_alpha3"] == {
        "shape": [72, 8],
        "finite_count": 482,
        "nan_count": 94,
        "posinf_count": 0,
        "neginf_count": 0,
        "mean": 0.5,
        "std": 0.3292728382194115,
        "min": 0.0,
        "max": 1.0,
        "sha256": "40326c3aba80d0161065cde5ff4eb24c29ef79fdcd1b6cf574a3bf482e74b011",
    }


def test_benchmark_alphas_reports_all_registered_alphas():
    panel = make_synthetic_panel(SyntheticPanelSpec(rows=64, cols=6, seed=5678, nan_rate=0.02))

    rows = benchmark_alphas(panel, repeat=1)

    assert [row["alpha"] for row in rows] == sorted(ALPHAS)
    assert all(row["seconds_min"] >= 0.0 for row in rows)
    assert all(row["digest"]["shape"] == [64, 6] for row in rows)
