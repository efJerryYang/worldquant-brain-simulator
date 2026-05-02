import numpy as np

from wqsim.config import SimulatorConfig
from wqsim.data import MarketPanel
from wqsim.simulator import _alpha_input_panel, _post_process, _simulate_pnl, run_simulation


def test_simulator_smoke_path_without_database(tmp_path):
    rows = 95
    cols = 4
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + rows)
    base = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) + 20.0
    fields = {
        "open": base,
        "high": base + 1,
        "low": base - 1,
        "close": base + 0.5,
        "volume": np.full((rows, cols), 1_000_000.0),
        "returns": np.full((rows, cols), 0.1),
        "vwap": base + 0.25,
        "cumulative_liq": np.tile(np.arange(cols, dtype=np.float64), (rows, 1)),
    }
    panel = MarketPanel(dates=dates, symbols=np.array(["A", "B", "C", "D"]), fields=fields)
    config = SimulatorConfig(
        universe="Top2",
        start_date="2020-03-01",
        output_dir=tmp_path,
        plot=False,
    )

    result = run_simulation(config, "eg_alpha3", panel)

    assert result.pnl.size > 0
    assert result.dates.size == result.pnl.size
    assert result.turnover.size == result.pnl.size
    assert np.isfinite(result.cumulative_pnl).all()
    assert result.metrics["traded_days"] == result.pnl.size


def test_simulator_accounting_with_known_alpha_matrix():
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-05"))
    fields = {
        "returns": np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [2.0, -2.0],
                [-2.0, 2.0],
            ]
        ),
        "cumulative_liq": np.ones((4, 2)),
    }
    panel = MarketPanel(dates=dates, symbols=np.array(["A", "B"]), fields=fields)
    alpha = np.array(
        [
            [0.0, 0.0],
            [10.0, -10.0],
            [-10.0, 10.0],
            [0.0, 0.0],
        ]
    )
    config = SimulatorConfig(
        universe="Top2",
        delay=1,
        neutralization="None",
        truncation=0.0,
        booksize=100.0,
        start_date="2020-01-03",
    )

    pnl_dates, pnl, turnover = _simulate_pnl(panel, alpha, config)

    np.testing.assert_array_equal(
        pnl_dates, np.array(["2020-01-03", "2020-01-04"], dtype="datetime64[D]")
    )
    np.testing.assert_allclose(pnl, [2.0, 2.0])
    np.testing.assert_allclose(turnover, [0.0, 2.0])


def test_simulation_writes_json_sidecar(tmp_path):
    rows = 95
    cols = 4
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + rows)
    base = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) + 20.0
    fields = {
        "open": base,
        "high": base + 1,
        "low": base - 1,
        "close": base + 0.5,
        "volume": np.full((rows, cols), 1_000_000.0),
        "returns": np.full((rows, cols), 0.1),
        "vwap": base + 0.25,
        "cumulative_liq": np.tile(np.arange(cols, dtype=np.float64), (rows, 1)),
    }
    panel = MarketPanel(dates=dates, symbols=np.array(["A", "B", "C", "D"]), fields=fields)
    config = SimulatorConfig(
        universe="Top2",
        start_date="2020-03-01",
        output_dir=tmp_path,
        plot=True,
    )

    result = run_simulation(config, "eg_alpha3", panel)

    assert result.plot_path is not None
    assert result.metrics_path is not None
    assert result.plot_path.exists()
    assert result.metrics_path.exists()
    assert result.metrics_path.read_text().startswith("{")


def test_post_process_modes_expose_truncation_neutralization_effects():
    alpha_row = np.array([10.0, 8.0, 7.0, -1.0])
    legacy = SimulatorConfig(
        neutralization="Market",
        truncation=0.1,
        post_process_mode="legacy",
    )
    renormalized = SimulatorConfig(
        neutralization="Market",
        truncation=0.1,
        post_process_mode="renormalize_after_truncation",
    )
    normalized_first = SimulatorConfig(
        neutralization="Market",
        truncation=0.1,
        post_process_mode="normalize_then_truncate",
    )
    no_truncation = SimulatorConfig(
        neutralization="Market",
        truncation=0.1,
        post_process_mode="no_truncation",
    )

    legacy_weights = _post_process(alpha_row, legacy)
    renormalized_weights = _post_process(alpha_row, renormalized)
    normalized_first_weights = _post_process(alpha_row, normalized_first)
    no_truncation_weights = _post_process(alpha_row, no_truncation)

    assert legacy_weights.sum() > 0.0
    assert normalized_first_weights.sum() > 0.0
    np.testing.assert_allclose(renormalized_weights.sum(), 0.0, atol=1e-15)
    np.testing.assert_allclose(no_truncation_weights.sum(), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.abs(renormalized_weights).sum(), 1.0)


def test_unknown_post_process_mode_errors():
    config = SimulatorConfig(post_process_mode="unknown")

    try:
        _post_process(np.array([1.0, -1.0]), config)
    except ValueError as exc:
        assert "Unknown post_process_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown post_process_mode")


def test_alpha_input_panel_applies_pasteurization_before_alpha_evaluation():
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    symbols = np.array(["A", "B", "C"])
    fields = {
        "open": np.ones((2, 3)),
        "high": np.ones((2, 3)),
        "low": np.ones((2, 3)),
        "close": np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]]),
        "volume": np.ones((2, 3)),
        "returns": np.ones((2, 3)),
        "vwap": np.ones((2, 3)),
        "cumulative_liq": np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]),
    }
    panel = MarketPanel(dates=dates, symbols=symbols, fields=fields)
    config = SimulatorConfig(universe="Top2", pasteurization=True)

    actual = _alpha_input_panel(panel, config)

    np.testing.assert_allclose(actual.field("close"), [[1.0, 10.0, np.nan], [2.0, 20.0, np.nan]])
    np.testing.assert_allclose(actual.field("cumulative_liq"), fields["cumulative_liq"])
