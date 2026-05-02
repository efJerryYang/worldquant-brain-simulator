import numpy as np

from wqsim.config import SimulatorConfig
from wqsim.data import MarketPanel
from wqsim.simulator import run_simulation


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
    assert np.isfinite(result.cumulative_pnl).all()
