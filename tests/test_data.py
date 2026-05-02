import sqlite3
from datetime import date, timedelta

import numpy as np

from wqsim.config import SimulatorConfig
from wqsim.data import MarketPanel, compute_universe_mask, load_market_panel, pasteurize_panel


def test_load_market_panel_from_sqlite_fixture(tmp_path):
    database_path = tmp_path / "fixture.db"
    _create_fixture_database(database_path)
    config = SimulatorConfig(
        database_path=database_path,
        load_start_date=date(2020, 1, 1),
        end_date=date(2020, 4, 3),
        min_symbols_per_day=2,
    )

    panel = load_market_panel(config)

    np.testing.assert_array_equal(
        panel.dates,
        np.array(["2020-03-30", "2020-03-31", "2020-04-01"], dtype="datetime64[D]"),
    )
    np.testing.assert_array_equal(panel.symbols, np.array(["AAA", "BBB"]))
    np.testing.assert_allclose(panel.field("vwap")[:, 0], 10.0)
    np.testing.assert_allclose(panel.field("vwap")[:, 1], 20.0)
    np.testing.assert_allclose(panel.field("cumulative_liq")[:, 0], 90 * np.log(1_000.0))
    np.testing.assert_allclose(panel.field("cumulative_liq")[:, 1], 90 * np.log(4_000.0))


def test_pasteurize_panel_masks_inputs_outside_universe():
    fields = {
        "close": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "volume": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        "cumulative_liq": np.array([[1.0, 3.0, 2.0], [4.0, 1.0, 5.0]]),
    }
    panel = MarketPanel(
        dates=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        symbols=np.array(["A", "B", "C"]),
        fields=fields,
    )

    mask = compute_universe_mask(panel.field("cumulative_liq"), universe_size=2)
    actual = pasteurize_panel(panel, mask)

    np.testing.assert_array_equal(mask, [[False, True, True], [True, False, True]])
    np.testing.assert_allclose(actual.field("close"), [[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0]])
    np.testing.assert_allclose(actual.field("volume"), [[np.nan, 20.0, 30.0], [40.0, np.nan, 60.0]])
    np.testing.assert_allclose(actual.field("cumulative_liq"), fields["cumulative_liq"])


def _create_fixture_database(database_path):
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE stock_data_US (
                timestamp_ms INTEGER,
                volume INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                percent REAL,
                amount REAL,
                symbol TEXT
            )
            """
        )
        start = date(2020, 1, 1)
        for offset in range(92):
            day = start + timedelta(days=offset)
            aaa_amount = 0.0 if offset == 90 else 1_000.0
            _insert_row(connection, day, "AAA", volume=100, close=10.0, amount=aaa_amount)
            _insert_row(connection, day, "BBB", volume=200, close=20.0, amount=4_000.0)
        _insert_row(connection, date(2020, 4, 2), "AAA", volume=100, close=10.0, amount=1_000.0)


def _insert_row(connection, day, symbol, volume, close, amount):
    timestamp_ms = int(np.datetime64(day, "ms").astype("int64"))
    connection.execute(
        """
        INSERT INTO stock_data_US (
            timestamp_ms, volume, open, high, low, close, percent, amount, symbol
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp_ms,
            volume,
            close - 0.5,
            close + 0.5,
            close - 1.0,
            close,
            0.1,
            amount,
            symbol,
        ),
    )
