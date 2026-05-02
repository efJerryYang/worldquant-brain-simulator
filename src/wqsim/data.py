from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .config import SimulatorConfig
from .fast_expr import log, ts_sum


@dataclass(frozen=True)
class MarketPanel:
    dates: np.ndarray
    symbols: np.ndarray
    fields: dict[str, np.ndarray]

    def field(self, name: str) -> np.ndarray:
        return self.fields[name]


PASTEURIZED_FIELDS = ("open", "high", "low", "close", "volume", "returns", "vwap", "typical_price")


def pasteurize_panel(panel: MarketPanel, universe_mask: np.ndarray) -> MarketPanel:
    fields = {}
    for name, values in panel.fields.items():
        if name in PASTEURIZED_FIELDS:
            masked = values.copy()
            masked[~universe_mask] = np.nan
            fields[name] = masked
        else:
            fields[name] = values
    return MarketPanel(dates=panel.dates, symbols=panel.symbols, fields=fields)


def compute_universe_mask(cumulative_liq: np.ndarray, universe_size: int) -> np.ndarray:
    mask = np.zeros(cumulative_liq.shape, dtype=bool)
    for row_idx, row in enumerate(cumulative_liq):
        valid = np.isfinite(row)
        if not valid.any():
            continue
        valid_idx = np.flatnonzero(valid)
        top_count = min(universe_size, valid_idx.size)
        ranked = valid_idx[np.argpartition(row[valid_idx], -top_count)[-top_count:]]
        mask[row_idx, ranked] = True
    return mask


def load_market_panel(config: SimulatorConfig) -> MarketPanel:
    start, end = config.load_range
    raw = _read_sqlite(config.database_path, config.table, start, end)
    prepared = _prepare_frame(raw, config.min_symbols_per_day)
    panel = _dense_panel(prepared)
    return _with_cumulative_liquidity(panel)


def _read_sqlite(database_path: Path, table: str, start: date, end: date) -> pl.DataFrame:
    query = f"""
        SELECT
            symbol,
            timestamp_ms,
            open,
            high,
            low,
            close,
            volume,
            percent AS returns,
            amount
        FROM {table}
        WHERE timestamp_ms BETWEEN ? AND ?
    """
    start_ms = int(np.datetime64(start, "ms").astype("int64"))
    end_ms = int(np.datetime64(end, "ms").astype("int64"))
    with sqlite3.connect(database_path) as connection:
        return pl.read_database(
            query, connection, execute_options={"parameters": [start_ms, end_ms]}
        )


def _prepare_frame(df: pl.DataFrame, min_symbols_per_day: int) -> pl.DataFrame:
    frame = (
        df.with_columns(
            pl.from_epoch("timestamp_ms", time_unit="ms").dt.date().alias("date"),
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("typical_price"),
            pl.when(pl.col("amount") == 0).then(None).otherwise(pl.col("amount")).alias("amount"),
        )
        .sort(["symbol", "date"])
        .with_columns(pl.col("amount").interpolate().over("symbol").alias("amount"))
        .with_columns((pl.col("amount") / pl.col("volume")).alias("vwap"))
        .sort(["date", "symbol"])
    )
    valid_dates = (
        frame.group_by("date")
        .agg(pl.col("symbol").count().alias("symbol_count"))
        .filter(pl.col("symbol_count") >= min_symbols_per_day)
        .select("date")
    )
    return frame.join(valid_dates, on="date", how="inner").drop_nulls()


def _dense_panel(df: pl.DataFrame) -> MarketPanel:
    dates = np.sort(df.get_column("date").to_numpy()).astype("datetime64[D]")
    dates = np.unique(dates)
    symbols = np.array(sorted(df.get_column("symbol").unique().to_list()), dtype=str)

    row_index = np.searchsorted(dates, df.get_column("date").to_numpy().astype("datetime64[D]"))
    symbol_to_col = {symbol: idx for idx, symbol in enumerate(symbols)}
    col_index = np.fromiter(
        (symbol_to_col[symbol] for symbol in df.get_column("symbol").to_list()),
        dtype=np.int64,
        count=df.height,
    )

    fields: dict[str, np.ndarray] = {}
    for name in ("open", "high", "low", "close", "volume", "returns", "vwap", "typical_price"):
        matrix = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
        matrix[row_index, col_index] = df.get_column(name).to_numpy().astype(np.float64)
        fields[name] = matrix
    return MarketPanel(dates=dates, symbols=symbols, fields=fields)


def _with_cumulative_liquidity(panel: MarketPanel) -> MarketPanel:
    close = panel.field("close")
    volume = panel.field("volume")
    liquidity = log(volume * close)
    cumulative_liq = ts_sum(liquidity, 90)

    valid_rows = np.isfinite(cumulative_liq).any(axis=1)
    fields = {name: values[valid_rows] for name, values in panel.fields.items()}
    fields["liquidity"] = liquidity[valid_rows]
    fields["cumulative_liq"] = cumulative_liq[valid_rows]
    return MarketPanel(dates=panel.dates[valid_rows], symbols=panel.symbols, fields=fields)
