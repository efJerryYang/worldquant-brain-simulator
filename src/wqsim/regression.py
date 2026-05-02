from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .alphas import ALPHAS, AlphaContext
from .data import MarketPanel


@dataclass(frozen=True)
class SyntheticPanelSpec:
    rows: int = 96
    cols: int = 12
    seed: int = 20260502
    nan_rate: float = 0.03


def make_synthetic_panel(spec: SyntheticPanelSpec = SyntheticPanelSpec()) -> MarketPanel:
    rng = np.random.default_rng(spec.seed)
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + spec.rows)
    symbols = np.array([f"S{i:04d}" for i in range(spec.cols)])

    base = 50.0 + np.cumsum(rng.normal(0.0, 0.8, size=(spec.rows, spec.cols)), axis=0)
    close = np.maximum(base, 1.0)
    open_ = close + rng.normal(0.0, 0.3, size=close.shape)
    high = np.maximum(open_, close) + rng.random(close.shape)
    low = np.minimum(open_, close) - rng.random(close.shape)
    volume = rng.lognormal(mean=12.0, sigma=0.35, size=close.shape)
    returns = np.full(close.shape, np.nan)
    returns[1:] = (close[1:] / close[:-1] - 1.0) * 100.0
    vwap = (open_ + high + low + close) / 4.0

    _inject_edges(close, open_, high, low, volume, returns, vwap)
    for field in (open_, high, low, close, volume, returns, vwap):
        _inject_nans(field, rng, spec.nan_rate)

    fields = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "returns": returns,
        "vwap": vwap,
        "cumulative_liq": np.tile(np.linspace(1.0, float(spec.cols), spec.cols), (spec.rows, 1)),
    }
    return MarketPanel(dates=dates, symbols=symbols, fields=fields)


def alpha_output_digest(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    canonical = np.nan_to_num(
        values.astype(np.float64), nan=1.23456789e308, posinf=9.87654321e307, neginf=-9.87654321e307
    )
    return {
        "shape": list(values.shape),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(values).sum()),
        "posinf_count": int(np.isposinf(values).sum()),
        "neginf_count": int(np.isneginf(values).sum()),
        "mean": _finite_float(np.nanmean(values)) if finite.any() else 0.0,
        "std": _finite_float(np.nanstd(values)) if finite.any() else 0.0,
        "min": _finite_float(np.nanmin(values)) if finite.any() else 0.0,
        "max": _finite_float(np.nanmax(values)) if finite.any() else 0.0,
        "sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
    }


def alpha_regression_snapshot(panel: MarketPanel) -> dict[str, dict[str, Any]]:
    context = AlphaContext.from_panel(panel)
    return {name: alpha_output_digest(fn(context)) for name, fn in sorted(ALPHAS.items())}


def benchmark_alphas(panel: MarketPanel, repeat: int = 1) -> list[dict[str, Any]]:
    context = AlphaContext.from_panel(panel)
    rows: list[dict[str, Any]] = []
    for name, fn in sorted(ALPHAS.items()):
        timings = []
        digest: dict[str, Any] | None = None
        for _ in range(repeat):
            start = perf_counter()
            values = fn(context)
            timings.append(perf_counter() - start)
            digest = alpha_output_digest(values)
        rows.append(
            {
                "alpha": name,
                "seconds_min": min(timings),
                "seconds_mean": sum(timings) / len(timings),
                "repeat": repeat,
                "digest": digest,
            }
        )
    return rows


def benchmark_report_json(panel: MarketPanel, repeat: int = 1) -> str:
    payload = {
        "panel_rows": int(panel.dates.size),
        "panel_symbols": int(panel.symbols.size),
        "repeat": repeat,
        "alphas": benchmark_alphas(panel, repeat=repeat),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _inject_edges(*fields: np.ndarray) -> None:
    close = fields[3]
    volume = fields[4]
    returns = fields[5]
    close[:, 0] = 42.0
    close[10:15, 1] = close[10, 1]
    volume[20:25, 2] = 0.0
    returns[30:35, 3] = -5.0
    returns[35:40, 3] = 5.0


def _inject_nans(values: np.ndarray, rng: np.random.Generator, nan_rate: float) -> None:
    if nan_rate <= 0:
        return
    mask = rng.random(values.shape) < nan_rate
    values[mask] = np.nan


def _finite_float(value: float | np.floating) -> float:
    as_float = float(value)
    return as_float if np.isfinite(as_float) else 0.0
