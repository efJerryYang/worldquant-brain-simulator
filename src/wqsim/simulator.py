from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .alphas import ALPHAS, AlphaContext
from .config import SimulatorConfig
from .data import MarketPanel, compute_universe_mask, load_market_panel, pasteurize_panel
from .fast_expr import scale


POST_PROCESS_MODES = (
    "legacy",
    "renormalize_after_truncation",
    "normalize_then_truncate",
    "no_truncation",
)


@dataclass(frozen=True)
class SimulationResult:
    alpha_name: str
    dates: np.ndarray
    pnl: np.ndarray
    turnover: np.ndarray
    cumulative_pnl: np.ndarray
    metrics: dict[str, Any]
    plot_path: Path | None = None
    metrics_path: Path | None = None

    @property
    def final_pnl(self) -> float:
        if self.cumulative_pnl.size == 0:
            return 0.0
        return float(self.cumulative_pnl[-1])


def run_simulation(
    config: SimulatorConfig,
    alpha_name: str = "eg_alpha3",
    panel: MarketPanel | None = None,
) -> SimulationResult:
    if alpha_name not in ALPHAS:
        available = ", ".join(sorted(ALPHAS))
        raise ValueError(f"Unknown alpha '{alpha_name}'. Available alphas: {available}")

    panel = panel or load_market_panel(config)
    alpha_panel = _alpha_input_panel(panel, config)
    context = AlphaContext.from_panel(alpha_panel)
    alpha = ALPHAS[alpha_name](context)
    pnl_dates, pnl, turnover = _simulate_pnl(panel, alpha, config)
    cumulative = np.cumsum(pnl)
    metrics = _build_metrics(config, panel, alpha_name, alpha, pnl_dates, pnl, turnover, cumulative)
    plot_path = _plot_result(config, alpha_name, pnl_dates, cumulative) if config.plot else None
    metrics_path = _write_metrics_sidecar(plot_path, config, metrics) if plot_path else None
    return SimulationResult(
        alpha_name, pnl_dates, pnl, turnover, cumulative, metrics, plot_path, metrics_path
    )


def _simulate_pnl(
    panel: MarketPanel, alpha: np.ndarray, config: SimulatorConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_idx = _simulation_start_index(panel, config)
    if start_idx <= config.delay:
        start_idx = config.delay + 1

    pnl: list[float] = []
    turnover: list[float] = []
    dates: list[np.datetime64] = []
    returns = panel.field("returns")
    cumulative_liq = panel.field("cumulative_liq")
    previous_weights: np.ndarray | None = None

    for today_idx in range(start_idx, len(panel.dates)):
        signal_idx = today_idx - config.delay
        row = alpha[signal_idx].copy()
        universe = _universe_mask(cumulative_liq[signal_idx], config.universe_size)
        row[~universe] = np.nan
        weights = _post_process(row, config)
        today_returns = returns[today_idx]
        valid = np.isfinite(today_returns) & np.isfinite(weights)
        pnl.append(float(np.sum(weights[valid] * today_returns[valid]) * config.booksize / 100.0))
        turnover.append(
            0.0 if previous_weights is None else float(np.sum(np.abs(weights - previous_weights)))
        )
        previous_weights = weights
        dates.append(panel.dates[today_idx])

    return (
        np.array(dates, dtype="datetime64[D]"),
        np.array(pnl, dtype=np.float64),
        np.array(turnover, dtype=np.float64),
    )


def _alpha_input_panel(panel: MarketPanel, config: SimulatorConfig) -> MarketPanel:
    if not config.pasteurization:
        return panel
    universe_mask = compute_universe_mask(panel.field("cumulative_liq"), config.universe_size)
    return pasteurize_panel(panel, universe_mask)


def _post_process(alpha_row: np.ndarray, config: SimulatorConfig) -> np.ndarray:
    values = alpha_row.reshape(1, -1).astype(np.float64)
    if not np.isfinite(values).any():
        return np.zeros(alpha_row.shape, dtype=np.float64)

    mode = config.post_process_mode.lower()
    if mode not in POST_PROCESS_MODES:
        available = ", ".join(POST_PROCESS_MODES)
        raise ValueError(
            f"Unknown post_process_mode '{config.post_process_mode}'. Available: {available}"
        )

    values = _neutralize(values, config)
    if mode == "no_truncation" or config.truncation <= 0:
        return scale(values)[0]
    if mode == "normalize_then_truncate":
        weights = scale(values)
        return scale(np.clip(weights, -config.truncation, config.truncation))[0]

    values = np.clip(values, -config.truncation, config.truncation)
    if mode == "renormalize_after_truncation":
        values = _neutralize(values, config)
    return scale(values)[0]


def _universe_mask(cumulative_liq: np.ndarray, universe_size: int) -> np.ndarray:
    valid = np.isfinite(cumulative_liq)
    mask = np.zeros(cumulative_liq.shape, dtype=bool)
    if not valid.any():
        return mask
    valid_idx = np.flatnonzero(valid)
    top_count = min(universe_size, valid_idx.size)
    ranked = valid_idx[np.argpartition(cumulative_liq[valid_idx], -top_count)[-top_count:]]
    mask[ranked] = True
    return mask


def _simulation_start_index(panel: MarketPanel, config: SimulatorConfig) -> int:
    start_date = config.simulation_start_date
    if start_date is None:
        return 0
    return int(np.searchsorted(panel.dates, np.datetime64(start_date, "D")))


def _plot_result(
    config: SimulatorConfig,
    alpha_name: str,
    dates: np.ndarray,
    cumulative_pnl: np.ndarray,
) -> Path | None:
    if dates.size == 0:
        return None
    config.output_dir.mkdir(parents=True, exist_ok=True)
    start = str(dates[0])
    end = str(dates[-1])
    path = config.output_dir / f"PnL_{alpha_name}_{start}_{end}.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates.astype("datetime64[D]").astype(object), cumulative_pnl)
    ax.set_title(f"{alpha_name} cumulative PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _build_metrics(
    config: SimulatorConfig,
    panel: MarketPanel,
    alpha_name: str,
    alpha: np.ndarray,
    pnl_dates: np.ndarray,
    pnl: np.ndarray,
    turnover: np.ndarray,
    cumulative_pnl: np.ndarray,
) -> dict[str, Any]:
    finite_alpha = np.isfinite(alpha)
    return {
        "alpha_name": alpha_name,
        "sample": config.sample,
        "load_start_date": str(panel.dates[0]) if panel.dates.size else None,
        "load_end_date": str(panel.dates[-1]) if panel.dates.size else None,
        "simulation_start_date": str(pnl_dates[0]) if pnl_dates.size else None,
        "simulation_end_date": str(pnl_dates[-1]) if pnl_dates.size else None,
        "panel_rows": int(panel.dates.size),
        "panel_symbols": int(panel.symbols.size),
        "alpha_nan_count": int(alpha.size - finite_alpha.sum()),
        "alpha_finite_count": int(finite_alpha.sum()),
        "traded_days": int(pnl.size),
        "universe_size": int(config.universe_size),
        "pasteurization": bool(config.pasteurization),
        "post_process_mode": config.post_process_mode,
        "booksize": float(config.booksize),
        "max_drawdown": _finite_float(
            np.min(cumulative_pnl - np.maximum.accumulate(cumulative_pnl))
        )
        if cumulative_pnl.size
        else 0.0,
        "mean_abs_net_exposure": _finite_float(
            _mean_abs_net_exposure(config, panel, alpha, pnl_dates)
        ),
        "final_pnl": _finite_float(cumulative_pnl[-1]) if cumulative_pnl.size else 0.0,
        "mean_daily_pnl": _finite_float(np.mean(pnl)) if pnl.size else 0.0,
        "std_daily_pnl": _finite_float(np.std(pnl)) if pnl.size else 0.0,
        "min_daily_pnl": _finite_float(np.min(pnl)) if pnl.size else 0.0,
        "max_daily_pnl": _finite_float(np.max(pnl)) if pnl.size else 0.0,
        "mean_turnover": _finite_float(np.mean(turnover)) if turnover.size else 0.0,
        "max_turnover": _finite_float(np.max(turnover)) if turnover.size else 0.0,
    }


def _write_metrics_sidecar(
    plot_path: Path, config: SimulatorConfig, metrics: dict[str, Any]
) -> Path:
    path = plot_path.with_suffix(".json")
    payload = {
        "config": _serialize_config(config),
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _serialize_config(config: SimulatorConfig) -> dict[str, Any]:
    return {
        "database_path": str(config.database_path),
        "table": config.table,
        "sample": config.sample,
        "universe": config.universe,
        "delay": config.delay,
        "neutralization": config.neutralization,
        "pasteurization": config.pasteurization,
        "post_process_mode": config.post_process_mode,
        "truncation": config.truncation,
        "booksize": config.booksize,
        "start_date": _serialize_date(config.start_date),
        "end_date": _serialize_date(config.end_date),
        "load_start_date": _serialize_date(config.load_start_date),
        "output_dir": str(config.output_dir),
        "plot": config.plot,
        "min_symbols_per_day": config.min_symbols_per_day,
    }


def _serialize_date(value: date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _finite_float(value: float | np.floating) -> float:
    as_float = float(value)
    return as_float if np.isfinite(as_float) else 0.0


def _neutralize(values: np.ndarray, config: SimulatorConfig) -> np.ndarray:
    if config.neutralization.lower() != "market":
        return values
    with np.errstate(all="ignore"):
        return values - np.nanmean(values, axis=1, keepdims=True)


def _mean_abs_net_exposure(
    config: SimulatorConfig, panel: MarketPanel, alpha: np.ndarray, pnl_dates: np.ndarray
) -> float:
    if pnl_dates.size == 0:
        return 0.0
    start_idx = _simulation_start_index(panel, config)
    if start_idx <= config.delay:
        start_idx = config.delay + 1
    cumulative_liq = panel.field("cumulative_liq")
    exposures: list[float] = []
    for today_idx in range(start_idx, len(panel.dates)):
        signal_idx = today_idx - config.delay
        row = alpha[signal_idx].copy()
        row[~_universe_mask(cumulative_liq[signal_idx], config.universe_size)] = np.nan
        exposures.append(float(abs(_post_process(row, config).sum())))
    return float(np.mean(exposures)) if exposures else 0.0
