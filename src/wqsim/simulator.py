from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .alphas import ALPHAS, AlphaContext
from .config import SimulatorConfig
from .data import MarketPanel, load_market_panel
from .fast_expr import scale


@dataclass(frozen=True)
class SimulationResult:
    alpha_name: str
    dates: np.ndarray
    pnl: np.ndarray
    cumulative_pnl: np.ndarray
    plot_path: Path | None = None

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
    context = AlphaContext.from_panel(panel)
    alpha = ALPHAS[alpha_name](context)
    pnl_dates, pnl = _simulate_pnl(panel, alpha, config)
    cumulative = np.cumsum(pnl)
    plot_path = _plot_result(config, alpha_name, pnl_dates, cumulative) if config.plot else None
    return SimulationResult(alpha_name, pnl_dates, pnl, cumulative, plot_path)


def _simulate_pnl(
    panel: MarketPanel, alpha: np.ndarray, config: SimulatorConfig
) -> tuple[np.ndarray, np.ndarray]:
    start_idx = _simulation_start_index(panel, config)
    if start_idx <= config.delay:
        start_idx = config.delay + 1

    pnl: list[float] = []
    dates: list[np.datetime64] = []
    returns = panel.field("returns")
    cumulative_liq = panel.field("cumulative_liq")

    for today_idx in range(start_idx, len(panel.dates)):
        signal_idx = today_idx - config.delay
        row = alpha[signal_idx].copy()
        universe = _universe_mask(cumulative_liq[signal_idx], config.universe_size)
        row[~universe] = np.nan
        weights = _post_process(row, config)
        today_returns = returns[today_idx]
        valid = np.isfinite(today_returns) & np.isfinite(weights)
        pnl.append(float(np.sum(weights[valid] * today_returns[valid]) * config.booksize / 100.0))
        dates.append(panel.dates[today_idx])

    return np.array(dates, dtype="datetime64[D]"), np.array(pnl, dtype=np.float64)


def _post_process(alpha_row: np.ndarray, config: SimulatorConfig) -> np.ndarray:
    values = alpha_row.reshape(1, -1).astype(np.float64)
    if not np.isfinite(values).any():
        return np.zeros(alpha_row.shape, dtype=np.float64)
    if config.neutralization.lower() == "market":
        with np.errstate(all="ignore"):
            values = values - np.nanmean(values, axis=1, keepdims=True)
    if config.truncation > 0:
        values = np.clip(values, -config.truncation, config.truncation)
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
