from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .data import MarketPanel
from .fast_expr import (
    correlation,
    delay,
    delta,
    log,
    rank,
    safe_div,
    ts_argmax,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std,
    ts_sum,
)


@dataclass(frozen=True)
class AlphaContext:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    returns: np.ndarray
    vwap: np.ndarray

    @classmethod
    def from_panel(cls, panel: MarketPanel) -> "AlphaContext":
        return cls(
            open=panel.field("open"),
            high=panel.field("high"),
            low=panel.field("low"),
            close=panel.field("close"),
            volume=panel.field("volume"),
            returns=panel.field("returns"),
            vwap=panel.field("vwap"),
        )


AlphaFn = Callable[[AlphaContext], np.ndarray]


def eg_alpha(ctx: AlphaContext) -> np.ndarray:
    adv30 = safe_div(ts_sum(ctx.volume, 30), 30.0)
    return -rank(delta(ctx.close, 2)) * rank(safe_div(ctx.volume, adv30))


def eg_alpha2(ctx: AlphaContext) -> np.ndarray:
    return -(ctx.close - ts_mean(ctx.close, 5))


def eg_alpha3(ctx: AlphaContext) -> np.ndarray:
    return rank(-(ctx.close - ts_mean(ctx.close, 10)))


def alpha001(ctx: AlphaContext) -> np.ndarray:
    inner = np.where(ctx.returns < 0, ts_std(ctx.returns, 20), ctx.close)
    return rank(ts_argmax(inner**2, 5)) - 0.5


def alpha002(ctx: AlphaContext) -> np.ndarray:
    return -correlation(
        rank(delta(log(ctx.volume), 2)), rank(safe_div(ctx.close - ctx.open, ctx.open)), 6
    )


def alpha003(ctx: AlphaContext) -> np.ndarray:
    return -correlation(rank(ctx.open), rank(ctx.volume), 10)


def alpha004(ctx: AlphaContext) -> np.ndarray:
    return -ts_rank(rank(ctx.low), 9)


def alpha005(ctx: AlphaContext) -> np.ndarray:
    return rank(ctx.open - safe_div(ts_sum(ctx.vwap, 10), 10.0)) * -np.abs(
        rank(ctx.close - ctx.vwap)
    )


def alpha006(ctx: AlphaContext) -> np.ndarray:
    return -correlation(ctx.open, ctx.volume, 10)


def alpha007(ctx: AlphaContext) -> np.ndarray:
    adv20 = ts_mean(ctx.volume, 20)
    close_delta = delta(ctx.close, 7)
    active = -ts_rank(np.abs(close_delta), 60) * np.sign(close_delta)
    return np.where(adv20 < ctx.volume, active, -1.0)


def alpha008(ctx: AlphaContext) -> np.ndarray:
    product = ts_sum(ctx.open, 5) * ts_sum(ctx.returns, 5)
    return -rank(product - delay(product, 10))


def alpha009(ctx: AlphaContext) -> np.ndarray:
    close_delta = delta(ctx.close, 1)
    trend = (ts_min(close_delta, 5) > 0) | (ts_max(close_delta, 5) < 0)
    return np.where(trend, close_delta, -close_delta)


def alpha010(ctx: AlphaContext) -> np.ndarray:
    close_delta = delta(ctx.close, 1)
    trend = (ts_min(close_delta, 4) > 0) | (ts_max(close_delta, 4) < 0)
    return rank(np.where(trend, close_delta, -close_delta))


ALPHAS: dict[str, AlphaFn] = {
    "eg_alpha": eg_alpha,
    "eg_alpha2": eg_alpha2,
    "eg_alpha3": eg_alpha3,
    "alpha001": alpha001,
    "alpha002": alpha002,
    "alpha003": alpha003,
    "alpha004": alpha004,
    "alpha005": alpha005,
    "alpha006": alpha006,
    "alpha007": alpha007,
    "alpha008": alpha008,
    "alpha009": alpha009,
    "alpha010": alpha010,
}
