from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

Array = np.ndarray


def as_float_array(x: Array) -> Array:
    return np.asarray(x, dtype=np.float64)


def rank(x: Array) -> Array:
    """Cross-sectional rank per date, normalized to 0..1 and ignoring NaNs."""
    values = as_float_array(x)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for row_idx, row in enumerate(values):
        valid = np.isfinite(row)
        count = int(valid.sum())
        if count == 0:
            continue
        if count == 1:
            out[row_idx, valid] = 0.0
            continue
        out[row_idx, valid] = (rankdata(row[valid], method="average") - 1.0) / (count - 1.0)
    return out


def delay(x: Array, period: int = 1) -> Array:
    values = as_float_array(x)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if period <= 0:
        return values.copy()
    out[period:] = values[:-period]
    return out


def delta(x: Array, period: int = 1) -> Array:
    return as_float_array(x) - delay(x, period)


def signed_power(x: Array, exponent: float) -> Array:
    values = as_float_array(x)
    return np.sign(values) * np.power(np.abs(values), exponent)


def safe_div(numerator: Array, denominator: Array) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = as_float_array(numerator) / as_float_array(denominator)
    out[~np.isfinite(out)] = np.nan
    return out


def log(x: Array) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(as_float_array(x))
    out[~np.isfinite(out)] = np.nan
    return out


def scale(x: Array, k: float = 1.0) -> Array:
    values = as_float_array(x)
    denom = np.nansum(np.abs(values), axis=1, keepdims=True)
    out = np.zeros_like(values)
    valid = denom[:, 0] > 0
    out[valid] = values[valid] * k / denom[valid]
    out[~np.isfinite(out)] = 0.0
    return out


def ts_sum(x: Array, window: int) -> Array:
    return _rolling_reduce(x, window, np.nansum)


def ts_mean(x: Array, window: int) -> Array:
    return _rolling_reduce(x, window, np.nanmean)


def ts_std(x: Array, window: int) -> Array:
    return _rolling_reduce(x, window, np.nanstd)


def ts_min(x: Array, window: int) -> Array:
    return _rolling_reduce(x, window, np.nanmin)


def ts_max(x: Array, window: int) -> Array:
    return _rolling_reduce(x, window, np.nanmax)


def ts_rank(x: Array, window: int) -> Array:
    values = as_float_array(x)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for end in range(window - 1, values.shape[0]):
        chunk = values[end - window + 1 : end + 1]
        last = chunk[-1]
        for col in range(values.shape[1]):
            series = chunk[:, col]
            valid = np.isfinite(series)
            if not np.isfinite(last[col]) or not valid.any():
                continue
            count = int(valid.sum())
            if count == 1:
                out[end, col] = 0.0
            else:
                out[end, col] = (rankdata(series[valid], method="average")[-1] - 1.0) / (
                    count - 1.0
                )
    return out


def ts_argmax(x: Array, window: int) -> Array:
    return _rolling_arg(x, window, np.nanargmax)


def ts_argmin(x: Array, window: int) -> Array:
    return _rolling_arg(x, window, np.nanargmin)


def correlation(x: Array, y: Array, window: int) -> Array:
    x_values = as_float_array(x)
    y_values = as_float_array(y)
    out = np.full(x_values.shape, np.nan, dtype=np.float64)
    for end in range(window - 1, x_values.shape[0]):
        xs = x_values[end - window + 1 : end + 1]
        ys = y_values[end - window + 1 : end + 1]
        for col in range(x_values.shape[1]):
            xv = xs[:, col]
            yv = ys[:, col]
            valid = np.isfinite(xv) & np.isfinite(yv)
            if valid.sum() < 2:
                continue
            x_valid = xv[valid]
            y_valid = yv[valid]
            if np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
                continue
            out[end, col] = np.corrcoef(x_valid, y_valid)[0, 1]
    return out


def covariance(x: Array, y: Array, window: int) -> Array:
    x_values = as_float_array(x)
    y_values = as_float_array(y)
    out = np.full(x_values.shape, np.nan, dtype=np.float64)
    for end in range(window - 1, x_values.shape[0]):
        xs = x_values[end - window + 1 : end + 1]
        ys = y_values[end - window + 1 : end + 1]
        for col in range(x_values.shape[1]):
            xv = xs[:, col]
            yv = ys[:, col]
            valid = np.isfinite(xv) & np.isfinite(yv)
            if valid.sum() < 2:
                continue
            out[end, col] = np.cov(xv[valid], yv[valid])[0, 1]
    return out


def decay_linear(x: Array, window: int) -> Array:
    values = as_float_array(x)
    weights = np.arange(1, window + 1, dtype=np.float64)
    weights /= weights.sum()
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for end in range(window - 1, values.shape[0]):
        chunk = values[end - window + 1 : end + 1]
        finite = np.isfinite(chunk)
        weighted = np.where(finite, chunk, 0.0) * weights[:, None]
        denom = np.where(finite, weights[:, None], 0.0).sum(axis=0)
        valid = denom > 0
        out[end, valid] = weighted[:, valid].sum(axis=0) / denom[valid]
    return out


def _rolling_reduce(x: Array, window: int, reducer) -> Array:
    values = as_float_array(x)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    with np.errstate(all="ignore"):
        for end in range(window - 1, values.shape[0]):
            chunk = values[end - window + 1 : end + 1]
            has_value = np.isfinite(chunk).any(axis=0)
            if has_value.any():
                out[end, has_value] = reducer(chunk[:, has_value], axis=0)
    return out


def _rolling_arg(x: Array, window: int, reducer) -> Array:
    values = as_float_array(x)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for end in range(window - 1, values.shape[0]):
        chunk = values[end - window + 1 : end + 1]
        for col in range(values.shape[1]):
            series = chunk[:, col]
            if np.isfinite(series).any():
                out[end, col] = float(reducer(series) + 1)
    return out
