# Accuracy Baseline

This simulator is an independent implementation, not an exact clone of the
WorldQuant BRAIN platform. The platform image in `docs/insample_platform.png`
is the only external reference currently available, and it may differ because
of data, operator semantics, platform settings, or survivorship/universe rules.

The goal of the current baseline is internal correctness: every behavior below
should be deterministic, tested where practical, and reflected in JSON metrics
written beside generated figures.

## Data Semantics

- Source data is read from SQLite with Polars.
- `timestamp_ms` is converted to calendar dates at day resolution.
- `returns` comes from the database `percent` column and is treated as percent,
  so PnL uses `weights * returns * booksize / 100`.
- `vwap` is `amount / volume` after replacing zero `amount` values with null and
  interpolating `amount` within each symbol.
- `liquidity` is `log(volume * close)`.
- `cumulative_liq` is a 90-day rolling sum of liquidity.
- Days with fewer than `min_symbols_per_day` rows are removed before panel
  construction.

## Fast Expression Semantics

- All alpha inputs are dense NumPy arrays shaped `(dates, symbols)`.
- Rolling operators use full windows: output is `NaN` until `window` rows are
  available.
- Rolling reductions ignore `NaN` values inside an available full window.
- Cross-sectional `rank` is computed per date, ignores `NaN`, uses average rank
  for ties, and normalizes finite values to `[0, 1]`.
- A row with one finite ranked value receives rank `0.0`.
- `delay(x, n)` shifts along the date axis and fills the first `n` rows with
  `NaN`.
- `delta(x, n)` is `x - delay(x, n)`.
- Rolling correlation and covariance are computed per symbol using finite pairs
  only; fewer than two finite pairs produce `NaN`.

## Simulation Semantics

- Default delay is one day: alpha from `today - delay` trades today's returns.
- When `pasteurization` is on, alpha input fields are set to `NaN` outside the
  selected universe before alpha evaluation.
- When `decay > 1`, the evaluated alpha matrix is smoothed with a linear
  time-series decay before the trading delay is applied.
- Universe selection uses the signal day's top `N` symbols by `cumulative_liq`.
- Market neutralization subtracts the cross-sectional mean from the selected
  alpha row.
- The default `legacy` post-processing mode applies market neutralization,
  truncation, and final normalization in that order. This preserves the legacy
  engine behavior but can reintroduce net exposure after clipping.
- Additional diagnostic modes are available with `--post-process-mode`:
  `renormalize_after_truncation`, `normalize_then_truncate`, and
  `no_truncation`.
- Final weights are scaled so `sum(abs(weights)) == 1` when any finite signal is
  available; empty or zero rows produce all-zero weights.
- Daily turnover is `sum(abs(weights_today - weights_previous_day))`; first
  traded day turnover is `0.0`.
- Generated JSON metrics are diagnostic artifacts, not claims of platform
  parity.
- `compare-postprocess` runs all post-processing modes on one loaded panel and
  prints comparable PnL, drawdown, exposure, and turnover metrics.

## Known Limits

- Sector, industry, subindustry, unit handling, and platform NaN-handling modes
  are not implemented.
- Alpha101 coverage is intentionally partial.
- The current rolling implementations prioritize clarity and testability over
  maximum speed.
- The platform comparison is visual and directional until exact platform inputs
  and settings are available.
