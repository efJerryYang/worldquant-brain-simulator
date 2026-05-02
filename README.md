# WorldQuant BRAIN Simulator

<!-- > **Help Wanted**: See issue [#3](https://github.com/efJerryYang/worldquant-brain-simulator/issues/3) for more details. -->

## Introduction

This is a simulator to help with offline backtesting for WorldQuant BRAIN-style alphas.
It uses Polars for data loading, dense NumPy panels for market data, and batch alpha evaluation.

## Demo

`eg_alpha3` on the `insample` range:

![insample](./docs/insample.png)

Platform reference:

![insample_platform](./docs/insample_platform.png)
<!-- 
> Totol time cost is about 20 minutes on one core, target is 5 minutes per core. -->

## Project Structure

```sh
worldquant-brain-simulator/
|-- data
|   `-- stock_snowball_us.db
|-- docs
|-- LICENSE
|-- README.md
|-- pyproject.toml
|-- uv.lock
|-- tests
`-- src
    |-- wqsim
    |   |-- alphas.py
    |   |-- cli.py
    |   |-- config.py
    |   |-- data.py
    |   |-- fast_expr.py
    |   |-- simulator.py
    |   `-- __init__.py
    |-- main.py
```

## Usage

This project uses `uv` and Python 3.14.

```sh
uv run wqsim alphas
uv run wqsim run --alpha eg_alpha3 --sample test --output-dir tmp --metrics
uv run wqsim run --alpha eg_alpha3 --sample insample --output-dir tmp --metrics --post-process-mode renormalize_after_truncation
uv run wqsim compare-postprocess --alpha eg_alpha3 --sample insample
uv run wqsim benchmark-alphas --source synthetic --rows 96 --cols 12 --output tmp/alpha_benchmark.json
uv run pytest
```

The CLI loads SQLite data with Polars, converts it into dense NumPy panels, evaluates alpha functions in batch, and writes cumulative PnL figures.
See [docs/accuracy.md](./docs/accuracy.md) for the current accuracy baseline and known limits.

## Runtime Snapshot

Reference benchmark on the full configured dataset (`sample=latest`, 1,806 panel dates,
8,434 symbols), using one shared data load:

- Data load and panel construction: 16.3 seconds.
- All currently registered alphas (`eg_alpha*` plus `alpha001`-`alpha010`): 1,939.6 seconds.
- End-to-end total: 1,955.9 seconds, about 32.6 minutes.
- The example alphas each evaluate in about 1-3 seconds after the panel is loaded; the current
  rolling correlation/rank Alpha101 implementations dominate total runtime.

<!-- 
## Todos

- [ ] Too slow, is it possible for Python to be faster?

- [x] Examine the data => There are truly some problems with data from Snowball. Differences detected when computing the `vwap`.
- [ ] Examine the procedure
- [ ] Examine the fast-expression implementation

- [ ] Add docs to code
- [ ] Do not use string date
- [x] Change `date` and `next_day` to `prev_day` and `today`
- [ ] Truncation correctness (current not necessarily working)

- [x] Implement multiprocessing -->

## References

- The fast expression and Alpha 101 implementation: [WorldQuant_alpha101_code](https://github.com/yli188/WorldQuant_alpha101_code)

<!-- 
actual dependencies:
```
pip install numpy pandas scipy polars pipreqs ruff
 -->
