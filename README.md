# WorldQuant BRAIN Simulator

## Introduction

This is a simulator to help with backtesting your alphas offline for platform WorldQuant Brain

## Todos

- [ ] Too slow, is it possible for Python to be faster?

- [x] Examine the data => There are truly some problem with data from Snowball. Differences detected when computing the `vwap`.
- [ ] Examine the procedure
- [ ] Examine the fast-expression implementation

- [ ] Add docs to code
- [ ] Do not use string date
- [x] Change `date` and `next_day` to `prev_day` and `today`
- [ ] Truncation correctness (current not necessarily working)

- [x] Implement multiprocessing

## References

- The fast expression and Alpha 101 implementation: [WorldQuant_alpha101_code](https://github.com/yli188/WorldQuant_alpha101_code)
