# yli188's original work here: https://github.com/yli188/WorldQuant_alpha101_code
import polars as pl
from scipy.stats import rankdata
import numpy as np


def ts_sum(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate rolling sum for each column.

    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the time-series sum over the past 'window' days for each column.
    """
    print(df)
    return df.select([
        pl.col(column).rolling_sum(window_size=window).alias(column)
        for column in df.columns
    ])

def sma(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate SMA.

    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the time-series min over the past 'window' days.
    """
    return df.select([
        pl.col(column).rolling_mean(window_size=window).alias(column)
        for column in df.columns
    ])


def stddev(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the time-series min over the past 'window' days.
    """
    return df.select([
        pl.col(column).rolling_std(window_size=window).alias(column)
        for column in df.columns
    ])


def correlation(x: pl.DataFrame, y: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate rolling correlations.

    :param x: a polars DataFrame.
    :param y: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the rolling correlation over the past 'window' days.
    """
    # TODO: from the polars, it states that: 
    # polars.rolling_corr: This functionality is considered unstable. It may be changed at any point without it being considered a breaking change./
    return pl.rolling_corr(x, y, window_size=window)


def covariance(x: pl.DataFrame, y: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate rolling covariance.
    :param x: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the time-series min over the past 'window' days.
    """
    # TODO: from the polars, it states that: 
    # polars.rolling_cov: This functionality is considered unstable. It may be changed at any point without it being considered a breaking change./
    return pl.rolling_cov(x, y, window_size=window)


def rolling_rank(na):
    """
    Auxiliary function to be used in polars.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]


def ts_rank(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate rolling rank.
    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na):
    """
    Auxiliary function to be used in polars.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate rolling product.
    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)


def ts_min(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate rolling min.
    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


def ts_max(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate rolling min.
    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def ts_mean(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate rolling mean.
    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: a polars DataFrame with the time-series mean over the past 'window' days.
    """
    return df.rolling(window).mean()


def delta(df: pl.DataFrame, period=1) -> pl.DataFrame:
    """
    Wrapper function to estimate difference.
    :param df: a polars DataFrame.
    :param period: the difference grade.
    :return: a polars DataFrame with today's value minus the value 'period' days ago.
    """
    return df.shift(period) - df


def ts_delta(df: pl.DataFrame, period=1) -> pl.DataFrame:
    return delta(df, period)


def delay(df: pl.DataFrame, period=1) -> pl.DataFrame:
    """
    Wrapper function to estimate lag.
    :param df: a polars DataFrame.
    :param period: the lag grade.
    :return: a polars DataFrame with lagged time series
    """
    return df.shift(period)


def rank(df: pl.DataFrame, method: str = 'average', descending: bool = False) -> pl.DataFrame:
    """
    Cross sectional rank along all columns.
    :param df: a polars DataFrame.
    :param method: The method used to assign ranks to tied elements.
                   Options: 'average', 'min', 'max', 'dense', 'ordinal', 'random'
    :param descending: Rank in descending order if True.
    :return: a polars DataFrame with rank along columns.
    """
    if df.height == 1:
        return df

    def rank_and_normalize(s: pl.Series) -> pl.Series:
        ranked = s.rank(method=method, descending=descending)
        return (ranked - 1.0) / (s.len() - 1)

    return df.select(
        [
            rank_and_normalize(pl.col(column)).alias(column)
            for column in df.columns
        ]
    )


def scale(df: pl.DataFrame, k=1) -> pl.DataFrame:
    """
    Scaling time serie.
    :param df: a polars DataFrame.
    :param k: scaling factor.
    :return: a polars DataFrame rescaled df such that sum(abs(df)) = k
    """
    # return df.mul(k).div(np.abs(df).sum())
    return df.with_columns(df.sum().abs() / k)


def ts_argmax(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a polars DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df: pl.DataFrame, period=10) -> pl.DataFrame:
    """
    Linear weighted moving average implementation.
    :param df: a polars DataFrame.
    :param period: the LWMA period
    :return: a polars DataFrame with the LWMA.
    """
    # Clean data
    if df.is_null().any():
        df = df.fill_null(method="ffill").fill_null(method="bfill").fill_null(0)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.to_numpy()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.height):
        x = na_series[row - period + 1 : row + 1, :]
        na_lwma[row, :] = np.dot(x.T, y)
    return pl.DataFrame(na_lwma, index=df.index, columns=["CLOSE"])
