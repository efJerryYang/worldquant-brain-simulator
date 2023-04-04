# yli188's original work here: https://github.com/yli188/WorldQuant_alpha101_code
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Fast expression implementation


def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate rolling sum.

    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.rolling(window).sum()


def sma(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate SMA.

    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()


def stddev(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate rolling correlations.

    :param x: a pandas DataFrame.

    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate rolling covariance.
    :param x: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]


def ts_rank(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)


def ts_min(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


def ts_max(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def ts_mean(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate rolling mean.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series mean over the past 'window' days.
    """
    return df.rolling(window).mean()


def delta(df: pd.DataFrame, period=1) -> pd.DataFrame:
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today's value minus the value 'period' days ago.
    """
    return df.diff(period)


def ts_delta(df: pd.DataFrame, period=1) -> pd.DataFrame:
    return delta(df, period)


def delay(df: pd.DataFrame, period=1) -> pd.DataFrame:
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)


def rank(df: pd.DataFrame, rate=2) -> pd.DataFrame:
    """
    Cross sectional rank.
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    # if shape[0] = 1, means no need to rank
    if df.shape[0] == 1:
        return df.rank()
    # Official Description:
    # The Rank operator ranks the value of the input data x for the given stock
    # among all instruments, and returns float numbers equally distributed
    # between 0.0 and 1.0. When rate is set to 0, the sorting is done precisely.
    # The default value of rate is 2.
    # https://platform.worldquantbrain.com/learn/data-and-operators/detailed-operator-descriptions#23-rankx-rate2
    return (df.rank() - 1.0) / (df.shape[0] - 1)


def scale(df: pd.DataFrame, k=1) -> pd.DataFrame:
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df: pd.DataFrame, period=10) -> pd.DataFrame:
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.as_matrix()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1 : row + 1, :]
        na_lwma[row, :] = np.dot(x.T, y)
    return pd.DataFrame(na_lwma, index=df.index, columns=["CLOSE"])
