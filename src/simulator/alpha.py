import pandas as pd
import numpy as np

from expression import *


def eg_alpha(prev_day: str, df: pd.DataFrame) -> pd.DataFrame:
    close = df.pivot(index="date", columns="symbol", values="close")
    volume = df.pivot(index="date", columns="symbol", values="volume")

    df = -rank(ts_delta(close, 2)) * rank(volume / ts_sum(volume, 30) / 30)

    df = df.loc[pd.Timestamp(prev_day).date()]
    return df


def eg_alpha2(prev_day: str, df: pd.DataFrame) -> pd.DataFrame:
    close = df.pivot(index="date", columns="symbol", values="close")

    df = -(close - ts_mean(close, 5))

    df = df.loc[pd.Timestamp(prev_day).date()]
    return df


def eg_alpha3(prev_day: str, df: pd.DataFrame) -> pd.DataFrame:
    close = df.pivot(index="date", columns="symbol", values="close")

    df = rank(-(close - ts_mean(close, 10)))

    df = df.loc[pd.Timestamp(prev_day).date()]
    return df
