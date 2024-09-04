import polars as pl
import numpy as np
from datetime import datetime

from .expression import *


def eg_alpha(prev_day: str, df: pl.DataFrame) -> pl.DataFrame:

    close = df.pivot(index="date", columns="symbol", values="close")
    volume = df.pivot(index="date", columns="symbol", values="volume")
    
    df = -1 * rank(ts_delta(close, 2)) * rank(volume / ts_sum(volume, 30) / 30)

    df = df.filter(pl.col("date") == datetime.strptime(prev_day, "%Y-%m-%d").date())
    return df


def eg_alpha2(prev_day: str, df: pl.DataFrame) -> pl.DataFrame:
    close = df.pivot(index="date", columns="symbol", values="close")

    df = -(close - ts_mean(close, 5))

    df = df.filter(pl.col("date") == datetime.strptime(prev_day, "%Y-%m-%d").date())
    return df


def eg_alpha3(prev_day: str, df: pl.DataFrame) -> pl.DataFrame:
    close = df.pivot(index="date", columns="symbol", values="close")

    df = rank(-(close - ts_mean(close, 10)))

    df = df.filter(pl.col("date") == datetime.strptime(prev_day, "%Y-%m-%d").date())
    return df
