from fast_expression import *
from database import *
import yaml
import timeit

from typing import Tuple, Dict, List


def load_settings() -> dict:
    filepath = os.path.join(os.path.dirname(__file__), "settings.yaml")
    with open(filepath, "r") as f:
        settings = yaml.safe_load(f)
    return settings


def set_date_range(settings: dict) -> Tuple[str, str]:
    ts_start = "1433088000000"  # 2015-06-01
    load_start = timeit.default_timer()
    if settings.get("sample") == "outsample":
        ts_end = "1646064000000"  # cheating, use outsample 2022-03-01
    elif settings.get("sample") == "latest":
        ts_end = "1677600000000"  # cheating, use 2023-03-01
    elif settings.get("sample") == "insample":
        ts_end = "1614528000000"  # insample 2021-03-01
    else:
        print("[WARNING] No sample setting, use insample by default.")
        ts_end = "1614528000000"  # insample 2021-03-01
    return ts_start, ts_end


def load_data_(ts_start: str, ts_end: str) -> pd.DataFrame:
    where_stmt = f"timestamp_ms between {ts_start} and {ts_end}"
    df = load_data(
        ["symbol", "timestamp_ms", "open", "high", "low", "close", "volume"],
        "stock_data_US",
        where=where_stmt,
        distinct=False,  # for efficiency, True will be slower by 30%
    )
    return df


def filter_region(df: pd.DataFrame, region: str, inplace=True):
    pass


def filter_type(df: pd.DataFrame, type: str, inplace=True):
    pass


def compute_liquidity(df: pd.DataFrame, inplace=True) -> None:
    df["liquidity"] = (df["volume"] * df["close"]).apply(np.log10)


def compute_cumulative_liq(df: pd.DataFrame, inplace=True) -> None:
    """
    Compute cumulative liquidity for each symbol in 90 days.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns "symbol", "timestamp_ms", "liquidity"
    inplace : bool, optional
        Whether to modify df in place, by default True
    """
    # cum by symbol in 90 days
    # df["cumulative_liq"] = df["liquidity"].cumsum()
    df.sort_values(by=["symbol", "timestamp_ms"], inplace=True)
    df["cumulative_liq"] = df.groupby("symbol")["liquidity"].transform(
        lambda x: x.rolling(90, min_periods=1).sum()
    )
    # print("Before drop 89 rows")
    # print(df[df["symbol"] == "UBCP"].iloc[0:30, :])
    df.drop(
        df.groupby("symbol").head(89).index, inplace=True
    )  # TODO: 只是单独的drop不行，这里可能需要还再计算一些内容，不然就浪费掉了一段时间的数据
    # print("After drop 89 rows")
    # print(df[df["symbol"] == "UBCP"].iloc[0:30, :])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


def compute_cum_liq_rank(df: pd.DataFrame, inplace=True) -> None:
    df.sort_values(
        by=["dates", "cumulative_liq"], inplace=True, ascending=[True, False]
    )
    df["cum_liq_rank"] = df.groupby("dates")["cumulative_liq"].rank(
        ascending=False, method="dense"
    )


def group_data_by_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("timestamp_ms").agg(
        {
            "symbol": "count",
            "liquidity": "max",
            "cumulative_liq": "max",
        }
    )


def filter_invalid_timestamp_ms(df: pd.DataFrame, inplace=True) -> None:
    # if the valid agg count of symbol is lower than 200, drop the grouped timestamp_ms
    ts_list = df.groupby("timestamp_ms").agg({"symbol": "count"})
    ts_list = ts_list[ts_list["symbol"] < 200].index
    df.drop(df[df["timestamp_ms"].isin(ts_list)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)


def create_dates_column(df: pd.DataFrame, inplace=True) -> None:
    df["dates"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date


def rank_universe(df: pd.DataFrame, inplace=True) -> None:
    compute_liquidity(df)
    compute_cumulative_liq(df)
    filter_invalid_timestamp_ms(df)
    compute_cum_liq_rank(df)


def prepare_data(settings: dict) -> pd.DataFrame:
    ts_start, ts_end = set_date_range(settings)

    load_start = timeit.default_timer()
    df = load_data_(ts_start, ts_end)
    load_end = timeit.default_timer()
    print(f"Load data in {load_end - load_start:.2f} seconds.")

    filter_region(df, settings.get("region", "USA").upper())
    filter_type(df, settings.get("instrument-type", "Equity").capitalize())

    create_dates_column(df)
    print(f"Date range: {df['dates'].min()} to {df['dates'].max()}")
    # print(df[df["dates"] == pd.Timestamp("2021-02-26").date()])

    rank_start = timeit.default_timer()
    rank_universe(df)
    rank_end = timeit.default_timer()
    print(f"Rank stocks in universe in {rank_end - rank_start:.2f} seconds.")
    return df


def simulate():
    settings = load_settings()
    df = prepare_data(settings)
    print(df)
    print("Done")


if __name__ == "__main__":
    simulate()
