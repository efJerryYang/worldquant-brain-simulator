from fast_expression import *
from database import *
import yaml
import timeit


def load_settings() -> dict:
    with open("settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    return settings


def filter_region(df: pd.DataFrame, region: str, inplace=True):
    pass


def filter_type(df: pd.DataFrame, type: str, inplace=True):
    pass


def setup_runtime_env(settings: dict) -> None:

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
    df["cumulative_liq"] = df.groupby("symbol")["liquidity"].transform(
        lambda x: x.rolling(90, min_periods=1).sum()
    )
    # print("Before drop 89 rows")
    # print(df[df["symbol"] == "UBCP"].iloc[0:30, :])
    df.drop(df.groupby("symbol").head(89).index, inplace=True)
    # print("After drop 89 rows")
    # print(df[df["symbol"] == "UBCP"].iloc[0:30, :])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    settings = load_settings()
    setup_runtime_env(settings)

    load_start = timeit.default_timer()
    where_stmt = "timestamp_ms between 1456761600000 and 1646064000000"
    df = load_data(
        ["symbol", "timestamp_ms", "open", "high", "low", "close", "volume"],
        "stock_data_US",
        where=where_stmt,
        distinct=False,  # for efficiency, True will be slower by 30%
    )
    load_end = timeit.default_timer()
    print(f"Load data in {load_end - load_start:.2f} seconds.")

    liq_start = timeit.default_timer()
    compute_liquidity(df)
    liq_end = timeit.default_timer()
    print(f"Compute liquidity in {liq_end - liq_start:.2f} seconds.")

    cum_liq_start = timeit.default_timer()
    compute_cumulative_liq(df)
    cum_liq_end = timeit.default_timer()
    print(f"Compute cumulative liquidity in {cum_liq_end - cum_liq_start:.2f} seconds.")
    print(df)
