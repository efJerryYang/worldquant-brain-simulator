from fast_expression import *
from database import *
import yaml
import timeit
import matplotlib.pyplot as plt

from typing import Tuple, Dict, List, Callable
from alpha101 import *


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
    elif settings.get("sample") == "test":
        ts_end = "1480521600000"  # test 2016-12-01
    else:
        print("[WARNING] No sample setting, use insample by default.")
        ts_end = "1614528000000"  # insample 2021-03-01
    return ts_start, ts_end


def load_data_(ts_start: str, ts_end: str) -> pd.DataFrame:
    where_stmt = f"timestamp_ms between {ts_start} and {ts_end}"
    df = load_data(
        [
            "symbol",
            "timestamp_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "percent as returns",
            "amount",
        ],
        "stock_data_US",
        where=where_stmt,
        distinct=False,  # for efficiency, True will be slower by 30%
    )
    return df


def filter_region(df: pd.DataFrame, region: str, inplace=True):
    pass


def filter_type(df: pd.DataFrame, type: str, inplace=True):
    pass


def compute_direct_factors(df: pd.DataFrame, inplace=True) -> None:

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["liquidity"] = (df["volume"] * df["close"]).apply(np.log10)


def compute_cumulative_factors(df: pd.DataFrame, inplace=True) -> None:
    """
    Compute cumulative factors for each symbol.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns "symbol", "timestamp_ms",
    inplace : bool, optional
        Whether to modify df in place, by default True
    """
    # Data preprocessing
    df["amount"].replace(0, np.nan, inplace=True)
    # df["amount"].interpolate(method="linear", inplace=True)
    df["amount"].fillna(
        df.groupby("symbol")["amount"].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        ),
    )

    # # Cumulative Factor#1 cumulative volume
    # df["cum_volume"] = (
    #     df.groupby("symbol").amount.cumsum() / df["amount"] * df["volume"]
    # )
    # # Cumulative Factor#2 cumulative typical price
    # df["cum_typical_price"] = (
    #     df.groupby("symbol").amount.cumsum() / df["amount"] * df["typical_price"]
    # )
    # Optimization:
    # df['cumulative_volume'] = df.groupby('symbol')['volume'].cumsum()
    # df['cumulative_amount_times_volume'] = df.groupby('symbol')['amount'].cumsum() / df['amount'] * df['volume']

    # # Cumulative Factor#3 volume weighted average price
    # df["vwap"] = df["cum_typical_price"] / df["cum_volume"]
    df["vwap"] = df["amount"] / df["volume"]

    # Cumulative Factor#4 cumulative liquidity
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
    df.sort_values(by=["date", "cumulative_liq"], inplace=True, ascending=[True, False])
    df["cum_liq_rank"] = (
        df.groupby("date")["cumulative_liq"]
        .rank(ascending=False, method="dense")
        .apply(np.int32)
    )  # set datatype to int


def group_data_by_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("timestamp_ms").agg(
        {
            "symbol": "count",
            "liquidity": "max",
            "cumulative_liq": "max",
        }
    )


def filter_invalid_timestamp_ms(df: pd.DataFrame, inplace=True) -> None:
    # If the valid agg count of symbol is lower than 200, drop the grouped timestamp_ms
    ts_list = df.groupby("timestamp_ms").agg({"symbol": "count"})
    ts_list = ts_list[ts_list["symbol"] < 200].index
    df.drop(df[df["timestamp_ms"].isin(ts_list)].index, inplace=True)
    df.reset_index(drop=True, inplace=True)


def create_date_column(df: pd.DataFrame, inplace=True) -> None:
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date


def rank_universe(df: pd.DataFrame, inplace=True) -> None:
    direct_start = timeit.default_timer()
    compute_direct_factors(df)
    direct_end = timeit.default_timer()
    print(f"Compute direct factors in {direct_end - direct_start:.2f} seconds.")

    cumulative_start = timeit.default_timer()
    compute_cumulative_factors(df)
    cumulative_end = timeit.default_timer()
    print(
        f"Compute cumulative factors in {cumulative_end - cumulative_start:.2f} seconds."
    )

    filter_start = timeit.default_timer()
    filter_invalid_timestamp_ms(df)
    filter_end = timeit.default_timer()
    print(f"Filter invalid timestamp_ms in {filter_end - filter_start:.2f} seconds.")

    rank_start = timeit.default_timer()
    compute_cum_liq_rank(df)
    rank_end = timeit.default_timer()
    print(f"Compute cumulative liquidity rank in {rank_end - rank_start:.2f} seconds.")


def prepare_data(settings: dict) -> pd.DataFrame:
    ts_start, ts_end = set_date_range(settings)

    load_start = timeit.default_timer()
    df = load_data_(ts_start, ts_end)
    load_end = timeit.default_timer()
    print(f"Load data in {load_end - load_start:.2f} seconds.")

    filter_region(df, settings.get("region", "USA").upper())
    filter_type(df, settings.get("instrument-type", "Equity").capitalize())

    create_date_column(df)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    # print(df[df["date"] == pd.Timestamp("2021-02-26").date()])

    rank_start = timeit.default_timer()
    rank_universe(df)
    rank_end = timeit.default_timer()
    print(f"Rank stocks in universe in {rank_end - rank_start:.2f} seconds.")
    # Drop unnecessary columns to save memory
    df.drop(columns=["timestamp_ms"], inplace=True)
    return df


def simulate():
    settings = load_settings()
    df = prepare_data(settings)
    # print(df)
    print(df[df["date"] == pd.Timestamp("2015-10-12").date()].iloc[0:30, :])
    # print(df[df["date"] == pd.Timestamp("2021-02-26").date()])
    alpha = Alphas(df)
    print(alpha)
    print("Done")


class Simulator:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.booksize = 20_000_000  # should not change
        ts_start, ts_end = set_date_range(self.settings)
        self.df = prepare_data(self.settings)
        self.data_groupby_date = self.df.groupby("date")
        self.data_dict = self.init_data_dict()
        self.date_list = sorted(self.data_dict.keys())

    def init_data_dict(self) -> Dict[str, pd.DataFrame]:
        d = {}
        return {
            str(date): group.set_index("symbol", drop=True)
            for date, group in self.data_groupby_date
        }

    def pre_processing(self, date: str) -> List[str]:
        return self.filter_by_universe(date)

    def filter_by_universe(self, prev_day: str) -> List[str]:
        # universe: Top3000 # Top1000, Top500, Top200
        # parse the digit from string, regardless of the letter in the string
        top = self.settings.get("universe", "top3000").lower().strip("top")
        if top.isdigit():
            top = int(top)
        else:
            top = 3000
            print("Invalid universe setting, use default value 3000.")
        day_data = self.data_dict[prev_day]
        day_data = day_data[day_data["cum_liq_rank"] < top + 1]
        return day_data.index.tolist()

    def post_processing(self, alpha: pd.DataFrame) -> pd.DataFrame:
        """
        Neutralization and normalization to get the final weights.
        """
        # Neutralization
        by_what = self.settings.get("neutralization", "Market").lower()
        alpha = alpha - alpha.mean()

        # Truncation
        boundary = self.settings.get("truncation", 0.1)
        alpha = alpha.clip(-boundary, boundary)

        # Normalization
        alpha = alpha / alpha.abs().sum()

        return alpha

    def simulate(self, f: Callable) -> None:
        total = 0
        PnL = []
        for prev_day, today in zip(self.date_list[:-1], self.date_list[1:]):
            if prev_day < "2016-03-01":
                continue
            universe = self.pre_processing(prev_day)
            alpha = f(prev_day, universe, self.df)
            alpha = self.post_processing(alpha)
            profit_pct = self.compute_profit_pct(today, alpha)
            profit = profit_pct * self.booksize
            total += profit
            PnL.append(total)
            print(f"{today}: {profit:.2f}, total: {total:.2f}")
        # fig size 4000 * 1500
        plt.figure(figsize=(40, 15))
        plt.plot(PnL)
        plt.savefig(f"tmp_PnL_{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}.png")

    def compute_profit_pct(self, today: str, alpha: pd.DataFrame) -> float:
        returns = self.data_dict[today]["returns"].reindex(alpha.index)
        return (alpha * returns).sum()


def example_alpha(prev_day: str, universe: List[str], df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["symbol"].isin(universe)]
    df = df[df["date"] <= pd.Timestamp(prev_day).date()]
    close = df.pivot(index="date", columns="symbol", values="close")
    volume = df.pivot(index="date", columns="symbol", values="volume")
    df = -rank(ts_delta(close, 2)) * rank(volume / ts_sum(volume, 30) / 30)
    df = df.loc[pd.Timestamp(prev_day).date()]
    return df


if __name__ == "__main__":
    s = Simulator()
    s.simulate(example_alpha)
