import os
import yaml
import timeit
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import Tuple, Dict, List, Callable
import polars as pl
from datetime import datetime, timedelta

from datasource.database import load_data
from alpha_pool.alpha101 import *
from alpha_pool.alpha import *
from .util import setup_logger, date2timestamp, timestamp2date

logger = setup_logger(__name__)


def load_settings() -> dict:
    filepath = os.path.join(os.path.dirname(__file__), "settings.yaml")
    with open(filepath, "r") as f:
        settings = yaml.safe_load(f)
    return settings


def set_ts_range(settings: dict) -> Tuple[str, str]:
    date_start = "2015-06-01"
    load_start = timeit.default_timer()
    if settings.get("sample") == "outsample":
        date_end = "2022-03-01"  # cheating, use outsample
    elif settings.get("sample") == "latest":
        date_end = "2023-03-01"  # cheating
    elif settings.get("sample") == "insample":
        date_end = "2021-03-01"  # insample
    elif settings.get("sample") == "test":
        date_end = "2020-09-01"  # abnormal data test end
        # date_end = "2020-03-01"  # test
        # date_end = "2019-03-01"  # test
        # date_end = "2018-03-01"  # test
        # date_end = "2017-03-01"  # test
        # date_end = "2016-12-01"  # test
        # date_end = "2016-06-01"  # test
        date_start = "2019-03-01"  # abnormal data test start
    else:
        logger.warning("No sample setting, use 'insample' by default.")
        date_end = "2021-03-01"  # insample 2021-03-01
    ts_start = date2timestamp(date_start)
    ts_end = date2timestamp(date_end)
    return ts_start, ts_end


def load_data_(ts_start: str, ts_end: str) -> pl.DataFrame:
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
        distinct=False,
    )
    return df


def filter_region(df: pl.DataFrame, region: str) -> pl.DataFrame:
    pass


def filter_type(df: pl.DataFrame, type: str) -> pl.DataFrame:
    pass


def compute_direct_factors(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
    )
    df = df.with_columns((pl.col("volume") * pl.col("close")).log().alias("liquidity"))
    return df


def compute_cumulative_factors(df: pl.DataFrame) -> pl.DataFrame:
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
    # df["amount"].replace(0, np.nan, inplace=True)
    # df["amount"].interpolate(method="linear", inplace=True)
    # TODO: weird, fill 0 with nan
    df = df.with_columns(pl.col("amount").replace(0, None))
    logger.info(df.shape)
    df = df.with_columns(pl.col("amount").interpolate(method="linear"))
    logger.info(df.shape)

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
    # df["vwap"] = df["amount"] / df["volume"]
    df = df.with_columns(pl.col("amount") / pl.col("volume").alias("vwap"))
    logger.info(df.shape)
    # Cumulative Factor#4 cumulative liquidity
    # cum by symbol in 90 days
    # df.sort_values(by=["symbol", "timestamp_ms"], inplace=True)
    df = df.sort(by=["symbol", "timestamp_ms"])
    logger.info(df.shape)
    # df["cumulative_liq"] = df.groupby("symbol")["liquidity"].transform(
    #     lambda x: x.rolling(90, min_periods=1).sum()
    # )

    logger.info(df.shape)
    df = df.sort(["symbol", "timestamp_ms"])

    result = (
        df
        .group_by("symbol")
        .agg([
            pl.col("timestamp_ms"),
            pl.col("liquidity").rolling_sum(window_size=90).alias("cumulative_liq")
        ])
        .explode(["timestamp_ms", "cumulative_liq"])
        .filter(pl.col("cumulative_liq").is_not_null())
    )
    logger.info(result)
    # Join the result back with the original dataframe to retain other columns
    df = df.join(result, on=["symbol", "timestamp_ms"], how="inner")
    logger.info(df)



    # df.drop(df.groupby("symbol").head(89).index, inplace=True)
    # df.dropna(inplace=True)
    # df = df.drop_nulls()
    # df.reset_index(drop=True, inplace=True)
    return df


def filter_invalid_timestamp_ms(df: pl.DataFrame) -> pl.DataFrame:
    ts_list = df.group_by("timestamp_ms").agg(pl.count("symbol"))
    ts_list = ts_list.filter(pl.col("symbol") < 200)["timestamp_ms"]
    return df.filter(~pl.col("timestamp_ms").is_in(ts_list))


def create_date_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms").dt.date().alias("date")
    )


def rank_universe(df: pl.DataFrame) -> pl.DataFrame:
    direct_start = timeit.default_timer()
    df = compute_direct_factors(df)
    logger.info(df)
    direct_end = timeit.default_timer()
    logger.info(f"Compute direct factors in {direct_end - direct_start:.2f} seconds.")

    cumulative_start = timeit.default_timer()
    df = compute_cumulative_factors(df)
    logger.info(df)
    cumulative_end = timeit.default_timer()
    logger.info(
        f"Compute cumulative factors in {cumulative_end - cumulative_start:.2f} seconds."
    )

    filter_start = timeit.default_timer()
    df = filter_invalid_timestamp_ms(df)
    logger.info(df)
    filter_end = timeit.default_timer()
    logger.info(
        f"Filter invalid timestamp_ms in {filter_end - filter_start:.2f} seconds."
    )

    rank_start = timeit.default_timer()
    # compute_cum_liq_rank(df)
    rank_end = timeit.default_timer()
    logger.info(
        f"Compute cumulative liquidity rank in {rank_end - rank_start:.2f} seconds."
    )

    return df


def prepare_data(settings: dict) -> pl.DataFrame:
    ts_start, ts_end = set_ts_range(settings)
    logger.info(f"Expected date range: {timestamp2date(ts_start)} to {timestamp2date(ts_end)}")
    load_start = timeit.default_timer()
    df = load_data_(ts_start, ts_end)
    load_end = timeit.default_timer()

    logger.info(f"Load data in {load_end - load_start:.2f} seconds.")
    filter_region(df, settings.get("region", "USA").upper())
    filter_type(df, settings.get("instrument-type", "Equity").capitalize())
    df = create_date_column(df)
    logger.info(df)

    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    rank_start = timeit.default_timer()
    df = rank_universe(df)
    rank_end = timeit.default_timer()
    logger.info(f"Rank stocks in universe in {rank_end - rank_start:.2f} seconds.")
    # Drop unnecessary columns to save memory
    logger.info(df)
    df = df.drop(
        [
            "timestamp_ms",
            "liquidity",
            "typical_price",
            "amount",
        ]
    )
    return df


def simulate():
    settings = load_settings()
    df = prepare_data(settings)
    logger.info(
        df[df["date"] == datetime.strptime("2015-10-12", "%Y-%m-%d").date()].head(
            30
        )
    )
    alpha = Alphas(df)
    logger.info(alpha)
    logger.info("Done")


class Simulator:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.booksize = 20_000_000  # should not change
        ts_start, ts_end = set_ts_range(self.settings)
        self.df = prepare_data(self.settings)
        logger.info(self.df)
        df_total_memory = self.df.estimated_size()
        logger.debug(
            f"DataFrame size: {df_total_memory} bytes (= {df_total_memory/1024**3:.2f} GiB)"
        )
        self.data_dict = self.init_data_dict()
        logger.info(self.data_dict)
        total_memory = sum(df.estimated_size() for df in self.data_dict.values())
        logger.debug(
            f"Data Dict: {total_memory} bytes (= {total_memory/1024**3:.2f} GiB)"
        )
        self.date_list = sorted(self.data_dict.keys())

    def init_data_dict(self) -> Dict[str, pl.DataFrame]:
        logger.info(self.df)
        return {
            date[0].strftime("%Y-%m-%d"): group.drop("date")
            for date, group in self.df.group_by("date")
        }

    def pre_processing(self, prev_day: str) -> List[str]:
        return self.filter_by_universe(prev_day)

    def filter_by_universe(self, prev_day: str) -> List[str]:
        top = self.settings.get("universe", "top3000").lower().strip("top")
        if top.isdigit():
            top = int(top)
        else:
            top = 3000
            logger.warning("Invalid universe setting, use default value 3000.")
        day_data = self.data_dict[prev_day]
        day_data = day_data.sort("cumulative_liq", descending=True).head(top)
        return day_data["symbol"].to_list()

    def post_processing(self, alpha: pl.DataFrame) -> pl.DataFrame:
        alpha = alpha.with_columns(
            (pl.col("alpha") - pl.col("alpha").mean()).alias("alpha")
        )
        boundary = self.settings.get("truncation", 0.1)
        alpha = alpha.with_columns(pl.col("alpha").clip(-boundary, boundary))
        alpha = alpha.with_columns(
            (pl.col("alpha") / pl.col("alpha").abs().sum()).alias("alpha")
        )
        return alpha

    def __set_sim_start_date(self) -> int:
        start_date = self.settings.get("start-date", "2016-03-01")
        try:
            idx = self.date_list.index(start_date)
        except ValueError as ve:
            logger.warning(f"Start date {start_date} not found, set 'idx' to 90.")
            idx = 90
        return idx

    def simulate(self, f: Callable) -> None:
        total = 0
        PnL = []
        idx = self.__set_sim_start_date()
        simulation_start = timeit.default_timer()
        for prev_day, today in zip(self.date_list[idx:-1], self.date_list[idx + 1 :]):
            logger.info(f"Simulating {prev_day} to {today}")
            profit = process_day(self, prev_day, today, f)
            total += profit
            PnL.append(total)
        simulation_end = timeit.default_timer()
        logger.info(f"Simulation in {simulation_end - simulation_start:.2f} seconds.")
        self.post_simulation(f.__name__, self.date_list[idx:-1], PnL)

    def post_simulation(
        self, alpha_name: str, date_list: List[str], PnL: List[float]
    ) -> None:
        fig, ax = plt.subplots(figsize=(40, 15))
        plt.subplots_adjust(
            left=0.02, bottom=0.10, right=0.98, top=0.90, wspace=0.1, hspace=0.1
        )
        ax.plot(PnL)

        if len(date_list) != len(PnL):
            raise ValueError("Length of date_list and PnL must be the same.")

        step = len(PnL) // 50
        logger.info(date_list)
        logger.info(PnL)
        logger.info(step)
        ax.set_xticks(np.arange(0, len(PnL), step))
        angle = 45  # angle to slant the x-axis labels
        ax.set_xticklabels(date_list[::step], rotation=angle, ha="right", va="top")

        # Add small red dots at the y-axis position
        for i in ax.get_xticks():
            ax.plot(i, PnL[i], "ro", markersize=3)

        # Save figure
        tmp_prefix = "tmp_" if self.settings.get("temporary", True) else ""
        start_date, end_date = date_list[0], date_list[-1]
        fig_prefix = f"{tmp_prefix}PnL_{alpha_name}_{start_date}_{end_date}_step{step}"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        fig_path = f"{fig_prefix}_{timestamp}.png"
        fig.savefig(fig_path)

    def simulate_with_multiprocessing(self, f: Callable) -> None:
        total = 0
        PnL = []
        results = []
        idx = self.__set_sim_start_date()
        sim_date_list = self.date_list[idx:]
        # Use all available cores
        num_processes = mp.cpu_count() // 4
        # # Split the date range into chunks
        chunk_size = len(sim_date_list) // num_processes
        mp_start = timeit.default_timer()
        with mp.Pool(processes=num_processes) as pool:
            process_args = [
                (self, prev_day, today, f)
                for prev_day, today in zip(sim_date_list[:-1], sim_date_list[1:])
            ]
            process_results = pool.starmap(
                process_day, process_args, chunksize=chunk_size
            )
            results.extend([r for r in process_results])
        mp_end = timeit.default_timer()
        logger.info(f"Multiprocessing in {mp_end - mp_start:.2f} seconds.")
        PnL = [
            sum(results[: i + 1]) for i in range(len(results))
        ]  # cumulative sum of results
        self.post_simulation(f.__name__, sim_date_list[:-1], PnL)

    def compute_profit_pct(self, today: str, alpha: pl.DataFrame) -> float:
        returns = self.data_dict[today].filter(pl.col("symbol").is_in(alpha["symbol"]))[
            "returns"
        ]
        return (alpha.join(returns, on="symbol", how="inner")["alpha"] * returns).sum()


def process_day(
    simulator: Simulator,
    prev_day: str,
    today: str,
    f: Callable,
):
    s = simulator

    universe = s.filter_by_universe(prev_day)
    prev_day_dt = datetime.strptime(prev_day, "%Y-%m-%d").date()
    start_day_dt = (prev_day_dt - timedelta(days=60))
    df = s.df.filter(
        (pl.col("symbol").is_in(universe))
        & (pl.col("date").is_between(start_day_dt, prev_day_dt))
    )
    alpha = f(prev_day, df)
    alpha = s.post_processing(alpha)
    profit_pct = s.compute_profit_pct(today, alpha)
    profit = profit_pct * s.booksize / 100
    logger.info(
        f"{today} - Profit: {profit:+10.2f} ({profit / 1e3:+8.2f}k), Percent: {profit_pct:+6.2f}%"
    )
    return profit
 
if __name__ == "__main__":
    s = Simulator()
    s.simulate_with_multiprocessing(eg_alpha3)
