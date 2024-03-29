import yaml
import timeit
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import Tuple, Dict, List, Callable


from datasource.database import *
from alpha_pool.alpha101 import *
from alpha_pool.alpha import *
from .util import setup_logger, date2timestamp

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


def load_data_(ts_start: str, ts_end: str) -> pd.DataFrame:
    where_stmt = f"timestamp_ms between {ts_start} and {ts_end}"
    df = load_data(
        [
            "symbol",  #                object
            "timestamp_ms",  #          int64
            "open",  #                  float64
            "high",  #                  float64
            "low",  #                   float64
            "close",  #                 float64
            "volume",  #                int64
            "percent as returns",  #    float64
            "amount",  #                float64
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
    df["liquidity"] = (df["volume"] * df["close"]).apply(np.log)


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
    df["amount"].interpolate(method="linear", inplace=True)

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

    df.drop(df.groupby("symbol").head(89).index, inplace=True)
    df.dropna(inplace=True)
    # df.reset_index(drop=True, inplace=True)


def filter_invalid_timestamp_ms(df: pd.DataFrame, inplace=True) -> None:
    ts_list = df.groupby("timestamp_ms").agg({"symbol": "count"})
    ts_list = ts_list[ts_list["symbol"] < 200].index  # fewer than 200 available stocks
    df.drop(df[df["timestamp_ms"].isin(ts_list)].index, inplace=True)
    # df.reset_index(drop=True, inplace=True)


def create_date_column(df: pd.DataFrame, inplace=True) -> None:
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date


def rank_universe(df: pd.DataFrame, inplace=True) -> None:
    direct_start = timeit.default_timer()
    compute_direct_factors(df)
    direct_end = timeit.default_timer()
    logger.info(f"Compute direct factors in {direct_end - direct_start:.2f} seconds.")
    cumulative_start = timeit.default_timer()
    compute_cumulative_factors(df)
    cumulative_end = timeit.default_timer()
    logger.info(
        f"Compute cumulative factors in {cumulative_end - cumulative_start:.2f} seconds."
    )
    filter_start = timeit.default_timer()
    filter_invalid_timestamp_ms(df)
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


def prepare_data(settings: dict) -> pd.DataFrame:
    ts_start, ts_end = set_ts_range(settings)
    load_start = timeit.default_timer()
    df = load_data_(ts_start, ts_end)
    load_end = timeit.default_timer()

    logger.info(f"Load data in {load_end - load_start:.2f} seconds.")
    filter_region(df, settings.get("region", "USA").upper())
    filter_type(df, settings.get("instrument-type", "Equity").capitalize())
    create_date_column(df)

    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    # logger.info(df[df["date"] == pd.Timestamp("2021-02-26").date()])
    rank_start = timeit.default_timer()
    rank_universe(df)
    rank_end = timeit.default_timer()
    logger.info(f"Rank stocks in universe in {rank_end - rank_start:.2f} seconds.")
    # Drop unnecessary columns to save memory
    df.drop(
        columns=[
            "timestamp_ms",
            "liquidity",
            "typical_price",
            "amount",
        ],
        inplace=True,
    )
    # print(df)
    return df


def simulate():
    settings = load_settings()
    df = prepare_data(settings)
    # logger.info(df)
    logger.info(df[df["date"] == pd.Timestamp("2015-10-12").date()].iloc[0:30, :])
    # logger.info(df[df["date"] == pd.Timestamp("2021-02-26").date()])
    alpha = Alphas(df)
    logger.info(alpha)
    logger.info("Done")


class Simulator:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.booksize = 20_000_000  # should not change
        ts_start, ts_end = set_ts_range(self.settings)
        self.df = prepare_data(self.settings)
        df_total_memory = self.df.memory_usage(deep=True).sum()
        # logger.debug(
        #     f"DataFrame size: {self.df.values.nbytes} bytes (= {self.df.values.nbytes/1024**3:.2f} GiB)"
        # )
        logger.debug(
            f"DataFrame size: {df_total_memory} bytes (= {df_total_memory/1024**3:.2f} GiB)"
        )
        # self.data_groupby_date = self.df.groupby("date")
        self.data_dict = self.init_data_dict()
        total_memory = sum(
            df.memory_usage(deep=True).sum() for df in self.data_dict.values()
        )
        logger.debug(
            f"Data Dict: {total_memory} bytes (= {total_memory/1024**3:.2f} GiB)"
        )
        self.date_list = sorted(self.data_dict.keys())

    def init_data_dict(self) -> Dict[str, pd.DataFrame]:
        return {
            str(date): group.set_index("symbol", drop=True)
            for date, group in self.df.groupby("date")
        }

    def pre_processing(self, prev_day: str) -> List[str]:
        return self.filter_by_universe(prev_day)

    # @staticmethod
    def filter_by_universe(self, prev_day: str) -> List[str]:
        # universe: Top3000 # Top1000, Top500, Top200
        # parse the digit from string, regardless of the letter in the string
        top = self.settings.get("universe", "top3000").lower().strip("top")
        if top.isdigit():
            top = int(top)
        else:
            top = 3000
            logger.warning("Invalid universe setting, use default value 3000.")
        day_data = self.data_dict[prev_day]
        # day_data = day_data[day_data["cum_liq_rank"] < top + 1]
        day_data = day_data.nlargest(top, "cumulative_liq")
        return day_data.index.tolist()

    # @staticmethod
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
            profit = process_day(self, prev_day, today, f)
            total += profit
            PnL.append(total)
        simulation_end = timeit.default_timer()
        logger.info(f"Simulation in {simulation_end - simulation_start:.2f} seconds.")
        self.post_simulation(f.__name__, self.date_list[idx:-1], PnL)

    def post_simulation(
        self, alpha_name: str, date_list: List[str], PnL: List[float]
    ) -> None:
        """Plot PnL and save the figure.

        Parameters
        ----------
            alpha_name (str): Name of the alpha.
            date_list (List[str]): List of date strings.
            PnL (List[float]): List of PnL values.

        Returns
        ----------
            None
        """
        fig, ax = plt.subplots(figsize=(40, 15))
        plt.subplots_adjust(
            left=0.02, bottom=0.10, right=0.98, top=0.90, wspace=0.1, hspace=0.1
        )
        ax.plot(PnL)

        if len(date_list) != len(PnL):
            raise ValueError("Length of date_list and PnL must be the same.")

        step = len(PnL) // 50
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
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

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
                for prev_day, today, in zip(sim_date_list[:-1], sim_date_list[1:])
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

    # @staticmethod
    def compute_profit_pct(self, today: str, alpha: pd.DataFrame) -> float:
        returns = self.data_dict[today]["returns"].reindex(alpha.index)
        return (alpha * returns).sum()


def process_day(
    simulator: Simulator,
    prev_day: str,
    today: str,
    f: Callable,
):
    s = simulator

    universe = s.filter_by_universe(prev_day)
    prev_day_dt = pd.Timestamp(prev_day).date()
    start_day_dt = (prev_day_dt - pd.DateOffset(days=60)).date()
    df = s.df.query("symbol in @universe and @start_day_dt <= date <= @prev_day_dt")

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
    # s.simulate(eg_alpha)
    s.simulate_with_multiprocessing(eg_alpha3)
