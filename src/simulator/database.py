import os
import time
from typing import List


import pandas as pd
import numpy as np
import sqlite3 as sl


def get_project_dir() -> str:
    filepath = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(filepath)))


def get_database_path() -> str:
    return os.path.join(get_project_dir(), "data", "stock_snowball_us.db")


def connect_to_database() -> sl.Connection:
    return sl.connect(get_database_path())


def select_query(conn, query: str, args: tuple = ()) -> pd.DataFrame:
    try:
        df = pd.read_sql(query, conn, params=args)
    except sl.OperationalError as e:
        raise e
    return df


def handle_load_error(
    e: sl.DatabaseError,
    conn: sl.Connection,
    query: str,
    args: tuple = (),
    max_attempts: int = 10,
    wait_seconds: float = 10.0,
) -> pd.DataFrame:
    original_error = e
    if isinstance(e, sl.OperationalError) and "locked" in str(e).lower():
        for i in range(max_attempts):
            print(
                f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S.%f}] - Database is locked, try again in {wait_seconds} seconds... [{i+1}/{max_attempts}]"
            )
            time.sleep(wait_seconds)
            try:
                df = select_query(conn, query, args)
                return df
            except sl.OperationalError as e:
                continue
        raise original_error
    else:
        raise original_error


def handle_store_error(
    e: sl.DatabaseError,
    conn: sl.Connection,
    data: pd.DataFrame,
    table_name: str,
    max_attempts: int = 10,
    wait_seconds: float = 10.0,
) -> int:
    original_error = e
    if isinstance(e, sl.OperationalError) and "locked" in str(e).lower():
        for i in range(max_attempts):
            print(
                f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S.%f}] - Database is locked, try again in {wait_seconds} seconds... [{i+1}/{max_attempts}]"
            )
            time.sleep(wait_seconds)
            try:
                data.to_sql(
                    table_name,
                    conn,
                    if_exists="append",
                    index=False,
                    dtype={"symbol": "TEXT"},
                )
                return conn.total_changes
            except sl.OperationalError as e:
                continue
        raise original_error
    else:
        raise original_error


def load_data(
    select_list: List[str], table_name: str, where: str = "true", distinct=True
) -> pd.DataFrame:
    """
    Load data from database based on select_list and table_name. Connection will be created and closed automatically.

    Parameters
    ----------
    select_list : List[str]
        List of columns to select
    table_name : str
        Name of table to select from
    where : str, optional
        Where clause, by default "true"
    distinct : bool, optional
        Whether to use DISTINCT in query, by default True

    Returns
    -------
    table_query : DataFrame
        DataFrame of query results
    """
    conn = connect_to_database()
    with conn:
        try:
            if distinct:
                query = f"SELECT DISTINCT {', '.join(select_list)} FROM {table_name} WHERE {where}"
            else:
                query = (
                    f"SELECT {', '.join(select_list)} FROM {table_name} WHERE {where}"
                )
            return select_query(conn, query)
        except sl.OperationalError as e:
            return handle_load_error(e, conn, query)


def store_data(data: pd.DataFrame, table_name: str) -> int:
    conn = connect_to_database()
    with conn:
        try:
            data.to_sql(
                table_name,
                conn,
                if_exists="append",
                index=False,
                dtype={"symbol": "TEXT"},
            )  # chunksize is optional, by default all rows will be inserted at once
            return conn.total_changes
        except sl.OperationalError as e:
            return handle_store_error(e, conn, data, table_name)


def cleanup_stockdata():
    conn = connect_to_database()
    market = "US"
    with conn:
        table_name = f"stock_data_{market}"
        print(f"Deleting duplicated rows from table: {table_name}")
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM {table_name} WHERE rowid NOT IN (SELECT MAX(rowid) FROM {table_name} GROUP BY timestamp_ms, symbol, period)"
        )
        deleted = cursor.rowcount
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        remaining = cursor.fetchone()[0]

        print(f"{'Deleted':12} {'Remaining':12}")
        print(f"{deleted:12} {remaining:12}")
        conn.commit()
    print("Done. ")


def cleanup_stocklist():
    conn = connect_to_database()
    market = "US"
    with conn:
        table_name = f"stock_list_{market}"
        print(f"Deleting duplicated rows from table: {table_name}")
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM {table_name} WHERE (symbol, timestamp_ms) NOT IN (SELECT symbol, MAX(timestamp_ms) as timestamp_ms FROM {table_name} GROUP BY symbol)"
        )
        deleted = cursor.rowcount
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        remaining = cursor.fetchone()[0]

        print(f"{'Deleted':12} {'Remaining':12}")
        print(f"{deleted:12} {remaining:12}")
        conn.commit()
    print("Done. ")


if __name__ == "__main__":
    # 数据来源于 xueqiu.com, 或许需要从 yahoo 数据接口获取数据进行对比（yfinance API 可能限制 2 年历史数据）

    # https://tool.chinaz.com/tools/unixtime.aspx 2016/3/1 - 2022/3/1
    where_stmt = "timestamp_ms between 1456761600000 and 1646064000000"

    df = load_data(
        ["symbol"],
        "stock_data_US",  # 这个表是完整的数据表，我们主要操作来获取这个表的数据
        where=where_stmt,
        distinct=True,
    )
    print(df)
    df = load_data(
        ["symbol", "current", "market_capital", "volume", "turnover_rate", "name"],
        "stock_list_US",  # 这个表是当前基本信息表，是当前的统计信息，类似于 metadata，我们用不太着，这里只是用作样例
        where=f"symbol in {tuple(df.values.flatten())}",
    )
    print(df)

    # 用于清理重复数据（如果有）
    # cleanup_stockdata() # data 表没有加约束（因为这样插入快得多，数据量是千万级别）
    # cleanup_stocklist() # list 表有 unique 约束（如果我没记错的话，当时想的是因为就几千条数据，插入速度影响不大），所以清理 list 是多余的操作
