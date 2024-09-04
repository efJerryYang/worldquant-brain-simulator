import polars as pl

# Create example data with 10 rows for each symbol
data = {
    "symbol": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
    "timestamp_ms": [
            1551675600000, 1551762000000, 1551848400000, 1551934800000, 1552021200000,
            1552107600000, 1552194000000, 1552280400000, 1552366800000, 1552453200000,
            1551675600000, 1551762000000, 1551848400000, 1551934800000, 1552021200000,
            1552107600000, 1552194000000, 1552280400000, 1552366800000, 1552453200000,
            1551675600000, 1551762000000, 1551848400000, 1551934800000, 1552021200000,
            1552107600000, 1552194000000, 1552280400000, 1552366800000, 1552453200000,
        ],
        "liquidity": [
            19.335068, 18.506245, 18.713152, 17.901002, 16.874623, 
            20.765412, 21.234567, 22.908123, 23.123456, 24.54321,
            15.621, 16.25365, 15.9222, 16.7801, 17.3451, 
            18.4561, 19.2342, 20.4563, 21.1234, 22.4565,
            20.0, 22.0, 24.0, 26.0, 28.0, 
            30.0, 32.0, 34.0, 36.0, 38.0
        ]
    }
df = pl.DataFrame(data)

# Perform the rolling_sum operation with group_by
result = (
    df.group_by("symbol")
    .agg([
        pl.col("timestamp_ms"),  # Retain the timestamp column
        pl.col("liquidity").rolling_sum(window_size=3).alias("cumulative_liq")  # Rolling sum
    ])
    .explode(["timestamp_ms", "cumulative_liq"])  # Explode to flatten both columns
)

print(result.head(10))
