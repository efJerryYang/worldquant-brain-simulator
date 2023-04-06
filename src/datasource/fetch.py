import yfinance as yf
import pandas as pd
import numpy as np

# Blocked by yahoo finance
df = yf.download(
    "MSFT",
    start="2017-01-01",
    end="2017-02-21",
    interval="1d",
    proxy="socks5://127.0.0.1:7890"
)

print(df)