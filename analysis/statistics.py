import pandas as pd
import numpy as np

TRADING_DAYS = 252

def prepare(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df

def daily_return(df, price_col="adj_close"):
    df = prepare(df)
    df["return"] = df[price_col].pct_change()
    return df

def moving_average(df, window=20, price_col="adj_close"):
    df = prepare(df)
    df[f"ma_{window}"] = df[price_col].rolling(window=window).mean()
    return df

def rolling_volatility(df, window=20, price_col="adj_close", annualize=True):
    df = daily_return(df, price_col)
    vol = df["return"].rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS / (TRADING_DAYS/window))
    df[f"vol_{window}"] = vol
    return df

def sharpe_ratio(returns, risk_free_rate=0.0, window=252):
    ex = returns - risk_free_rate / TRADING_DAYS
    rolling_mean = ex.rolling(window=window).mean() * TRADING_DAYS
    rolling_std = ex.rolling(window=window).std() * np.sqrt(TRADING_DAYS)
    return rolling_mean / rolling_std

def build_indicators(df, price_col="adj_close"):
    df = prepare(df)
    df["return"] = df[price_col].pct_change()
    df["ma20"] = df[price_col].rolling(20).mean()
    df["ma50"] = df[price_col].rolling(50).mean()
    df["vol20"] = df["return"].rolling(20).std() * np.sqrt(TRADING_DAYS / (TRADING_DAYS/20))
    df["sharpe252"] = sharpe_ratio(df["return"], 0.0, 252)
    return df
