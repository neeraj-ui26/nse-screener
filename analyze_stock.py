"""
Single Stock Analyzer

Paste any NSE stock symbol (doesn't need to be in your Nifty 500 database) -
this fetches fresh data via yfinance and runs it through the same
classification logic as screener.py, plus a mini-backtest on that stock's
own history.

IMPORTANT: a single stock's backtest has far fewer independent samples than
the full 432-stock, 5-year backtest already validated. Treat this as
context, not a standalone verdict - the number of historical "Fresh
Momentum" instances for one stock alone is usually too small to be
statistically reliable by itself.

Run: python analyze_stock.py
Then enter a symbol when prompted, e.g. TATAMOTORS, INFY, ZOMATO
"""

import sys
import sqlite3
import os
import pandas as pd
import numpy as np
import yfinance as yf

RSI_PERIOD = 14
SMA_SHORT = 50
SMA_LONG = 200
VOLUME_LOOKBACK = 20
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_HEALTHY_LOW = 40
RSI_HEALTHY_HIGH = 65
FIB_SWING_LOOKBACK = 90
FORWARD_HORIZONS = [5, 10, 20]
SAMPLE_EVERY_N_DAYS = 5
NIFTY500_DB = "nifty500.db"  # used only for optional percentile context


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rsi = pd.Series(index=close.index, dtype=float)
    valid = avg_gain.notna() & avg_loss.notna()
    rs = avg_gain[valid] / avg_loss[valid].replace(0, np.nan)
    rsi[valid] = 100 - (100 / (1 + rs))
    all_gains = valid & (avg_loss == 0) & (avg_gain > 0)
    rsi[all_gains] = 100
    no_movement = valid & (avg_loss == 0) & (avg_gain == 0)
    rsi[no_movement] = 50
    return rsi


def fetch_stock_data(symbol, period="5y"):
    yf_symbol = symbol.upper().strip()
    if not yf_symbol.endswith(".NS"):
        yf_symbol += ".NS"
    ticker = yf.Ticker(yf_symbol)
    hist = ticker.history(period=period, interval="1d")
    if hist.empty:
        return None
    hist = hist.reset_index()
    hist = hist.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    return hist[["date", "open", "high", "low", "close", "volume"]]


def compute_fibonacci_levels(df, lookback):
    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    diff = swing_high - swing_low
    return {
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "fib_38.2": round(swing_high - 0.382 * diff, 2),
        "fib_50.0": round(swing_high - 0.500 * diff, 2),
        "fib_61.8": round(swing_high - 0.618 * diff, 2),
    }


def classify_current(df):
    df = df.copy()
    df["sma50"] = df["close"].rolling(SMA_SHORT).mean()
    df["sma200"] = df["close"].rolling(SMA_LONG).mean()
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["vol_avg20"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()

    latest = df.iloc[-1]
    if pd.isna(latest["sma200"]) or pd.isna(latest["rsi"]):
        return None

    close, sma50, sma200, rsi = latest["close"], latest["sma50"], latest["sma200"], latest["rsi"]
    volume, vol_avg20 = latest["volume"], latest["vol_avg20"]

    is_uptrend = close > sma50 > sma200
    is_downtrend = close < sma50 < sma200
    volume_confirmed = volume > 1.5 * vol_avg20

    if is_uptrend and rsi > RSI_OVERBOUGHT:
        bucket = "OVERBOUGHT_WATCH"
    elif is_uptrend and RSI_HEALTHY_LOW <= rsi <= RSI_HEALTHY_HIGH:
        bucket = "FRESH_MOMENTUM"
    elif is_downtrend or rsi < RSI_OVERSOLD:
        bucket = "WEAK_AVOID"
    else:
        bucket = "NEUTRAL"

    result = {
        "date": latest["date"], "close": round(close, 2), "sma50": round(sma50, 2),
        "sma200": round(sma200, 2), "rsi": round(rsi, 2), "volume_confirmed": volume_confirmed,
        "trend": "UP" if is_uptrend else ("DOWN" if is_downtrend else "SIDEWAYS"), "bucket": bucket,
    }
    if bucket == "OVERBOUGHT_WATCH":
        result["fib_levels"] = compute_fibonacci_levels(df, FIB_SWING_LOOKBACK)
    return result


def mini_backtest(df):
    """Classify every historical day for THIS stock and measure forward returns per bucket."""
    df = df.copy()
    df["sma50"] = df["close"].rolling(SMA_SHORT).mean()
    df["sma200"] = df["close"].rolling(SMA_LONG).mean()
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)

    is_uptrend = (df["close"] > df["sma50"]) & (df["sma50"] > df["sma200"])
    is_downtrend = (df["close"] < df["sma50"]) & (df["sma50"] < df["sma200"])
    conditions = [
        is_uptrend & (df["rsi"] > RSI_OVERBOUGHT),
        is_uptrend & (df["rsi"] >= RSI_HEALTHY_LOW) & (df["rsi"] <= RSI_HEALTHY_HIGH),
        is_downtrend | (df["rsi"] < RSI_OVERSOLD),
    ]
    df["bucket"] = np.select(conditions, ["OVERBOUGHT_WATCH", "FRESH_MOMENTUM", "WEAK_AVOID"], default="NEUTRAL")
    df.loc[df["sma200"].isna() | df["rsi"].isna(), "bucket"] = "INVALID"

    for h in FORWARD_HORIZONS:
        df[f"fwd_ret_{h}d"] = df["close"].shift(-h) / df["close"] - 1

    sampled = df.iloc[::SAMPLE_EVERY_N_DAYS]
    sampled = sampled[sampled["bucket"] != "INVALID"]
    sampled = sampled.dropna(subset=[f"fwd_ret_{h}d" for h in FORWARD_HORIZONS])
    return sampled


def print_mini_backtest(sampled):
    print(f"\n--- THIS STOCK'S OWN HISTORY (illustrative only - small sample) ---")
    print(f"{'Bucket':<20}{'Count':<10}" + "".join(f"{f'Avg {h}d Fwd Ret':<18}" for h in FORWARD_HORIZONS))
    print("-" * 75)
    for bucket in ["FRESH_MOMENTUM", "OVERBOUGHT_WATCH", "NEUTRAL", "WEAK_AVOID"]:
        subset = sampled[sampled["bucket"] == bucket]
        if len(subset) == 0:
            print(f"{bucket:<20}{'0':<10}(no historical instances)")
            continue
        row = f"{bucket:<20}{len(subset):<10}"
        for h in FORWARD_HORIZONS:
            row += f"{subset[f'fwd_ret_{h}d'].mean()*100:>+7.2f}%{'':<9}"
        print(row)

    fm_count = len(sampled[sampled["bucket"] == "FRESH_MOMENTUM"])
    if fm_count < 20:
        print(f"\n  NOTE: only {fm_count} historical 'Fresh Momentum' samples for this stock alone.")
        print("  Too few to draw a reliable conclusion - treat as context, not a verdict.")
        print("  The full 432-stock backtest (backtest.py) is the statistically meaningful one.")


def analyze(symbol):
    print(f"\nFetching data for {symbol}...")
    df = fetch_stock_data(symbol)
    if df is None:
        print(f"No data found for '{symbol}'. Check the symbol is correct and NSE-listed.")
        return
    if len(df) < 250:
        print(f"Only {len(df)} days of history available - not enough for reliable SMA200/RSI classification.")
        return

    result = classify_current(df)
    if result is None:
        print("Could not classify - insufficient recent data.")
        return

    print(f"\n{'='*70}")
    print(f"{symbol.upper()} — CURRENT CLASSIFICATION (as of {result['date'].strftime('%Y-%m-%d')})")
    print(f"{'='*70}")
    print(f"Close: {result['close']}   SMA50: {result['sma50']}   SMA200: {result['sma200']}")
    print(f"RSI(14): {result['rsi']}   Trend: {result['trend']}   Volume confirmed: {result['volume_confirmed']}")
    print(f"BUCKET: {result['bucket']}")

    if result["bucket"] == "OVERBOUGHT_WATCH":
        fib = result["fib_levels"]
        print(f"\nSwing range: {fib['swing_low']} -> {fib['swing_high']}")
        print(f"Fibonacci re-entry zones: 38.2%={fib['fib_38.2']}  50%={fib['fib_50.0']}  61.8%={fib['fib_61.8']}")

    sampled = mini_backtest(df)
    print_mini_backtest(sampled)
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = input("Enter NSE stock symbol (e.g. TATAMOTORS): ").strip()
    analyze(symbol)
