"""
Nifty 500 Technical Screener
Logic: Trend (SMA50/SMA200) + Momentum (RSI14) + Volume confirmation
For overbought stocks, computes Fibonacci retracement re-entry levels.

Run: python screener.py
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "nifty500.db"
MIN_ROWS_REQUIRED = 250          # need enough history for SMA200 + buffer
RSI_PERIOD = 14
SMA_SHORT = 50
SMA_LONG = 200
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_HEALTHY_LOW = 40
RSI_HEALTHY_HIGH = 65
FIB_SWING_LOOKBACK = 90          # days to look back for swing high/low


def load_price_data(conn, symbol):
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM price_history WHERE symbol = ? ORDER BY date",
        conn, params=(symbol,),
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


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
    # Edge case: no down days at all in the window -> RSI = 100 (max overbought)
    all_gains = valid & (avg_loss == 0) & (avg_gain > 0)
    rsi[all_gains] = 100
    # Edge case: no movement at all -> RSI = 50 (neutral, undefined direction)
    no_movement = valid & (avg_loss == 0) & (avg_gain == 0)
    rsi[no_movement] = 50
    return rsi


def compute_fibonacci_levels(df, lookback):
    """Fib retracement from the swing low to swing high in the lookback window."""
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


def classify_stock(df):
    """Apply trend + momentum + volume logic. Returns a result dict or None if not enough data."""
    if len(df) < MIN_ROWS_REQUIRED:
        return None

    df = df.copy()
    df["sma50"] = df["close"].rolling(SMA_SHORT).mean()
    df["sma200"] = df["close"].rolling(SMA_LONG).mean()
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["vol_avg20"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()

    latest = df.iloc[-1]
    if pd.isna(latest["sma200"]) or pd.isna(latest["rsi"]):
        return None

    close = latest["close"]
    sma50 = latest["sma50"]
    sma200 = latest["sma200"]
    rsi = latest["rsi"]
    volume = latest["volume"]
    vol_avg20 = latest["vol_avg20"]

    is_uptrend = close > sma50 > sma200
    is_downtrend = close < sma50 < sma200
    volume_confirmed = volume > VOLUME_SPIKE_MULTIPLIER * vol_avg20

    # Bucket classification
    if is_uptrend and rsi > RSI_OVERBOUGHT:
        bucket = "OVERBOUGHT_WATCH"
    elif is_uptrend and RSI_HEALTHY_LOW <= rsi <= RSI_HEALTHY_HIGH:
        bucket = "FRESH_MOMENTUM"
    elif is_downtrend or rsi < RSI_OVERSOLD:
        bucket = "WEAK_AVOID"
    else:
        bucket = "NEUTRAL"

    result = {
        "close": round(close, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "rsi": round(rsi, 2),
        "volume_confirmed": volume_confirmed,
        "trend": "UP" if is_uptrend else ("DOWN" if is_downtrend else "SIDEWAYS"),
        "bucket": bucket,
    }

    if bucket == "OVERBOUGHT_WATCH":
        result["fib_levels"] = compute_fibonacci_levels(df, FIB_SWING_LOOKBACK)

    return result


def run_screener(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute("SELECT symbol FROM stocks").fetchall()]

    buckets = {"FRESH_MOMENTUM": [], "OVERBOUGHT_WATCH": [], "WEAK_AVOID": [], "NEUTRAL": []}
    skipped = 0

    for symbol in symbols:
        df = load_price_data(conn, symbol)
        result = classify_stock(df)
        if result is None:
            skipped += 1
            continue
        result["symbol"] = symbol
        buckets[result["bucket"]].append(result)

    conn.close()
    return buckets, skipped


def print_report(buckets, skipped):
    print(f"\n{'='*70}")
    print(f"NIFTY 500 SCREENER REPORT")
    print(f"{'='*70}")
    print(f"Skipped (insufficient history): {skipped}\n")

    print(f"--- FRESH MOMENTUM ({len(buckets['FRESH_MOMENTUM'])}) ---")
    print("Uptrend + healthy RSI (40-65) + not yet overbought\n")
    for r in sorted(buckets["FRESH_MOMENTUM"], key=lambda x: x["rsi"], reverse=True):
        vol_flag = " [VOL SPIKE]" if r["volume_confirmed"] else ""
        print(f"  {r['symbol']:<15} Close={r['close']:<10} RSI={r['rsi']:<6} SMA50={r['sma50']:<10}{vol_flag}")

    print(f"\n--- OVERBOUGHT — WATCH FOR RE-ENTRY ({len(buckets['OVERBOUGHT_WATCH'])}) ---")
    print("Uptrend but RSI > 70 — not a fresh entry. Fib levels = potential re-entry zones on pullback.\n")
    for r in sorted(buckets["OVERBOUGHT_WATCH"], key=lambda x: x["rsi"], reverse=True):
        print(f"  {r['symbol']:<15} Close={r['close']:<10} RSI={r['rsi']}")
        fib = r["fib_levels"]
        print(f"      Swing: {fib['swing_low']} -> {fib['swing_high']}")
        print(f"      Re-entry zones: 38.2%={fib['fib_38.2']}  50%={fib['fib_50.0']}  61.8%={fib['fib_61.8']}")

    print(f"\n--- WEAK / AVOID ({len(buckets['WEAK_AVOID'])}) ---")
    for r in sorted(buckets["WEAK_AVOID"], key=lambda x: x["rsi"]):
        print(f"  {r['symbol']:<15} Close={r['close']:<10} RSI={r['rsi']:<6} Trend={r['trend']}")

    print(f"\n(Neutral/sideways stocks: {len(buckets['NEUTRAL'])} — not shown, no clear signal)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    buckets, skipped = run_screener()
    print_report(buckets, skipped)
