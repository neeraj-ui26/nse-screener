"""
Backtest for the Nifty 500 screener buckets.

For every historical date (with enough lookback), classifies each stock into
the same buckets as screener.py, then measures forward returns over 5/10/20
trading days. Compares each bucket's average forward return against the
overall market average for that same period - this tells you whether the
classification actually carries predictive signal, not just a nice label.

No look-ahead bias: classification on day T only uses data up to and
including day T. Forward returns use day T+N close vs day T close.

Run: python backtest.py
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "nifty500.db"
MIN_ROWS_REQUIRED = 250
RSI_PERIOD = 14
SMA_SHORT = 50
SMA_LONG = 200
VOLUME_LOOKBACK = 20
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_HEALTHY_LOW = 40
RSI_HEALTHY_HIGH = 65
FORWARD_HORIZONS = [5, 10, 20]     # trading days ahead
SAMPLE_EVERY_N_DAYS = 5             # reduce overlap bias between consecutive signals


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


def classify_series(df):
    """Vectorized bucket classification for every day in the dataframe."""
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
    choices = ["OVERBOUGHT_WATCH", "FRESH_MOMENTUM", "WEAK_AVOID"]
    df["bucket"] = np.select(conditions, choices, default="NEUTRAL")

    # rows without enough history to have a valid sma200/rsi get marked invalid
    df.loc[df["sma200"].isna() | df["rsi"].isna(), "bucket"] = "INVALID"

    for h in FORWARD_HORIZONS:
        df[f"fwd_ret_{h}d"] = df["close"].shift(-h) / df["close"] - 1

    return df


def run_backtest(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute("SELECT symbol FROM stocks").fetchall()]

    all_rows = []
    skipped = 0

    for symbol in symbols:
        df = pd.read_sql(
            "SELECT date, close FROM price_history WHERE symbol = ? ORDER BY date",
            conn, params=(symbol,),
        )
        if len(df) < MIN_ROWS_REQUIRED:
            skipped += 1
            continue

        df["date"] = pd.to_datetime(df["date"])
        classified = classify_series(df)
        classified["symbol"] = symbol

        # sample every N days to reduce overlap between consecutive near-identical signals
        classified = classified.iloc[::SAMPLE_EVERY_N_DAYS]
        classified = classified[classified["bucket"] != "INVALID"]
        # drop rows near the end with no forward data
        classified = classified.dropna(subset=[f"fwd_ret_{h}d" for h in FORWARD_HORIZONS])

        all_rows.append(classified[["symbol", "date", "bucket"] + [f"fwd_ret_{h}d" for h in FORWARD_HORIZONS]])

    conn.close()

    combined = pd.concat(all_rows, ignore_index=True)
    return combined, skipped


def print_backtest_report(combined, skipped):
    print(f"\n{'='*75}")
    print("SCREENER BACKTEST REPORT")
    print(f"{'='*75}")
    print(f"Symbols skipped (insufficient history): {skipped}")
    print(f"Total classified samples (all dates, all symbols, sampled every {SAMPLE_EVERY_N_DAYS} days): {len(combined)}\n")

    # Overall market baseline: average forward return across ALL samples regardless of bucket
    baseline = {h: combined[f"fwd_ret_{h}d"].mean() * 100 for h in FORWARD_HORIZONS}

    print(f"{'Bucket':<20}{'Count':<10}" + "".join(f"{f'Avg {h}d Fwd Ret':<18}" for h in FORWARD_HORIZONS))
    print("-" * 75)

    for bucket in ["FRESH_MOMENTUM", "OVERBOUGHT_WATCH", "NEUTRAL", "WEAK_AVOID"]:
        subset = combined[combined["bucket"] == bucket]
        count = len(subset)
        if count == 0:
            print(f"{bucket:<20}{count:<10}(no samples)")
            continue
        row = f"{bucket:<20}{count:<10}"
        for h in FORWARD_HORIZONS:
            avg_ret = subset[f"fwd_ret_{h}d"].mean() * 100
            row += f"{avg_ret:>+7.2f}%{'':<9}"
        print(row)

    print("-" * 75)
    row = f"{'ALL (baseline)':<20}{len(combined):<10}"
    for h in FORWARD_HORIZONS:
        row += f"{baseline[h]:>+7.2f}%{'':<9}"
    print(row)

    print(f"\n{'='*75}")
    print("INTERPRETATION")
    print(f"{'='*75}")
    for h in FORWARD_HORIZONS:
        fm = combined[combined["bucket"] == "FRESH_MOMENTUM"][f"fwd_ret_{h}d"].mean() * 100
        wa = combined[combined["bucket"] == "WEAK_AVOID"][f"fwd_ret_{h}d"].mean() * 100
        edge = fm - baseline[h]
        print(f"  {h}-day: Fresh Momentum beat the overall average by {edge:+.2f} percentage points. "
              f"Weak/Avoid averaged {wa:+.2f}%.")
    print(f"{'='*75}\n")

    # Win rate: what % of FRESH_MOMENTUM signals had a positive forward return
    print("WIN RATES (% of signals with positive forward return):")
    for bucket in ["FRESH_MOMENTUM", "OVERBOUGHT_WATCH", "WEAK_AVOID"]:
        subset = combined[combined["bucket"] == bucket]
        if len(subset) == 0:
            continue
        for h in FORWARD_HORIZONS:
            win_rate = (subset[f"fwd_ret_{h}d"] > 0).mean() * 100
            print(f"  {bucket:<20} {h}-day win rate: {win_rate:.1f}%")
    print()


if __name__ == "__main__":
    combined, skipped = run_backtest()
    print_backtest_report(combined, skipped)
