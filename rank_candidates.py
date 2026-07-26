"""
Weekly Candidate Ranker

Takes the FRESH_MOMENTUM bucket from the screener and ranks stocks within it
using a composite score built from signals that are each independently
sensible for short-term (next week) continuation:

  1. RSI proximity to the "sweet spot" (~55) - not just healthy, but in the
     part of the range with the most room to run before overbought.
  2. 1-month relative strength - price change over the last ~20 trading days,
     ranked against the rest of the universe (not absolute return).
  3. Volume confirmation - current volume vs 20-day average.
  4. Trend quality - how cleanly price is stacked above SMA50 above SMA200
     (distance, not just a binary pass/fail).

Each signal is converted to a percentile rank (0-100) across the eligible
universe, then averaged. This avoids arbitrary dollar-value weightings and
keeps the score interpretable: "how does this stock rank vs its peers on
each dimension, on average."

IMPORTANT: this is a ranking tool, not a prediction. It surfaces stocks that
score well on signals that showed a modest, real backtested edge - it does
not guarantee any individual stock's near-term return.

Run: python rank_candidates.py
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "nifty500.db"
MIN_ROWS_REQUIRED = 250
RSI_PERIOD = 14
RSI_SWEET_SPOT = 55
SMA_SHORT = 50
SMA_LONG = 200
VOLUME_LOOKBACK = 20
MOMENTUM_LOOKBACK_1M = 20   # ~1 trading month
TOP_N = 15


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


def compute_features(df):
    """Compute all indicators needed for scoring, return the latest row's features."""
    df = df.copy()
    df["sma50"] = df["close"].rolling(SMA_SHORT).mean()
    df["sma200"] = df["close"].rolling(SMA_LONG).mean()
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["vol_avg20"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()
    df["ret_1m"] = df["close"].pct_change(MOMENTUM_LOOKBACK_1M)
    df["trend_stack_pct"] = (df["sma50"] - df["sma200"]) / df["sma200"] * 100

    latest = df.iloc[-1]
    if pd.isna(latest["sma200"]) or pd.isna(latest["rsi"]) or pd.isna(latest["ret_1m"]):
        return None

    is_uptrend = latest["close"] > latest["sma50"] > latest["sma200"]
    is_healthy_rsi = 40 <= latest["rsi"] <= 65
    if not (is_uptrend and is_healthy_rsi):
        return None  # only rank stocks already in Fresh Momentum

    return {
        "close": latest["close"],
        "rsi": latest["rsi"],
        "rsi_distance_from_sweet_spot": abs(latest["rsi"] - RSI_SWEET_SPOT),
        "ret_1m": latest["ret_1m"],
        "volume_ratio": latest["volume"] / latest["vol_avg20"] if latest["vol_avg20"] > 0 else np.nan,
        "trend_stack_pct": latest["trend_stack_pct"],
    }


def build_candidate_table(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute("SELECT symbol FROM stocks").fetchall()]

    rows = []
    for symbol in symbols:
        df = pd.read_sql(
            "SELECT date, close, volume FROM price_history WHERE symbol = ? ORDER BY date",
            conn, params=(symbol,),
        )
        if len(df) < MIN_ROWS_REQUIRED:
            continue
        feats = compute_features(df)
        if feats is None:
            continue
        feats["symbol"] = symbol
        rows.append(feats)

    conn.close()

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)

    # Convert each signal to a 0-100 percentile rank across the eligible universe.
    # Lower rsi_distance_from_sweet_spot is better -> invert its rank.
    table["score_rsi"] = (1 - table["rsi_distance_from_sweet_spot"].rank(pct=True)) * 100
    table["score_ret_1m"] = table["ret_1m"].rank(pct=True) * 100
    table["score_volume"] = table["volume_ratio"].rank(pct=True) * 100
    table["score_trend"] = table["trend_stack_pct"].rank(pct=True) * 100

    table["composite_score"] = table[["score_rsi", "score_ret_1m", "score_volume", "score_trend"]].mean(axis=1)

    table = table.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return table


def print_candidates(table, top_n=TOP_N):
    print(f"\n{'='*90}")
    print("WEEKLY CANDIDATE SHORTLIST — Top-ranked Fresh Momentum stocks")
    print(f"{'='*90}")
    print("Ranking, not prediction. These score well on signals with a modest backtested edge.")
    print("Not a buy signal on their own — use for further research, position sizing, and risk control.\n")

    if table.empty:
        print("No eligible candidates found (either no stocks in Fresh Momentum, or insufficient data).")
        return

    print(f"{'Rank':<6}{'Symbol':<15}{'Close':<10}{'RSI':<8}{'1M Ret':<10}{'Vol Ratio':<12}{'Score':<8}")
    print("-" * 90)
    for i, row in table.head(top_n).iterrows():
        print(f"{i+1:<6}{row['symbol']:<15}{row['close']:<10.2f}{row['rsi']:<8.1f}"
              f"{row['ret_1m']*100:>+7.2f}%  {row['volume_ratio']:<10.2f}{row['composite_score']:<8.1f}")

    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    table = build_candidate_table()
    print_candidates(table)
