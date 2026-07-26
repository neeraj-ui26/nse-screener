"""
Validates rank_candidates.py's composite score using a walk-forward backtest.

For each historical date, computes the composite score for all Fresh Momentum
stocks on that date, splits them into top-half vs bottom-half by score, then
checks forward returns. If the ranking is meaningful, top-half should beat
bottom-half - not just "Fresh Momentum beats baseline" (already proven),
but "higher-ranked WITHIN Fresh Momentum beats lower-ranked."
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
MOMENTUM_LOOKBACK_1M = 20
FORWARD_HORIZONS = [5, 10, 20]
SAMPLE_EVERY_N_DAYS = 5


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


def compute_all_features(df):
    df = df.copy()
    df["sma50"] = df["close"].rolling(SMA_SHORT).mean()
    df["sma200"] = df["close"].rolling(SMA_LONG).mean()
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["vol_avg20"] = df["volume"].rolling(VOLUME_LOOKBACK).mean()
    df["ret_1m"] = df["close"].pct_change(MOMENTUM_LOOKBACK_1M)
    df["trend_stack_pct"] = (df["sma50"] - df["sma200"]) / df["sma200"] * 100
    df["volume_ratio"] = df["volume"] / df["vol_avg20"]
    df["rsi_distance"] = (df["rsi"] - RSI_SWEET_SPOT).abs()

    is_uptrend = (df["close"] > df["sma50"]) & (df["sma50"] > df["sma200"])
    is_healthy_rsi = (df["rsi"] >= 40) & (df["rsi"] <= 65)
    df["is_fresh_momentum"] = is_uptrend & is_healthy_rsi

    for h in FORWARD_HORIZONS:
        df[f"fwd_ret_{h}d"] = df["close"].shift(-h) / df["close"] - 1

    return df


def run_ranking_validation(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    symbols = [r[0] for r in conn.execute("SELECT symbol FROM stocks").fetchall()]

    all_rows = []
    for symbol in symbols:
        df = pd.read_sql(
            "SELECT date, close, volume FROM price_history WHERE symbol = ? ORDER BY date",
            conn, params=(symbol,),
        )
        if len(df) < MIN_ROWS_REQUIRED:
            continue
        df["date"] = pd.to_datetime(df["date"])
        feat = compute_all_features(df)
        feat["symbol"] = symbol
        feat = feat.iloc[::SAMPLE_EVERY_N_DAYS]
        feat = feat[feat["is_fresh_momentum"]]
        feat = feat.dropna(subset=[f"fwd_ret_{h}d" for h in FORWARD_HORIZONS] +
                                   ["ret_1m", "volume_ratio", "trend_stack_pct", "rsi_distance"])
        all_rows.append(feat[["symbol", "date", "rsi_distance", "ret_1m", "volume_ratio",
                               "trend_stack_pct"] + [f"fwd_ret_{h}d" for h in FORWARD_HORIZONS]])
    conn.close()

    combined = pd.concat(all_rows, ignore_index=True)

    # Rank WITHIN each date (cross-sectional ranking, like the real tool would do)
    combined["score_rsi"] = combined.groupby("date")["rsi_distance"].rank(pct=True, ascending=False) * 100
    combined["score_ret_1m"] = combined.groupby("date")["ret_1m"].rank(pct=True) * 100
    combined["score_volume"] = combined.groupby("date")["volume_ratio"].rank(pct=True) * 100
    combined["score_trend"] = combined.groupby("date")["trend_stack_pct"].rank(pct=True) * 100
    combined["composite_score"] = combined[["score_rsi", "score_ret_1m", "score_volume", "score_trend"]].mean(axis=1)

    # Split into top-half vs bottom-half BY DATE (cross-sectional, not global)
    # Using a rank-percentile median split instead of qcut - avoids qcut's failure
    # mode when many stocks share the same composite_score (duplicate bin edges).
    combined["score_pct_rank"] = combined.groupby("date")["composite_score"].rank(pct=True)
    combined["rank_half"] = np.where(combined["score_pct_rank"] >= 0.5, "TOP_HALF", "BOTTOM_HALF")

    return combined


def print_validation_report(combined):
    print(f"\n{'='*80}")
    print("RANKING MODEL VALIDATION")
    print(f"{'='*80}")
    print(f"Total Fresh Momentum samples used: {len(combined)}\n")
    print(f"{'Group':<15}{'Count':<10}" + "".join(f"{f'Avg {h}d Fwd Ret':<18}" for h in FORWARD_HORIZONS))
    print("-" * 80)
    for group in ["TOP_HALF", "BOTTOM_HALF"]:
        subset = combined[combined["rank_half"] == group]
        row = f"{group:<15}{len(subset):<10}"
        for h in FORWARD_HORIZONS:
            row += f"{subset[f'fwd_ret_{h}d'].mean()*100:>+7.2f}%{'':<9}"
        print(row)

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    for h in FORWARD_HORIZONS:
        top = combined[combined["rank_half"] == "TOP_HALF"][f"fwd_ret_{h}d"].mean() * 100
        bot = combined[combined["rank_half"] == "BOTTOM_HALF"][f"fwd_ret_{h}d"].mean() * 100
        edge = top - bot
        verdict = "ranking adds value" if edge > 0 else "ranking does NOT separate winners here"
        print(f"  {h}-day: Top-half beat bottom-half by {edge:+.2f}pp -> {verdict}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    combined = run_ranking_validation()
    print_validation_report(combined)
