"""
Multi-Timeframe Analysis

Adds two faster tiers on top of your existing long-term screener logic:

  PRIMARY (unchanged)  - SMA50/SMA200, RSI14 - the long-term trend filter
                          you already built and backtested. Untouched.
  SWING   (new)        - SMA20/SMA50 - is the medium-term wave (roughly a
                          3-6 month view) agreeing with the primary trend?
  TRIGGER (new)        - price vs SMA10 + 5-day rate of change - is the
                          immediate ~1 month move helping or fighting the
                          bigger picture right now?

These are NOT three independent opinions on separate chopped-up date
ranges. All three are computed on the SAME full price history - they just
use different moving-average speeds to look at different scales of
structure, the way real multi-timeframe technical analysis is done.
This avoids the "SMA200 needs 200 days but a 1-month window only has 20
days" problem entirely.

A single ALIGNMENT verdict synthesizes the three tiers into one plain-English
read, instead of three badges that could look contradictory on their own.

Does not modify screener.py, backtest.py, rank_candidates.py, or
analyze_stock.py - this is a standalone addition.

Run: python multi_timeframe.py SYMBOL
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf

NIFTY500_DB = "nifty500.db"
MIN_ROWS_REQUIRED = 250
RSI_PERIOD = 14


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


def load_from_db(symbol, db_path=NIFTY500_DB):
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM price_history WHERE symbol = ? ORDER BY date",
        conn, params=(symbol,),
    )
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_live(symbol, period="5y"):
    yf_symbol = symbol.upper().strip()
    if not yf_symbol.endswith(".NS"):
        yf_symbol += ".NS"
    hist = yf.Ticker(yf_symbol).history(period=period, interval="1d")
    if hist.empty:
        return None
    hist = hist.reset_index()
    hist = hist.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                 "Low": "low", "Close": "close", "Volume": "volume"})
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    return hist[["date", "open", "high", "low", "close", "volume"]]


def get_price_data(symbol):
    """Try local DB first (fast, no network), fall back to live fetch."""
    df = load_from_db(symbol)
    if df is not None and len(df) >= MIN_ROWS_REQUIRED:
        return df, "local database"
    df = fetch_live(symbol)
    if df is not None:
        return df, "live fetch (yfinance)"
    return None, None


def classify_primary(df):
    """Long-term tier - identical logic to screener.py. Unchanged."""
    close = df["close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi = compute_rsi(close, RSI_PERIOD)

    c, s50, s200, r = close.iloc[-1], sma50.iloc[-1], sma200.iloc[-1], rsi.iloc[-1]
    if pd.isna(s200) or pd.isna(r):
        return None

    is_uptrend = c > s50 > s200
    is_downtrend = c < s50 < s200

    if is_uptrend and r > 70:
        bucket = "OVERBOUGHT_WATCH"
    elif is_uptrend and 40 <= r <= 65:
        bucket = "FRESH_MOMENTUM"
    elif is_downtrend or r < 30:
        bucket = "WEAK_AVOID"
    else:
        bucket = "NEUTRAL"

    return {"close": round(c, 2), "sma50": round(s50, 2), "sma200": round(s200, 2),
            "rsi": round(r, 2), "trend": "UP" if is_uptrend else ("DOWN" if is_downtrend else "SIDEWAYS"),
            "bucket": bucket}


def classify_swing(df):
    """Medium-term tier (~3-6 month structure) - faster SMAs, same RSI."""
    close = df["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = compute_rsi(close, RSI_PERIOD)

    c, s20, s50, r = close.iloc[-1], sma20.iloc[-1], sma50.iloc[-1], rsi.iloc[-1]
    if pd.isna(s50) or pd.isna(r):
        return None

    is_uptrend = c > s20 > s50
    is_downtrend = c < s20 < s50
    trend = "UP" if is_uptrend else ("DOWN" if is_downtrend else "SIDEWAYS")

    return {"sma20": round(s20, 2), "sma50": round(s50, 2), "rsi": round(r, 2), "trend": trend}


def classify_trigger(df):
    """Immediate (~1 month) tier - is the last few weeks helping or fighting the bigger trend."""
    close = df["close"]
    sma10 = close.rolling(10).mean()
    ret_5d = close.pct_change(5)

    c, s10, r5 = close.iloc[-1], sma10.iloc[-1], ret_5d.iloc[-1]
    if pd.isna(s10) or pd.isna(r5):
        return None

    above_sma10 = c > s10
    momentum = "POSITIVE" if r5 > 0 else "NEGATIVE"

    return {"sma10": round(s10, 2), "ret_5d_pct": round(r5 * 100, 2),
            "above_sma10": above_sma10, "momentum": momentum}


def synthesize_alignment(primary, swing, trigger):
    """Turn three tier reads into one plain-English verdict."""
    primary_bullish = primary["bucket"] in ("FRESH_MOMENTUM", "OVERBOUGHT_WATCH")
    primary_bearish = primary["bucket"] == "WEAK_AVOID"
    swing_bullish = swing["trend"] == "UP"
    swing_bearish = swing["trend"] == "DOWN"
    trigger_bullish = trigger["above_sma10"] and trigger["momentum"] == "POSITIVE"
    trigger_bearish = (not trigger["above_sma10"]) and trigger["momentum"] == "NEGATIVE"

    if primary_bullish and swing_bullish and trigger_bullish:
        return "STRONG ALIGNED UPTREND — all three timeframes agree bullish."
    if primary_bullish and swing_bullish and not trigger_bullish:
        return "UPTREND INTACT, SHORT-TERM PAUSE — bigger picture still bullish, last month is cooling off. Watch for stabilization before adding."
    if primary_bullish and swing_bearish:
        return "LONG-TERM BULLISH BUT LOSING MOMENTUM — the medium-term wave has turned down inside a longer uptrend. Caution, not yet a clear buy or sell."
    if primary_bearish and swing_bullish:
        return "POSSIBLE EARLY REVERSAL — long-term still weak, but medium-term has turned up. Unconfirmed — could be a real reversal or a countertrend bounce."
    if primary_bearish and swing_bearish and trigger_bearish:
        return "CONFIRMED DOWNTREND AT MULTIPLE TIMEFRAMES — avoid."
    if primary_bearish and swing_bearish:
        return "DOWNTREND — long-term and medium-term both weak."
    if primary["bucket"] == "NEUTRAL" and swing_bullish and trigger_bullish:
        return "EARLY BREAKOUT WATCH — long-term has been flat (SMA50≈SMA200, a sideways base), but the medium and short term are both turning up. Not yet a confirmed trend since the long-term indicator hasn't caught up - worth watching for follow-through over the next few weeks, not an immediate signal on its own."
    if primary["bucket"] == "NEUTRAL" and swing_bearish and trigger_bearish:
        return "FADING WITHIN A SIDEWAYS BASE — long-term flat, and short-term is now weakening too. No edge here."
    return "MIXED / NO CLEAR ALIGNMENT — timeframes disagree, no clean signal right now."


def analyze(symbol):
    df, source = get_price_data(symbol)
    if df is None:
        print(f"Could not get data for '{symbol}'.")
        return
    if len(df) < MIN_ROWS_REQUIRED:
        print(f"Only {len(df)} days available - not enough for the primary (long-term) tier.")
        return

    primary = classify_primary(df)
    swing = classify_swing(df)
    trigger = classify_trigger(df)

    if not all([primary, swing, trigger]):
        print("Not enough data to compute all three tiers.")
        return

    print(f"\n{'='*72}")
    print(f"{symbol.upper()} — MULTI-TIMEFRAME ANALYSIS  (data source: {source})")
    print(f"{'='*72}")

    print(f"\nPRIMARY (long-term, ~1-5yr structure) — unchanged from screener.py")
    print(f"  Close={primary['close']}  SMA50={primary['sma50']}  SMA200={primary['sma200']}  RSI={primary['rsi']}")
    print(f"  Trend: {primary['trend']}   Bucket: {primary['bucket']}")

    print(f"\nSWING (medium-term, ~3-6mo structure)")
    print(f"  SMA20={swing['sma20']}  SMA50={swing['sma50']}  RSI={swing['rsi']}")
    print(f"  Trend: {swing['trend']}")

    print(f"\nTRIGGER (immediate, ~1mo)")
    print(f"  SMA10={trigger['sma10']}  5-day return={trigger['ret_5d_pct']:+.2f}%")
    print(f"  Above SMA10: {trigger['above_sma10']}   Momentum: {trigger['momentum']}")

    verdict = synthesize_alignment(primary, swing, trigger)
    print(f"\n{'-'*72}")
    print(f"ALIGNMENT VERDICT:")
    print(f"  {verdict}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = input("Enter NSE stock symbol (e.g. TATAMOTORS): ").strip()
    analyze(symbol)
