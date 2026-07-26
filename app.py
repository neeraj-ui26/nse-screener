"""
Nifty Screener — Streamlit Web App (v2)

Adds to the validated v1:
  - Live Nifty 50 / Bank Nifty index ticker strip
  - "Browse Screener" tab: filter the full local database by bucket
    (Fresh Momentum / Overbought / Weak-Avoid / Neutral) - reuses the exact
    same classification logic as screener.py, just surfaced in the UI
    instead of only the terminal.
  - Price charts (SMA50/SMA200 overlay) for any analyzed or browsed stock

Run locally:  streamlit run app.py
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NIFTY500_DB = "nifty500.db"
MIN_ROWS_REQUIRED = 250
RSI_PERIOD = 14

st.set_page_config(page_title="screener.terminal", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
    --bg:#0B0F14; --panel:#12181F; --panel-border:#1F2830;
    --text:#E4E8EC; --muted:#7C8894;
    --green:#3FB68C; --green-dim:#1D3A32;
    --amber:#D9A441; --amber-dim:#3A2F1A;
    --red:#C9564F; --red-dim:#3A2320;
}
.stApp { background-color: var(--bg); }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text); }
.mono { font-family: 'JetBrains Mono', monospace; }
.ticker-strip { display:flex; gap:24px; overflow-x:auto; padding:10px 0 16px;
    border-bottom:1px solid var(--panel-border); margin-bottom:20px;
    font-family:'JetBrains Mono',monospace; font-size:13px; }
.ticker-item { color:var(--muted); white-space:nowrap; }
.ticker-item b { color:var(--text); margin-right:8px; }
.up { color:var(--green); } .down { color:var(--red); }
.card { background: var(--panel); border: 1px solid var(--panel-border);
    border-radius: 8px; padding: 18px 20px; margin-bottom: 16px; }
.card-top { display:flex; justify-content:space-between; align-items:flex-start; }
.symbol { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:18px; color:var(--text); }
.close-price { font-family:'JetBrains Mono',monospace; font-size:22px; font-weight:600; color:var(--text); margin-top:2px;}
.badge { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; letter-spacing:0.04em;
    padding:5px 10px; border-radius:4px; white-space:nowrap; }
.badge-fresh { background:var(--green-dim); color:var(--green); }
.badge-overbought { background:var(--amber-dim); color:var(--amber); }
.badge-weak { background:var(--red-dim); color:var(--red); }
.badge-neutral { background:#22303c; color:var(--muted); }
.rsi-track { position:relative; height:6px; border-radius:3px; margin-top:6px;
    background:linear-gradient(90deg, var(--red) 0%, var(--muted) 30%, var(--green) 50%, var(--muted) 70%, var(--amber) 100%);
    opacity:0.85; }
.rsi-marker { position:absolute; top:-4px; width:2px; height:14px; background:var(--text); }
.rsi-label-row { display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace;
    font-size:10px; color:var(--muted); margin-bottom:4px; margin-top:14px;}
.stat-row { display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace;
    font-size:13px; color:var(--muted); margin-top:8px; }
.stat-row b { color:var(--text); font-weight:500; }
.fib-block { margin-top:14px; padding-top:12px; border-top:1px dashed var(--panel-border); }
.fib-title { font-size:11px; color:var(--amber); font-family:'JetBrains Mono',monospace;
    margin-bottom:8px; letter-spacing:0.03em; }
.fib-row { display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace;
    font-size:13px; padding:3px 0; color:var(--text); }
.fib-row span:first-child { color:var(--muted); }
.verdict-block { margin-top:14px; padding:12px 14px; border-radius:6px; background:#0F151B;
    border-left:3px solid var(--muted); font-size:13px; color:var(--text); line-height:1.5; }
.verdict-title { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--muted);
    letter-spacing:0.03em; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# INDICATOR LOGIC (unchanged from v1 / screener.py)
# ---------------------------------------------------------------------------
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


def compute_fibonacci_levels(df, lookback=90):
    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    diff = swing_high - swing_low
    return {
        "swing_high": round(swing_high, 2), "swing_low": round(swing_low, 2),
        "fib_38.2": round(swing_high - 0.382 * diff, 2),
        "fib_50.0": round(swing_high - 0.500 * diff, 2),
        "fib_61.8": round(swing_high - 0.618 * diff, 2),
    }


def classify_primary(df):
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
    result = {"close": round(c, 2), "sma50": round(s50, 2), "sma200": round(s200, 2),
              "rsi": round(r, 2), "trend": "UP" if is_uptrend else ("DOWN" if is_downtrend else "SIDEWAYS"),
              "bucket": bucket}
    if bucket == "OVERBOUGHT_WATCH":
        result["fib_levels"] = compute_fibonacci_levels(df)
    return result


def classify_swing(df):
    close = df["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    c, s20, s50 = close.iloc[-1], sma20.iloc[-1], sma50.iloc[-1]
    if pd.isna(s50):
        return None
    is_uptrend = c > s20 > s50
    is_downtrend = c < s20 < s50
    return {"trend": "UP" if is_uptrend else ("DOWN" if is_downtrend else "SIDEWAYS")}


def classify_trigger(df):
    close = df["close"]
    sma10 = close.rolling(10).mean()
    ret_5d = close.pct_change(5)
    c, s10, r5 = close.iloc[-1], sma10.iloc[-1], ret_5d.iloc[-1]
    if pd.isna(s10) or pd.isna(r5):
        return None
    return {"above_sma10": c > s10, "momentum": "POSITIVE" if r5 > 0 else "NEGATIVE", "ret_5d_pct": round(r5 * 100, 2)}


def synthesize_alignment(primary, swing, trigger):
    primary_bullish = primary["bucket"] in ("FRESH_MOMENTUM", "OVERBOUGHT_WATCH")
    primary_bearish = primary["bucket"] == "WEAK_AVOID"
    swing_bullish = swing["trend"] == "UP"
    swing_bearish = swing["trend"] == "DOWN"
    trigger_bullish = trigger["above_sma10"] and trigger["momentum"] == "POSITIVE"
    trigger_bearish = (not trigger["above_sma10"]) and trigger["momentum"] == "NEGATIVE"

    if primary_bullish and swing_bullish and trigger_bullish:
        return "STRONG ALIGNED UPTREND — all three timeframes agree bullish.", "green"
    if primary_bullish and swing_bullish and not trigger_bullish:
        return "UPTREND INTACT, SHORT-TERM PAUSE — bigger picture still bullish, last month is cooling off.", "amber"
    if primary_bullish and swing_bearish:
        return "LONG-TERM BULLISH BUT LOSING MOMENTUM — medium-term wave has turned down inside a longer uptrend.", "amber"
    if primary_bearish and swing_bullish:
        return "POSSIBLE EARLY REVERSAL — long-term still weak, medium-term turned up. Unconfirmed.", "amber"
    if primary_bearish and swing_bearish and trigger_bearish:
        return "CONFIRMED DOWNTREND AT MULTIPLE TIMEFRAMES — avoid.", "red"
    if primary_bearish and swing_bearish:
        return "DOWNTREND — long-term and medium-term both weak.", "red"
    if primary["bucket"] == "NEUTRAL" and swing_bullish and trigger_bullish:
        return "EARLY BREAKOUT WATCH — long-term base is flat, but medium and short term are turning up. Not yet confirmed.", "green"
    if primary["bucket"] == "NEUTRAL" and swing_bearish and trigger_bearish:
        return "FADING WITHIN A SIDEWAYS BASE — no edge here.", "red"
    return "MIXED / NO CLEAR ALIGNMENT — timeframes disagree, no clean signal right now.", "muted"


# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_from_db(symbol):
    if not os.path.exists(NIFTY500_DB):
        return None
    conn = sqlite3.connect(NIFTY500_DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM price_history WHERE symbol = ? ORDER BY date",
        conn, params=(symbol,))
    conn.close()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live(symbol, period="5y"):
    yf_symbol = symbol.upper().strip()
    if not yf_symbol.endswith(".NS"):
        yf_symbol += ".NS"
    try:
        hist = yf.Ticker(yf_symbol).history(period=period, interval="1d")
    except Exception:
        return None
    if hist.empty:
        return None
    hist = hist.reset_index()
    hist = hist.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                 "Low": "low", "Close": "close", "Volume": "volume"})
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    return hist[["date", "open", "high", "low", "close", "volume"]]


def get_price_data(symbol):
    df = load_from_db(symbol)
    if df is not None and len(df) >= MIN_ROWS_REQUIRED:
        return df, "local database"
    df = fetch_live(symbol)
    if df is not None:
        return df, "live fetch"
    return None, None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_index_quote(ticker_symbol):
    """Fetch last close + % change for an index. Fails gracefully (returns None) if unreachable."""
    try:
        hist = yf.Ticker(ticker_symbol).history(period="5d", interval="1d")
        if hist.empty or len(hist) < 2:
            return None
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        pct = (last / prev - 1) * 100
        return {"last": round(last, 2), "pct": round(pct, 2)}
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_intraday(yf_symbol, interval):
    """
    Fetch intraday data. Yahoo Finance only retains a limited history per
    interval - these periods are chosen to stay within what Yahoo actually
    serves (requesting more just gets rejected or silently truncated):
      15m -> ~5 trading days
      30m -> ~60 days
      1h  -> ~90 days
    This is a real data-source limit, not a choice - intraday history simply
    isn't kept as far back as daily data.
    """
    period_map = {"15m": "5d", "30m": "60d", "1h": "3mo"}
    period = period_map.get(interval, "5d")
    try:
        hist = yf.Ticker(yf_symbol).history(period=period, interval=interval)
    except Exception:
        return None
    if hist.empty:
        return None
    hist = hist.reset_index()
    date_col = "Datetime" if "Datetime" in hist.columns else "Date"
    hist = hist.rename(columns={date_col: "date", "Open": "open", "High": "high",
                                 "Low": "low", "Close": "close", "Volume": "volume"})
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    return hist[["date", "open", "high", "low", "close", "volume"]]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index_history(ticker_symbol, period="1y"):
    try:
        hist = yf.Ticker(ticker_symbol).history(period=period, interval="1d")
    except Exception:
        return None
    if hist.empty:
        return None
    hist = hist.reset_index()
    hist = hist.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                 "Low": "low", "Close": "close", "Volume": "volume"})
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    return hist[["date", "open", "high", "low", "close", "volume"]]


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_symbols():
    if not os.path.exists(NIFTY500_DB):
        return []
    conn = sqlite3.connect(NIFTY500_DB)
    symbols = [r[0] for r in conn.execute("SELECT symbol FROM stocks").fetchall()]
    conn.close()
    return symbols


@st.cache_data(ttl=3600, show_spinner=False)
def compute_all_buckets():
    """Classify every symbol in the local DB. Expensive - cached for an hour."""
    symbols = get_all_symbols()
    rows = []
    for symbol in symbols:
        df = load_from_db(symbol)
        if df is None or len(df) < MIN_ROWS_REQUIRED:
            continue
        result = classify_primary(df)
        if result is None:
            continue
        rows.append({
            "Symbol": symbol, "Close": result["close"], "RSI": result["rsi"],
            "Trend": result["trend"], "Bucket": result["bucket"],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RENDERING HELPERS
# ---------------------------------------------------------------------------
BADGE_CLASS = {
    "FRESH_MOMENTUM": "badge-fresh", "OVERBOUGHT_WATCH": "badge-overbought",
    "WEAK_AVOID": "badge-weak", "NEUTRAL": "badge-neutral",
}
BADGE_TEXT = {
    "FRESH_MOMENTUM": "FRESH MOMENTUM", "OVERBOUGHT_WATCH": "OVERBOUGHT · WATCH",
    "WEAK_AVOID": "WEAK · AVOID", "NEUTRAL": "NEUTRAL",
}
VERDICT_COLOR = {"green": "var(--green)", "amber": "var(--amber)", "red": "var(--red)", "muted": "var(--muted)"}


def render_ticker_strip():
    nifty = fetch_index_quote("^NSEI")
    banknifty = fetch_index_quote("^NSEBANK")
    buckets_df = compute_all_buckets()

    items = []
    if nifty:
        cls = "up" if nifty["pct"] >= 0 else "down"
        items.append(f'<div class="ticker-item"><b>NIFTY 50</b>{nifty["last"]} <span class="{cls}">{nifty["pct"]:+.2f}%</span></div>')
    else:
        items.append('<div class="ticker-item"><b>NIFTY 50</b>unavailable</div>')

    if banknifty:
        cls = "up" if banknifty["pct"] >= 0 else "down"
        items.append(f'<div class="ticker-item"><b>BANK NIFTY</b>{banknifty["last"]} <span class="{cls}">{banknifty["pct"]:+.2f}%</span></div>')
    else:
        items.append('<div class="ticker-item"><b>BANK NIFTY</b>unavailable</div>')

    if not buckets_df.empty:
        counts = buckets_df["Bucket"].value_counts()
        items.append(f'<div class="ticker-item"><b>FRESH MOMENTUM</b><span class="up">{counts.get("FRESH_MOMENTUM",0)} stocks</span></div>')
        items.append(f'<div class="ticker-item"><b>OVERBOUGHT</b>{counts.get("OVERBOUGHT_WATCH",0)} stocks</div>')
        items.append(f'<div class="ticker-item"><b>WEAK/AVOID</b><span class="down">{counts.get("WEAK_AVOID",0)} stocks</span></div>')

    st.markdown(f'<div class="ticker-strip">{"".join(items)}</div>', unsafe_allow_html=True)


def render_chart(df, symbol, yf_symbol=None, key_prefix=""):
    """
    Price chart with a timeframe selector. Daily uses the full history
    already loaded (with SMA50/SMA200 overlay, matching the classification
    logic). Intraday timeframes (15m/30m/1h) are visual-only - recent price
    action, not run through the long-term indicators, since those need
    years of daily data to mean anything.
    """
    timeframe = st.radio(
        "Timeframe", ["Daily", "15m", "30m", "1h"],
        horizontal=True, key=f"{key_prefix}_tf_{symbol}", label_visibility="collapsed",
    )

    if timeframe == "Daily":
        plot_df = df.tail(300).copy()
        plot_df["sma50"] = df["close"].rolling(50).mean().tail(300)
        plot_df["sma200"] = df["close"].rolling(200).mean().tail(300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["close"], name="Close",
                                  line=dict(color="#E4E8EC", width=1.5)))
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["sma50"], name="SMA50",
                                  line=dict(color="#3FB68C", width=1.2, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["sma200"], name="SMA200",
                                  line=dict(color="#D9A441", width=1.2, dash="dot")))
    else:
        if yf_symbol is None:
            yf_symbol = symbol.upper().strip()
            if not yf_symbol.endswith(".NS") and not yf_symbol.startswith("^"):
                yf_symbol += ".NS"
        st.caption({"15m": "Last ~5 trading days (Yahoo Finance's intraday history limit at this resolution)",
                    "30m": "Last ~60 days (Yahoo Finance's intraday history limit at this resolution)",
                    "1h": "Last ~3 months (Yahoo Finance's intraday history limit at this resolution)"}[timeframe])
        intraday_df = fetch_intraday(yf_symbol, timeframe)
        if intraday_df is None or intraday_df.empty:
            st.warning(f"No {timeframe} data available for {symbol} right now (market may be closed, or data temporarily unavailable).")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=intraday_df["date"], y=intraday_df["close"], name="Close",
                                  line=dict(color="#E4E8EC", width=1.3)))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#12181F", plot_bgcolor="#12181F",
        margin=dict(l=10, r=10, t=30, b=10), height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#1F2830"),
        font=dict(family="JetBrains Mono", size=10, color="#7C8894"),
    )
    st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_chart_{symbol}_{timeframe}")


def render_card(symbol, primary, swing, trigger, verdict, verdict_color, source, df=None, show_chart_toggle=True):
    rsi = primary["rsi"]
    fib_html = ""
    if primary["bucket"] == "OVERBOUGHT_WATCH":
        fib = primary["fib_levels"]
        fib_html = (
            '<div class="fib-block">'
            '<div class="fib-title">RE-ENTRY ZONES (FIBONACCI)</div>'
            f'<div class="fib-row"><span>38.2%</span><span>₹{fib["fib_38.2"]}</span></div>'
            f'<div class="fib-row"><span>50.0%</span><span>₹{fib["fib_50.0"]}</span></div>'
            f'<div class="fib-row"><span>61.8%</span><span>₹{fib["fib_61.8"]}</span></div>'
            '</div>'
        )
    card_html = (
        '<div class="card">'
        '<div class="card-top">'
        f'<div><div class="symbol">{symbol.upper()}</div>'
        f'<div class="close-price">₹{primary["close"]}</div></div>'
        f'<div class="badge {BADGE_CLASS[primary["bucket"]]}">{BADGE_TEXT[primary["bucket"]]}</div>'
        '</div>'
        f'<div class="rsi-label-row"><span>OVERSOLD</span><span>RSI {rsi}</span><span>OVERBOUGHT</span></div>'
        f'<div class="rsi-track"><div class="rsi-marker" style="left:{min(max(rsi,0),100)}%"></div></div>'
        f'<div class="stat-row"><span>Primary trend</span><b>{primary["trend"]}</b></div>'
        f'<div class="stat-row"><span>Swing (3-6mo)</span><b>{swing["trend"]}</b></div>'
        f'<div class="stat-row"><span>Trigger (1mo, 5d ret)</span><b>{trigger["ret_5d_pct"]:+.2f}%</b></div>'
        f'{fib_html}'
        f'<div class="verdict-block" style="border-left-color:{VERDICT_COLOR[verdict_color]}">'
        '<div class="verdict-title">ALIGNMENT VERDICT</div>'
        f'{verdict}'
        '</div>'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)
    st.caption(f"Data source: {source}")
    if show_chart_toggle and df is not None:
        with st.expander(f"Show chart — {symbol.upper()}"):
            render_chart(df, symbol, key_prefix="card")


def analyze_and_render(symbol, column):
    df, source = get_price_data(symbol)
    if df is None:
        with column:
            st.error(f"{symbol}: could not fetch data. Check the symbol is correct and NSE-listed.")
        return
    if len(df) < MIN_ROWS_REQUIRED:
        with column:
            st.warning(f"{symbol}: only {len(df)} days of history — not enough for reliable analysis.")
        return
    primary = classify_primary(df)
    swing = classify_swing(df)
    trigger = classify_trigger(df)
    if not all([primary, swing, trigger]):
        with column:
            st.warning(f"{symbol}: insufficient data to classify.")
        return
    verdict, verdict_color = synthesize_alignment(primary, swing, trigger)
    with column:
        render_card(symbol, primary, swing, trigger, verdict, verdict_color, source, df=df)


# ---------------------------------------------------------------------------
# APP LAYOUT
# ---------------------------------------------------------------------------
st.markdown("# screener.terminal")
st.caption(f"Personal technical screener — data as of {datetime.now().strftime('%Y-%m-%d')} · not financial advice")

render_ticker_strip()

idx_col1, idx_col2, _ = st.columns([1, 1, 3])
with idx_col1:
    show_nifty = st.button("📈 Nifty 50 chart", key="btn_nifty")
with idx_col2:
    show_banknifty = st.button("📈 Bank Nifty chart", key="btn_banknifty")

if "active_index_chart" not in st.session_state:
    st.session_state.active_index_chart = None
if show_nifty:
    st.session_state.active_index_chart = "NIFTY 50" if st.session_state.active_index_chart != "NIFTY 50" else None
if show_banknifty:
    st.session_state.active_index_chart = "BANK NIFTY" if st.session_state.active_index_chart != "BANK NIFTY" else None

if st.session_state.active_index_chart == "NIFTY 50":
    st.markdown("### NIFTY 50")
    daily_hist = fetch_index_history("^NSEI", period="1y")
    if daily_hist is not None:
        render_chart(daily_hist, "NIFTY 50", yf_symbol="^NSEI", key_prefix="index")
    else:
        st.warning("Could not fetch Nifty 50 data right now.")
elif st.session_state.active_index_chart == "BANK NIFTY":
    st.markdown("### BANK NIFTY")
    daily_hist = fetch_index_history("^NSEBANK", period="1y")
    if daily_hist is not None:
        render_chart(daily_hist, "BANK NIFTY", yf_symbol="^NSEBANK", key_prefix="index")
    else:
        st.warning("Could not fetch Bank Nifty data right now.")

tab_analyze, tab_browse = st.tabs(["🔍 Analyze Symbols", "📋 Browse Screener"])

with tab_analyze:
    if "analyzed_symbols" not in st.session_state:
        st.session_state.analyzed_symbols = []

    symbols_input = st.text_input(
        "Paste symbols (comma-separated)",
        placeholder="TATAMOTORS, JKPAPER, THYROCARE, RADICO",
    )
    run = st.button("Run analysis", type="primary")

    if run:
        if symbols_input.strip():
            st.session_state.analyzed_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        else:
            st.warning("Enter at least one symbol first.")

    # Reading from session_state (not the transient button value) so results
    # survive reruns triggered by widgets inside the cards themselves - e.g.
    # the chart timeframe radio button. Without this, clicking anything
    # inside a card would wipe the whole results section, since st.button
    # only returns True on the exact click that triggered it.
    if st.session_state.analyzed_symbols:
        symbols = st.session_state.analyzed_symbols
        st.markdown(f"### Results — {len(symbols)} symbol(s)")
        cols = st.columns(2)
        for i, symbol in enumerate(symbols):
            with st.spinner(f"Analyzing {symbol}..."):
                analyze_and_render(symbol, cols[i % 2])

with tab_browse:
    st.caption("Filters your local database (screener.py logic) by bucket — browse instead of typing symbols.")
    if not os.path.exists(NIFTY500_DB):
        st.warning(f"No local database found ({NIFTY500_DB}). Run ingest_nifty500.py first, or use the Analyze tab for live lookups.")
    else:
        with st.spinner("Classifying full universe (cached for an hour after first run)..."):
            buckets_df = compute_all_buckets()

        if buckets_df.empty:
            st.warning("No classifiable stocks found in the local database.")
        else:
            bucket_choice = st.selectbox(
                "Filter by bucket",
                ["FRESH_MOMENTUM", "OVERBOUGHT_WATCH", "WEAK_AVOID", "NEUTRAL"],
                format_func=lambda b: BADGE_TEXT[b],
            )
            filtered = buckets_df[buckets_df["Bucket"] == bucket_choice].sort_values("RSI", ascending=False)
            st.write(f"{len(filtered)} stock(s) in {BADGE_TEXT[bucket_choice]}")
            st.dataframe(filtered[["Symbol", "Close", "RSI", "Trend"]], width='stretch', hide_index=True)

            selected_symbol = st.selectbox("View full analysis + chart for:", [""] + filtered["Symbol"].tolist())
            if selected_symbol:
                st.markdown("---")
                analyze_and_render(selected_symbol, st.container())
