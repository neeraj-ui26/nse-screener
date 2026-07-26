"""
Nifty 500 historical data ingestion script.
Pulls 5 years of daily OHLCV data via yfinance and stores it in SQLite.
"""

import sqlite3
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

DB_PATH = "nifty500.db"
SYMBOL_LIST_CSV = "nifty500.csv"
YEARS_OF_HISTORY = "5y"
LOG_FILE = "ingestion_log.txt"


def setup_database(conn):
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            industry TEXT,
            isin TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_history(symbol, date)")
    conn.commit()


def load_symbol_list():
    """Read the Nifty 500 constituent list."""
    df = pd.read_csv(SYMBOL_LIST_CSV)
    df.columns = [c.strip() for c in df.columns]
    return df


def populate_stocks_table(conn, symbols_df):
    """Insert stock metadata (symbol, name, industry, isin)."""
    rows = [
        (row["Symbol"], row["Company Name"], row["Industry"], row["ISIN Code"])
        for _, row in symbols_df.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO stocks (symbol, name, industry, isin) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def fetch_and_store(conn, symbol, log):
    """Fetch daily history for one symbol via yfinance and upsert into price_history."""
    yf_symbol = f"{symbol}.NS"
    try:
        hist = yf.Ticker(yf_symbol).history(period=YEARS_OF_HISTORY, interval="1d")
        if hist.empty:
            log.write(f"{symbol}: NO DATA RETURNED\n")
            return 0

        hist = hist.reset_index()
        rows = [
            (
                symbol,
                row["Date"].strftime("%Y-%m-%d"),
                round(float(row["Open"]), 2),
                round(float(row["High"]), 2),
                round(float(row["Low"]), 2),
                round(float(row["Close"]), 2),
                int(row["Volume"]),
            )
            for _, row in hist.iterrows()
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO price_history
               (symbol, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        return len(rows)
    except Exception as e:
        log.write(f"{symbol}: ERROR - {e}\n")
        return 0


def main():
    conn = sqlite3.connect(DB_PATH)
    setup_database(conn)

    symbols_df = load_symbol_list()
    populate_stocks_table(conn, symbols_df)

    symbols = symbols_df["Symbol"].tolist()
    print(f"Starting ingestion for {len(symbols)} symbols...")

    success_count = 0
    fail_count = 0
    total_rows = 0

    with open(LOG_FILE, "a") as log:
        log.write(f"\n=== Run started {datetime.now()} ===\n")
        for i, symbol in enumerate(symbols, 1):
            rows_added = fetch_and_store(conn, symbol, log)
            if rows_added > 0:
                success_count += 1
                total_rows += rows_added
            else:
                fail_count += 1

            if i % 25 == 0:
                print(f"  Progress: {i}/{len(symbols)} | success={success_count} fail={fail_count}")

            # be polite to Yahoo's servers
            time.sleep(0.3)

    conn.close()
    print(f"\nDone. Success: {success_count}, Failed: {fail_count}, Total rows: {total_rows}")
    print(f"Failures logged in {LOG_FILE}")


if __name__ == "__main__":
    main()
