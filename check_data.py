import sqlite3

conn = sqlite3.connect('nifty500.db')

# Check for duplicates
dupes = conn.execute(
    'SELECT symbol, date, COUNT(*) c FROM price_history GROUP BY symbol, date HAVING c > 1'
).fetchall()
print('Duplicates:', len(dupes))

# Check for null/zero prices
bad = conn.execute(
    'SELECT COUNT(*) FROM price_history WHERE close <= 0 OR close IS NULL'
).fetchone()
print('Invalid prices:', bad[0])

# Spot check RELIANCE latest close
r = conn.execute(
    "SELECT date, close FROM price_history WHERE symbol='RELIANCE' ORDER BY date DESC LIMIT 3"
).fetchall()
print('RELIANCE last 3 days:', r)

# Row count distribution - flag any symbol with unusually few rows
low = conn.execute(
    'SELECT symbol, COUNT(*) c FROM price_history GROUP BY symbol HAVING c < 1000 ORDER BY c ASC LIMIT 10'
).fetchall()
print('Symbols with <1000 rows (check these):', low)

conn.close()
