import pandas as pd
import sqlite3
import os

def import_csv(symbol, tf):
    fpath = f"data/{symbol}_{tf}.csv"
    df = pd.read_csv(fpath)
    conn = sqlite3.connect("iris_trading.db")
    df['symbol'] = symbol
    cursor = conn.cursor()
    for idx, row in df.iterrows():
        cursor.execute('''
            INSERT OR IGNORE INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            tf, 
            int(row['timestamp']),
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume']),
        ))
    conn.commit()
    conn.close()

# List tất cả symbol-timeframe bạn muốn import
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
TFS = ['1m', '5m', '15m', '1h', '4h']

for symbol in COINS:
    for tf in TFS:
        if os.path.exists(f"data/{symbol}_{tf}.csv"):
            print(f"Import {symbol}_{tf} ...")
            import_csv(symbol, tf)