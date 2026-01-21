import ccxt
import pandas as pd
import time
import os

exchange = ccxt.bybit({"enableRateLimit": True})
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
tfs = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h'}
since = exchange.parse8601('2021-01-01T00:00:00Z')
limit = 1000
os.makedirs('data', exist_ok=True)

for symbol in symbols:
    for tf_name, tf_ccxt in tfs.items():
        print(f'Downloading {symbol} {tf_ccxt} ...')
        all_ohlcv = []
        since_this = since
        last_timestamp = None
        i = 0
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_ccxt, since=since_this, limit=limit)
            if not ohlcv:
                print(f"Break: fetch empty at iteration {i}")
                break
            # Remove possible duplicate at front
            if last_timestamp:
                ohlcv = [row for row in ohlcv if row[0] > last_timestamp]
            if not ohlcv:
                print(f"Break: filtered duplicated at iteration {i}, nothing new")
                break
            all_ohlcv += ohlcv
            print(f"Fetched {len(ohlcv)} rows, from {pd.to_datetime(ohlcv[0][0], unit='ms')} to {pd.to_datetime(ohlcv[-1][0], unit='ms')}, total {len(all_ohlcv)} rows")
            last_timestamp = ohlcv[-1][0]
            since_this = last_timestamp + 1
            time.sleep(exchange.rateLimit / 1000)
            i += 1
            # CHỈ break nếu KHÔNG NHẬN ĐƯỢC thêm nến! KHÔNG break nếu len(ohlcv) < limit!
        df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
        out_file = f"data/{symbol.replace('/','')}_{tf_name}.csv"
        df.to_csv(out_file, index=False)
        print(f"Saved {out_file}. Total {len(df)} rows.")