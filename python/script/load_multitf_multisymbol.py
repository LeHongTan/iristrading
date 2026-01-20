import pandas as pd
from functools import reduce

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TFS = ["1m", "5m", "15m", "1h", "4h"]
dfs = {}
min_date, max_date = None, None

# Load all data
for coin in COINS:
    sub = {}
    for tf in TFS:
        f = f"data/{coin}_{tf}.csv"
        try:
            df = pd.read_csv(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            sub[tf] = df
            # Track min start, max stop
            t0 = df["timestamp"].iloc[0]
            t1 = df["timestamp"].iloc[-1]
            min_date = min(min_date, t0) if min_date else t0
            max_date = max(max_date, t1) if max_date else t1
        except Exception as e:
            print(f"Skip {coin} {tf}: {e}")
    dfs[coin] = sub

# Sync timeline: lấy các timestamp chung trên tất cả các coin đã list  
timeline = []
for ts in pd.date_range(min_date, max_date, freq="1min"):
    n_coin = sum([ts in sub["1m"]["timestamp"].values for sub in dfs.values() if "1m" in sub])
    if n_coin >= 2:  # at least 2 coin list để SMT
        timeline.append(ts)

# Output: MultiSymbolData dạng dict {coin: {tf: DataFrame}}
print("Timeline aligned", len(timeline), "min, coins:", [c for c in dfs if "1m" in dfs[c]])