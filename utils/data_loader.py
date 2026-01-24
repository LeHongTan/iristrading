import os
import pandas as pd
from utils.ict_feature import calc_ict_features

def load_multisymbol_multitf(data_dir, symbols, timeframes, sequence_length=256):
    """
    Returns:
        symbol_tf_to_df: dict[symbol][tf] = pd.DataFrame (time ascending, index timestamp)
        timeline: sorted intersection of all timestamps (anchor)
    """
    symbol_tf_to_df = {}
    all_timestamps = None

    for symbol in symbols:
        symbol_tf_to_df[symbol] = {}
        for tf in timeframes:
            fn = os.path.join(data_dir, f"{symbol}_{tf}.csv")
            df = pd.read_csv(fn)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.sort_values('timestamp')
            df = df.set_index('timestamp')
            symbol_tf_to_df[symbol][tf] = df

            if all_timestamps is None:
                all_timestamps = set(df.index)
            else:
                all_timestamps = all_timestamps & set(df.index)  # only keep synchronized timestamps

    anchor_timeline = sorted(all_timestamps)
    return symbol_tf_to_df, anchor_timeline

def get_state_window(symbol_tf_to_df, anchor_timeline, idx, symbols, timeframes, sequence_length, use_ict=True):
    """
    idx: index in anchor_timeline
    Returns flatten state: [symbol1_tf1_features... symbolN_tfm_features...] + ICT features
    """
    state = []
    ict_features_all = []
    for symbol in symbols:
        for tf in timeframes:
            df = symbol_tf_to_df[symbol][tf]
            ts_window = anchor_timeline[max(0, idx-sequence_length+1): idx+1]
            subdf = df.loc[df.index.isin(ts_window)]
            # If window too short (đầu dãy), pad zeros
            if len(subdf) < sequence_length:
                pad = pd.DataFrame(0, index=range(sequence_length - len(subdf)), columns=subdf.columns)
                subdf = pd.concat([pad, subdf], ignore_index=True)
            else:
                subdf = subdf.tail(sequence_length)
            subdf = subdf.fillna(0)
            # OHLCV features
            raw_feats = subdf[['open','high','low','close','volume']].values.flatten()
            state.extend(raw_feats)
            # ICT feats (extend or flatten)
            if use_ict:
                ict_feats = calc_ict_features(subdf)
                flatten_ict = []
                for v in ict_feats.values():
                    flatten_ict.extend(v[-sequence_length:])  # window
                ict_features_all.extend(flatten_ict)
    return state + ict_features_all