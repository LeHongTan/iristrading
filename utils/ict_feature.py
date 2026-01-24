import numpy as np
import pandas as pd

def calc_swing_highs_lows(df, w=3):
    highs = (df['high'].rolling(window=w, center=True).max() == df['high'])
    lows = (df['low'].rolling(window=w, center=True).min() == df['low'])
    return highs.astype(int), lows.astype(int)

def calc_fvg(df):
    FVG_up = np.zeros(len(df))
    FVG_down = np.zeros(len(df))
    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            FVG_up[i] = 1
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            FVG_down[i] = 1
    return FVG_up, FVG_down

def calc_order_block(df, window=12):
    OB_bull = np.zeros(len(df))
    OB_bear = np.zeros(len(df))
    # Simplified: block = local min/max close in window
    closes = df['close']
    rolling_max = closes.rolling(window=window).max()
    rolling_min = closes.rolling(window=window).min()
    OB_bull[closes == rolling_min] = 1
    OB_bear[closes == rolling_max] = 1
    return OB_bull, OB_bear

def calc_smt(df1, df2):
    # Identify swing points divergence (Simplified)
    swing_high1, swing_low1 = calc_swing_highs_lows(df1)
    swing_high2, swing_low2 = calc_swing_highs_lows(df2)
    SMT = np.zeros(len(df1))
    for i in range(len(df1)):
        if swing_high1[i] and not swing_high2[i]:
            SMT[i] = 1  # Classic SMT, leader makes new high, laggard fails
        elif swing_low1[i] and not swing_low2[i]:
            SMT[i] = -1
    return SMT

def calc_ict_features(df, others={}):
    """
    df: pandas dataframe, OHLCV + index timestamp
    others: dict symbol->df for SMT reference
    Returns: Dict of feature column: numpy array
    """
    highs, lows = calc_swing_highs_lows(df)
    FVG_up, FVG_down = calc_fvg(df)
    OB_bull, OB_bear = calc_order_block(df)
    SMT = np.zeros(len(df))
    if others:
        # Chỉ tính SMT so với symbol đầu tiên trong others
        for sym, other in others.items():
            SMT += calc_smt(df, other)
        SMT = np.clip(SMT, -1, 1)

    return {
        'swing_high': highs,
        'swing_low': lows,
        'fvg_up': FVG_up,
        'fvg_down': FVG_down,
        'ob_bull': OB_bull,
        'ob_bear': OB_bear,
        'smt': SMT,
    }