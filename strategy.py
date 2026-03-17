"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Long-only momentum + ADX/DI + BB, SMA(200) trend filter.

    Change: Use SMA(200) instead of SMA(50) for more stable trend detection.
    This should improve train/val consistency since 200-day trend is more robust.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Long-term trend filter
    sma200 = close.rolling(200).mean()
    trend_up = close > sma200

    # Momentum
    roc = close.pct_change(20)

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_lower = bb_mid - 2 * bb_std

    # ADX(14) with DI
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(14).mean()

    strong_trend = adx > 20
    di_bullish = plus_di > minus_di

    signals = pd.Series(0, index=df.index)

    # Long-only momentum with ADX/DI confirmation
    signals[trend_up & (roc > 0) & strong_trend & di_bullish] = 1

    # BB oversold bounce in uptrend
    signals[trend_up & (close < bb_lower)] = 1

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
