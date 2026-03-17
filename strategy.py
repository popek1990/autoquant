"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Momentum + BB + ADX with tuned parameters.

    Tuning: ADX threshold 25 (stricter), ROC(15) for faster signals,
    SMA(40) for slightly faster trend detection.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend filter (slightly faster)
    sma40 = close.rolling(40).mean()
    trend_up = close > sma40
    trend_down = close < sma40

    # Momentum (shorter lookback)
    roc = close.pct_change(15)

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ADX(14)
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

    strong_trend = adx > 25  # Stricter threshold

    signals = pd.Series(0, index=df.index)

    # Momentum signals only in strong trends
    signals[trend_up & (roc > 0) & strong_trend] = 1
    signals[trend_down & (roc < 0) & strong_trend] = -1

    # BB mean reversion
    signals[trend_up & (close < bb_lower)] = 1
    signals[trend_down & (close > bb_upper)] = -1

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
