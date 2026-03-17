"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Momentum + Bollinger Bands + ATR volatility filter.

    Core: ROC(20) momentum with SMA50 trend filter.
    BB: Mean reversion at Bollinger Band extremes.
    ATR: Go flat during extreme volatility to reduce drawdown.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend filter
    sma50 = close.rolling(50).mean()
    trend_up = close > sma50
    trend_down = close < sma50

    # Momentum
    roc = close.pct_change(20)

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ATR(14) volatility filter
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = atr / close  # ATR as % of price
    atr_avg = atr_pct.rolling(50).mean()
    high_vol = atr_pct > (atr_avg * 1.8)  # Extreme volatility

    signals = pd.Series(0, index=df.index)

    # Base momentum signals
    signals[trend_up & (roc > 0)] = 1
    signals[trend_down & (roc < 0)] = -1

    # Bollinger Band mean reversion overrides
    signals[trend_up & (close < bb_lower)] = 1
    signals[trend_down & (close > bb_upper)] = -1

    # Go flat during extreme volatility
    signals[high_vol] = 0

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
