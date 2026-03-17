"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Multi-timeframe momentum with SMA trend filter.

    Uses 20-day rate of change for momentum signals,
    confirmed by 50-day SMA trend direction.
    Generates more trades while maintaining consistency.
    """
    close = df["close"]

    # Trend filter: 50-day SMA direction
    sma50 = close.rolling(50).mean()
    trend_up = close > sma50
    trend_down = close < sma50

    # Momentum: 20-day rate of change
    roc = close.pct_change(20)

    signals = pd.Series(0, index=df.index)

    # Long when price above SMA50 and positive momentum
    signals[trend_up & (roc > 0)] = 1
    # Short when price below SMA50 and negative momentum
    signals[trend_down & (roc < 0)] = -1

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
