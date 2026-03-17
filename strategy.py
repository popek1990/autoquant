"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Momentum + Bollinger Band mean reversion hybrid.

    Primary signal: ROC(20) with SMA50 trend filter.
    Override: Bollinger Band extremes trigger mean reversion.
    """
    close = df["close"]

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

    signals = pd.Series(0, index=df.index)

    # Base momentum signals
    signals[trend_up & (roc > 0)] = 1
    signals[trend_down & (roc < 0)] = -1

    # Bollinger Band mean reversion overrides
    # If price drops below lower band in uptrend, go long (oversold bounce)
    signals[trend_up & (close < bb_lower)] = 1
    # If price rises above upper band in downtrend, go short (overbought reversal)
    signals[trend_down & (close > bb_upper)] = -1

    # Exit signals: fade extreme positions
    # Go flat when price returns to middle band against trend
    signals[(~trend_up) & (close > bb_upper)] = -1

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
