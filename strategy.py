"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """SMA crossover with RSI filter for better entries."""
    close = df["close"]

    # Faster SMAs for more trades
    fast_ma = close.rolling(10).mean()
    slow_ma = close.rolling(30).mean()

    # RSI(14) for overbought/oversold filtering
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    signals = pd.Series(0, index=df.index)

    # Long when fast > slow AND RSI not overbought (< 70)
    signals[(fast_ma > slow_ma) & (rsi < 70)] = 1
    # Short when fast < slow AND RSI not oversold (> 30)
    signals[(fast_ma < slow_ma) & (rsi > 30)] = -1

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
