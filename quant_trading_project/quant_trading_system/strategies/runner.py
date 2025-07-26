from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

from quant_trading_system.data.data_fetcher import DataFetcher
from .basic import run_sma_crossover, run_rsi_strategy


def run_strategy(strategy: str, ticker: str, start: str, end: str, params: Dict) -> Tuple[object, dict]:
    """Fetch data and run the selected strategy.

    Returns a tuple of (Backtest object, stats dictionary).
    """
    fetcher = DataFetcher()
    data = fetcher.get_market_data(ticker, start, end)
    if data is None or data.empty:
        raise ValueError("No market data fetched")

    data = data.dropna()

    if strategy == "SMA Crossover":
        short_w = int(params.get("short_window", 10))
        long_w = int(params.get("long_window", 20))
        bt, stats = run_sma_crossover(data, short_w, long_w)
    elif strategy == "RSI":
        period = int(params.get("rsi_period", 14))
        overbought = int(params.get("overbought", 70))
        oversold = int(params.get("oversold", 30))
        bt, stats = run_rsi_strategy(data, period, overbought, oversold)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return bt, stats
