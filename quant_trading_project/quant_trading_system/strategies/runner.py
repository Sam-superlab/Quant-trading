from datetime import datetime
from typing import Dict, Any

import pandas as pd

from quant_trading_system.data.data_fetcher import DataFetcher
from quant_trading_system.models.enhanced_backtester import EnhancedBacktester
from .basic import run_sma_crossover, run_rsi_strategy


def run_strategy(strategy: str, ticker: str, start: str, end: str, risk_pct: float, params: Dict) -> Dict[str, Any]:
    """Fetch data and run the selected strategy returning useful artifacts."""

    fetcher = DataFetcher()

    if strategy == "ML Model":
        backtester = EnhancedBacktester(ticker)
        stats, error = backtester.run_backtest()
        if error:
            raise ValueError(error)

        pnl = None
        if backtester.equity_curve is not None:
            pnl = backtester.equity_curve['Equity_Curve']

        return {
            'pnl_curve': pnl,
            'stats': stats,
            'feature_importance': backtester.feature_importance,
            'trades': None,
            'data': None
        }

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

    return {
        'pnl_curve': stats._equity_curve['Equity'],
        'stats': stats,
        'feature_importance': None,
        'trades': stats._trades,
        'data': bt._data
    }
