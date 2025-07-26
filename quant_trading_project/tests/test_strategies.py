import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from quant_trading_system.strategies.basic import run_sma_crossover, run_rsi_strategy


def sample_data():
    dates = pd.date_range("2024-01-01", periods=100)
    price = pd.Series(range(100), index=dates)
    df = pd.DataFrame({
        'Open': price,
        'High': price + 1,
        'Low': price - 1,
        'Close': price,
        'Volume': 1000
    })
    return df


def test_run_sma_crossover():
    df = sample_data()
    bt, stats = run_sma_crossover(df, 5, 10)
    assert 'Return [%]' in stats


def test_run_rsi_strategy():
    df = sample_data()
    bt, stats = run_rsi_strategy(df, 14, 70, 30)
    assert 'Return [%]' in stats
