import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from quant_trading_system.data.data_preprocessor import DataPreprocessor


def test_handle_missing_values_ffill():
    df = pd.DataFrame({'A': [1, None, 3]})
    dp = DataPreprocessor()
    result = dp.handle_missing_values(df, method='ffill')
    assert result.iloc[1, 0] == 1


def test_handle_missing_values_mean():
    df = pd.DataFrame({'A': [1, None, 3]})
    dp = DataPreprocessor()
    result = dp.handle_missing_values(df, method='mean')
    assert result.iloc[1, 0] == 2
