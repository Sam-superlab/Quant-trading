# quant_trading_system/utils/config.py

import os
import re

class Config:
    """
    A central configuration class for the quantitative trading system.
    This makes it easy to manage parameters without changing the core logic.
    """
    # --- Data Fetching ---
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', "d21bk3pr01qkdupiodggd21bk3pr01qkdupiodh0")

    # --- Alpaca API Keys (prioritizes environment variables) ---
    APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID', 'YOUR_KEY_ID_HERE')
    APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY', 'YOUR_SECRET_KEY_HERE')
    # Set to False for live trading
    PAPER_TRADING = True

    # --- Strategy & Model Parameters ---
    DEFAULT_TICKER = "NVDA"
    # The confidence level required from the model to place a trade
    CONFIDENCE_THRESHOLD = 0.55 
    # Historical data period to use for training the model
    TRAINING_HISTORY_YEARS = 3

    # --- Risk Management ---
    # The percentage of total equity to risk on a single trade
    RISK_PERCENTAGE = 0.015 # 1.5%

    # --- Backtesting Parameters ---
    BACKTEST_CASH = 100_000
    BACKTEST_COMMISSION = 0.002 # 0.2%

def sanitize_feature_names(columns):
    """
    Sanitize feature names to be compatible with LightGBM (alphanumeric and underscores only).
    Replaces non-alphanumeric characters with a single underscore, strips leading/trailing underscores, and prepends 'f_' if the name is empty or starts with a digit.
    """
    sanitized = []
    for col in columns:
        # Replace non-alphanumeric with underscore, collapse multiple underscores, strip
        new_col = re.sub(r'[^0-9a-zA-Z]+', '_', str(col))
        new_col = new_col.strip('_')
        if not new_col or new_col[0].isdigit():
            new_col = 'f_' + new_col
        sanitized.append(new_col)
    return sanitized

# Instantiate the config so it can be imported directly
config = Config()
