# quant_trading_system/features/feature_engineering.py

import pandas as pd
import numpy as np

class FeatureEngineering:
    """
    UPGRADED module for creating predictive features.
    Now includes the ability to generate features from fundamental data and volume profile.
    """

    def __init__(self, data):
        """
        Initializes the FeatureEngineering class.
        """
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Input DataFrame must contain 'Open', 'High', 'Low', 'Close', 'Volume' columns.")
        self.data = data.copy()

    def add_technical_indicators(self):
        """
        A wrapper method to add all technical indicators at once.
        """
        print("Adding technical indicators...")
        self._add_moving_averages()
        self._add_momentum_indicators()
        self._add_volatility_indicators()
        self._add_volume_profile_indicators() # <-- NEW
        self._add_lagged_returns()
        return self

    def add_fundamental_features(self):
        """
        Adds features based on fundamental data.
        This method assumes the fundamental data has already been aligned.
        """
        print("Adding fundamental features (YoY Growth)...")
        # We expect columns like 'QuarterlyRevenue' and 'QuarterlyNetIncome'
        # from the preprocessing step.
        
        # Calculate Year-over-Year (YoY) growth for quarterly data
        # A quarter has roughly 63 trading days. 4 quarters = 252 days.
        for col in ['QuarterlyRevenue', 'QuarterlyNetIncome']:
            if col in self.data.columns:
                # Calculate YoY percentage change
                self.data[f'{col}_YoY_Growth'] = self.data[col].pct_change(periods=252)
        
        return self

    def get_feature_data(self):
        """
        Returns the final DataFrame with all added features.
        """
        # Replace infinite values that can result from pct_change with NaNs
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop initial rows with NaNs created by rolling windows and growth calculations
        return self.data.dropna()

    # --- Private helper methods for technical indicators ---
    def _add_moving_averages(self, windows=[10, 20, 50]):
        for window in windows:
            self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
            self.data[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()

    def _add_momentum_indicators(self, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9):
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        ema_fast = self.data['Close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=macd_slow, adjust=False).mean()
        self.data['MACD'] = ema_fast - ema_slow
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=macd_signal, adjust=False).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']

    def _add_volatility_indicators(self, bb_window=20, bb_std=2, atr_window=14):
        sma = self.data['Close'].rolling(window=bb_window).mean()
        std = self.data['Close'].rolling(window=bb_window).std()
        self.data['BB_Upper'] = sma + (std * bb_std)
        self.data['BB_Lower'] = sma - (std * bb_std)
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['ATR'] = tr.ewm(alpha=1/atr_window, adjust=False).mean()
        
    def _add_volume_profile_indicators(self, vwap_window=14):
        """
        Adds Volume Weighted Average Price (VWAP) as a feature.
        """
        print("Adding volume profile indicators (VWAP)...")
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        tpv = typical_price * self.data['Volume']
        self.data[f'VWAP_{vwap_window}'] = tpv.rolling(window=vwap_window).sum() / self.data['Volume'].rolling(window=vwap_window).sum()

    def _add_lagged_returns(self, lags=[1, 2, 3, 5, 10]):
        self.data['Returns'] = self.data['Close'].pct_change()
        for lag in lags:
            self.data[f'Lag_Return_{lag}'] = self.data['Returns'].shift(lag)
