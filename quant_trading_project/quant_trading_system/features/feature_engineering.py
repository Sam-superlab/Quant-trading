# quant_trading_system/features/feature_engineering.py

import pandas as pd
import numpy as np

# We import the previous modules to demonstrate the full pipeline
from quant_trading_system.data.data_fetcher import DataFetcher
from quant_trading_system.data.data_preprocessor import DataPreprocessor

class FeatureEngineering:
    """
    A module for creating predictive features from financial time-series data.

    This class fulfills Deliverable 3.2 of the project plan. It takes
    preprocessed data and generates a variety of technical indicators and
    time-series features that will be used in the modeling phase.
    """

    def __init__(self, data):
        """
        Initializes the FeatureEngineering class.

        Args:
            data (pd.DataFrame): The preprocessed input data, which must include
                                 'Open', 'High', 'Low', 'Close', and 'Volume' columns.
        """
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Input DataFrame must contain 'Open', 'High', 'Low', 'Close', 'Volume' columns.")
        self.data = data.copy()

    def add_moving_averages(self, windows=[10, 20, 50]):
        """
        Adds Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
        for a list of window sizes.

        Args:
            windows (list, optional): A list of integer window sizes. Defaults to [10, 20, 50].
        """
        print("Adding moving averages...")
        for window in windows:
            self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
            self.data[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
        return self

    def add_momentum_indicators(self, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9):
        """
        Adds momentum indicators: Relative Strength Index (RSI) and Moving
        Average Convergence Divergence (MACD).

        Args:
            rsi_window (int, optional): Window for RSI. Defaults to 14.
            macd_fast (int, optional): Fast EMA period for MACD. Defaults to 12.
            macd_slow (int, optional): Slow EMA period for MACD. Defaults to 26.
            macd_signal (int, optional): Signal line EMA period for MACD. Defaults to 9.
        """
        print("Adding momentum indicators (RSI, MACD)...")
        # RSI calculation
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # MACD calculation
        ema_fast = self.data['Close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=macd_slow, adjust=False).mean()
        self.data['MACD'] = ema_fast - ema_slow
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=macd_signal, adjust=False).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        return self

    def add_volatility_indicators(self, bb_window=20, bb_std=2, atr_window=14):
        """
        Adds volatility indicators: Bollinger Bands (BB) and Average True Range (ATR).

        Args:
            bb_window (int, optional): Window for Bollinger Bands. Defaults to 20.
            bb_std (int, optional): Number of standard deviations for BB. Defaults to 2.
            atr_window (int, optional): Window for ATR. Defaults to 14.
        """
        print("Adding volatility indicators (Bollinger Bands, ATR)...")
        # Bollinger Bands
        sma = self.data['Close'].rolling(window=bb_window).mean()
        std = self.data['Close'].rolling(window=bb_window).std()

        if isinstance(self.data.columns, pd.MultiIndex):
            # Debug: print MultiIndex structure
            print("Column MultiIndex names:", self.data.columns.names)
            print("All columns:", self.data.columns)
            level_names = self.data.columns.names
            if 'Ticker' in level_names:
                ticker_level = level_names.index('Ticker')
            else:
                ticker_level = 1
            tickers = self.data.columns.get_level_values(ticker_level).unique()
            print("Detected tickers:", tickers)
            for ticker in tickers:
                if not ticker or ticker not in sma or ticker not in std:
                    print(f"Skipping invalid or empty ticker: '{ticker}'")
                    continue  # Skip empty or invalid tickers
                self.data[('BB_Upper', ticker)] = sma[ticker] + (std[ticker] * bb_std)
                self.data[('BB_Lower', ticker)] = sma[ticker] - (std[ticker] * bb_std)
                self.data[('BB_Width', ticker)] = (
                    self.data[('BB_Upper', ticker)] - self.data[('BB_Lower', ticker)]
                ) / sma[ticker]
        else:
            # Single ticker
            self.data['BB_Upper'] = sma + (std * bb_std)
            self.data['BB_Lower'] = sma - (std * bb_std)
            self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / sma

        # Average True Range (ATR)
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['ATR'] = tr.ewm(alpha=1/atr_window, adjust=False).mean()
        return self
        
    def add_lagged_returns(self, lags=[1, 2, 3, 5, 10]):
        """
        Adds lagged daily returns as features.

        Args:
            lags (list, optional): A list of integer lags. Defaults to [1, 2, 3, 5, 10].
        """
        print("Adding lagged returns...")
        self.data['Returns'] = self.data['Close'].pct_change()
        for lag in lags:
            self.data[f'Lag_Return_{lag}'] = self.data['Returns'].shift(lag)
        return self

    def get_feature_data(self):
        """
        Returns the final DataFrame with all added features.

        Returns:
            pd.DataFrame: The data with all generated features.
        """
        # Drop initial rows with NaNs created by rolling windows
        return self.data.dropna()


# --- Example Usage ---
if __name__ == '__main__':
    # --- Setup: Use previous modules to get clean data ---
    user_api_key = 'd21bk3pr01qkdupiodggd21bk3pr01qkdupiodh0'
    fetcher = DataFetcher(finnhub_api_key=user_api_key)
    preprocessor = DataPreprocessor()
    
    ticker = 'TSLA'
    start = '2022-01-01'
    end = '2023-12-31'

    # --- 1. Fetch and Preprocess Data ---
    print("\n" + "="*50)
    print(f"1. Fetching and preprocessing data for {ticker}")
    print("="*50)
    market_data = fetcher.get_market_data(ticker, start_date=start, end_date=end)
    
    if market_data is not None:
        # For a single ticker, yfinance returns a simple column index.
        clean_data = preprocessor.handle_missing_values(market_data, method='ffill')

        # --- 2. Apply Feature Engineering ---
        print("\n" + "="*50)
        print("2. Generating features...")
        print("="*50)
        
        # Chain all feature engineering methods together
        feature_generator = FeatureEngineering(clean_data)
        feature_generator.add_moving_averages()
        feature_generator.add_momentum_indicators()
        feature_generator.add_volatility_indicators()
        feature_generator.add_lagged_returns()
        
        # Get the final DataFrame
        final_data_with_features = feature_generator.get_feature_data()

        # --- 3. Display Results ---
        print("\n" + "="*50)
        print("3. Sample of final data with all features")
        print("="*50)
        
        print(f"Original data shape: {clean_data.shape}")
        print(f"Data shape after feature engineering and NaN removal: {final_data_with_features.shape}")
        
        # Display a sample of the final data
        print("\nFinal DataFrame Head:")
        print(final_data_with_features.head())
        
        print("\nFinal DataFrame Tail:")
        print(final_data_with_features.tail())
        
        print("\nColumns created:")
        print(final_data_with_features.columns.tolist())
