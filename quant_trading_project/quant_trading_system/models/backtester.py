# quant_trading_system/models/backtester.py

import pandas as pd
import lightgbm as lgb
from backtesting import Backtest, Strategy
from datetime import datetime, timedelta
import numpy as np
import re

# Import our existing modules
from quant_trading_system.data.data_fetcher import DataFetcher
from quant_trading_system.data.data_preprocessor import DataPreprocessor
from quant_trading_system.features.feature_engineering import FeatureEngineering
from quant_trading_system.utils.config import config, sanitize_feature_names

class MLStrategy(Strategy):
    """
    A trading strategy that uses a pre-trained machine learning model
    to make trading decisions.
    """
    def init(self):
        self.predictions = self.I(lambda x: x, self.data.df['Predictions'])

    def next(self):
        if self.predictions[-1] == 1 and not self.position:
            self.buy()
        elif self.predictions[-1] == 0 and self.position:
            self.position.close()

def run_backtest(ticker, cash, commission):
    """
    UPGRADED to a Walk-Forward Backtester with fixes for LightGBM.
    """
    fetcher = DataFetcher(finnhub_api_key=config.FINNHUB_API_KEY)
    preprocessor = DataPreprocessor()
    
    start_date = (datetime.now() - timedelta(days=config.TRAINING_HISTORY_YEARS * 365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    # --- 1. Data Pipeline ---
    print(f"Backtester: Running data pipeline for {ticker}...")
    market_data = fetcher.get_market_data(ticker, start_date=start_date, end_date=end_date)
    if market_data is None:
        return None, "Could not fetch market data."

    clean_data = preprocessor.handle_missing_values(market_data, method='ffill')
    
    feature_generator = FeatureEngineering(clean_data)
    feature_generator.add_technical_indicators()
    data_with_features = feature_generator.get_feature_data()

    # Flatten MultiIndex columns if present
    if isinstance(data_with_features.columns, pd.MultiIndex):
        data_with_features.columns = [
            col if isinstance(col, str) else '_'.join([str(c) for c in col if c])
            for col in data_with_features.columns.values
        ]

    # --- 2. Walk-Forward Optimization ---
    print("Backtester: Starting Walk-Forward Analysis...")
    data_with_features.loc[:, 'Future_Return'] = data_with_features['Returns'].shift(-1)
    data_with_features.loc[:, 'Target'] = (data_with_features['Future_Return'] > 0).astype(int)
    data_with_features.dropna(inplace=True)

    y = data_with_features['Target']
    # Save a copy for backtesting before dropping columns
    data_for_backtest = data_with_features.copy()

    # Only drop columns for ML model input
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Future_Return', 'Target']
    drop_cols = [col for col in data_with_features.columns for base in cols_to_drop if col.startswith(base)]
    X = data_with_features.drop(columns=drop_cols)
    
    # FIX: Sanitize column names for LightGBM
    X.columns = sanitize_feature_names(X.columns)

    print(f"Sanitized X columns: {list(X.columns)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("DEBUG: About to start walk-forward splits")

    n_splits = 5
    total_size = len(X)
    split_size = total_size // n_splits
    all_predictions = pd.Series(np.nan, index=X.index)

    for i in range(1, n_splits):
        print(f"DEBUG: Fold {i} starting")
        train_end_index = i * split_size
        test_start_index = train_end_index
        test_end_index = (i + 1) * split_size if i < n_splits - 1 else total_size
        
        X_train, y_train = X.iloc[:train_end_index], y.iloc[:train_end_index]
        X_test = X.iloc[test_start_index:test_end_index]

        # Sanitize feature names for LightGBM compatibility
        X_train.columns = sanitize_feature_names(X_train.columns)
        X_test.columns = sanitize_feature_names(X_test.columns)

        print(f"Fold {i} X_train columns: {list(X_train.columns)}")
        print(f"Fold {i} X_test columns: {list(X_test.columns)}")

        print(f"  Fold {i}: Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        fold_predictions = model.predict(X_test)
        all_predictions.iloc[test_start_index:test_end_index] = fold_predictions

    data_with_features.loc[:, 'Predictions'] = all_predictions
    backtest_data = data_with_features.dropna(subset=['Predictions'])

    # Ensure data_for_backtest has the Predictions column for the same index
    data_for_backtest['Predictions'] = all_predictions
    
    # --- 3. Run the Backtest Simulation ---
    print("\nBacktester: Running simulation...")
    if backtest_data.empty:
        return None, "Not enough data to run a walk-forward backtest."

    # Rename price columns to standard names for Backtest
    for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
        price_col = [col for col in data_for_backtest.columns if col.startswith(base)]
        if price_col:
            data_for_backtest[base] = data_for_backtest[price_col[0]]

    # Use the preserved data_for_backtest (with price columns) for Backtest
    bt = Backtest(data_for_backtest.loc[backtest_data.index], MLStrategy, cash=cash, commission=commission)
    stats = bt.run()
    
    return stats, None
