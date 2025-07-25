# quant_trading_system/models/backtester.py

import pandas as pd
import lightgbm as lgb
from backtesting import Backtest, Strategy
from sklearn.model_selection import train_test_split

# Import our existing modules to build the full pipeline
from data_fetcher import DataFetcher
from data_preprocessor import DataPreprocessor
from feature_engineering import FeatureEngineering

class MLStrategy(Strategy):
    """
    A trading strategy that uses a pre-trained machine learning model
    to make trading decisions. This class is designed to work with the
    backtesting.py library.
    """
    def init(self):
        """
        Initialize the strategy. We receive the model's predictions
        as a data series.
        """
        # The `backtesting.py` library automatically aligns the data.
        # We will pass the predictions as an additional data column.
        self.predictions = self.I(lambda x: x, self.data.df['Predictions'])

    def next(self):
        """
        The main trading logic that is executed on each bar of data.
        """
        # If the model predicts an "Up" move (1) and we are not in a position, buy.
        if self.predictions[-1] == 1 and not self.position:
            self.buy()
        # If the model predicts a "Down/Same" move (0) and we are in a position, sell.
        elif self.predictions[-1] == 0 and self.position:
            self.position.close()

def run_backtest():
    """
    Main function to execute the entire pipeline from data fetching to backtesting.
    """
    # --- Setup: This pipeline uses all the modules we've built so far ---
    user_api_key = 'd21bk3pr01qkdupiodggd21bk3pr01qkdupiodh0'
    fetcher = DataFetcher(finnhub_api_key=user_api_key)
    preprocessor = DataPreprocessor()
    
    ticker = 'TSLA'
    start = '2021-01-01'
    end = '2023-12-31'

    # --- 1. Fetch, Preprocess, and Engineer Features ---
    print("\n" + "="*50)
    print("STEP 1: Data Pipeline Execution")
    print("="*50)
    market_data = fetcher.get_market_data(ticker, start_date=start, end_date=end)
    if market_data is None:
        print("Could not fetch market data. Exiting.")
        return

    clean_data = preprocessor.handle_missing_values(market_data, method='ffill')
    
    feature_generator = FeatureEngineering(clean_data)
    feature_generator.add_moving_averages()
    feature_generator.add_momentum_indicators()
    feature_generator.add_volatility_indicators()
    feature_generator.add_lagged_returns()
    
    data_with_features = feature_generator.get_feature_data()

    # --- 2. Create Target and Prepare Data ---
    print("\n" + "="*50)
    print("STEP 2: Preparing Data for Model Training")
    print("="*50)
    
    # Create target variable
    data_with_features['Future_Return'] = data_with_features['Returns'].shift(-1)
    data_with_features['Target'] = (data_with_features['Future_Return'] > 0).astype(int)
    data_with_features.dropna(inplace=True)

    # Prepare feature matrix (X) and target vector (y)
    y = data_with_features['Target']
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Future_Return', 'Target']
    X = data_with_features.drop(columns=cols_to_drop)

    # --- 3. Train Model and Generate Predictions ---
    print("\n" + "="*50)
    print("STEP 3: Training Model and Generating Predictions")
    print("="*50)
    
    # We train on the first 80% of the data to simulate a hold-out set,
    # which is the period we will backtest on.
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Training model on data up to {X_train.index[-1].date()}...")
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Generate predictions for the entire dataset to pass to the backtester
    # The backtester will then slice this data appropriately.
    all_predictions = model.predict(X)
    data_with_features['Predictions'] = all_predictions
    
    # --- 4. Run the Backtest ---
    print("\n" + "="*50)
    print("STEP 4: Running Vectorized Backtest")
    print("="*50)

    # We will backtest on the "test" period, the last 20% of the data.
    backtest_data = data_with_features[split_index:]

    bt = Backtest(backtest_data, MLStrategy,
                  cash=100_000, commission=.002)

    stats = bt.run()
    
    print("\nBacktest Results:")
    print(stats)
    
    print("\nNOTE: The 'Equity Final' and 'Return [%]' are calculated only on the test period.")

    # You can uncomment the line below to generate an interactive plot
    # in a browser window.
    # bt.plot()

if __name__ == '__main__':
    # To run this, you will need to install the backtesting.py library:
    # pip install backtesting
    run_backtest()
