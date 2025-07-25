# scripts/run_live_strategy.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta

# Import all our custom modules
from quant_trading_system.data.data_fetcher import DataFetcher
from quant_trading_system.data.data_preprocessor import DataPreprocessor
from quant_trading_system.features.feature_engineering import FeatureEngineering
from quant_trading_system.execution.execution_handler import ExecutionHandler
from quant_trading_system.risk.risk_manager import RiskManager

def run_live_strategy():
    """
    This is the main operational script that runs the entire trading pipeline.
    It fetches data, trains a model, generates a signal for the current day,
    and places a trade if the signal is strong.
    """
    # --- Configuration ---
    user_api_key = 'd21bk3pr01qkdupiodggd21bk3pr01qkdupiodh0'
    ticker = 'NVDA'
    # Use a long history for robust model training
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # --- Modules Initialization ---
    fetcher = DataFetcher(finnhub_api_key=user_api_key)
    preprocessor = DataPreprocessor()

    # --- 1. Data Pipeline ---
    print("\n" + "="*50)
    print(f"STEP 1: Running Data Pipeline for {ticker}")
    print("="*50)
    market_data = fetcher.get_market_data(ticker, start_date=start_date, end_date=end_date)
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

    # --- 2. Model Training & Prediction ---
    print("\n" + "="*50)
    print("STEP 2: Training Model and Generating Today's Signal")
    print("="*50)
    
    # Create target variable
    data_with_features['Future_Return'] = data_with_features['Returns'].shift(-1)
    data_with_features['Target'] = (data_with_features['Future_Return'] > 0).astype(int)
    
    # Check if we have data for today or the most recent trading day
    latest_date = data_with_features.index.max()
    
    # Handle NaT (Not a Time) values in the index
    if pd.isna(latest_date):
        print("Error: Invalid dates found in data. Cleaning data...")
        # Remove rows with NaT index
        data_with_features = data_with_features.dropna()
        if data_with_features.empty:
            print("Error: No valid data remaining after cleaning.")
            return
        latest_date = data_with_features.index.max()
    
    print(f"Latest available data date: {latest_date.date()}")
    print(f"Today's date: {datetime.now().date()}")
    
    # If we don't have today's data, use the most recent available data for prediction
    if latest_date.date() < datetime.now().date():
        print(f"Using most recent available data ({latest_date.date()}) for prediction")
        # Use the last row for prediction instead of looking for NaN target
        data_to_predict = data_with_features.iloc[[-1]]  # Last row
        training_data = data_with_features.iloc[:-1]  # All but last row
    else:
        # We have today's data, use the standard approach
        data_to_predict = data_with_features[data_with_features['Target'].isna()]
        training_data = data_with_features.dropna()

    y_train = training_data['Target']
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Future_Return', 'Target']
    X_train = training_data.drop(columns=cols_to_drop)
    
    # Prepare the feature vector for today's prediction
    X_predict = data_to_predict.drop(columns=cols_to_drop)

    if X_predict.empty:
        print("No data available for today's prediction. The market may be closed or data is delayed.")
        return

    print(f"Training model on {len(X_train)} historical data points...")
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict the probability for today
    prediction_proba = model.predict_proba(X_predict)
    probability_of_up = prediction_proba[0][1] # Probability of class '1' (Up)
    prediction_date = X_predict.index[0].date()
    
    print(f"Prediction for {prediction_date}:")
    print(f"  - Probability of price going UP: {probability_of_up:.2%}")
    
    # If we're predicting for a past date, add a note
    if prediction_date < datetime.now().date():
        print(f"  - Note: Predicting for {prediction_date} (most recent available data)")

    # --- 3. Trade Execution Logic ---
    print("\n" + "="*50)
    print("STEP 3: Applying Trading Logic and Risk Management")
    print("="*50)

    # Define a confidence threshold to place a trade
    CONFIDENCE_THRESHOLD = 0.55 

    if probability_of_up > CONFIDENCE_THRESHOLD:
        print(f"Signal is BUY (Confidence: {probability_of_up:.2%})")
        
        try:
            # Initialize execution and risk management modules
            execution_handler = ExecutionHandler(paper_trading=True)
            account_info = execution_handler.get_account_info()
            if not account_info: return
            
            risk_manager = RiskManager(equity=float(account_info.equity))

            # Define trade parameters based on the latest data
            current_price = data_to_predict['Close'].iloc[0]
            atr = data_to_predict['ATR'].iloc[0]
            
            # Set stop-loss based on volatility (e.g., 2 * ATR)
            stop_loss_price = current_price - (2 * atr)
            # Set a simple risk-reward ratio for the take-profit (e.g., 1.5:1)
            take_profit_price = current_price + (1.5 * (current_price - stop_loss_price))
            
            print(f"\nTrade Parameters for {ticker}:")
            print(f"  - Current Price: ${current_price:,.2f}")
            print(f"  - Take Profit: ${take_profit_price:,.2f}")
            print(f"  - Stop Loss: ${stop_loss_price:,.2f}")

            # Place the final, risk-managed bracket order
            execution_handler.place_bracket_order(
                symbol=ticker,
                side='buy',
                entry_price=current_price,
                risk_manager=risk_manager,
                risk_percentage=0.01, # Risk 1% of equity
                take_profit_price=round(take_profit_price, 2),
                stop_loss_price=round(stop_loss_price, 2)
            )
        except Exception as e:
            print(f"An error occurred during execution: {e}")
            
    else:
        print(f"Signal is HOLD (Confidence {probability_of_up:.2%} is below threshold of {CONFIDENCE_THRESHOLD:.2%})")
        print("No trade will be placed.")

if __name__ == '__main__':
    # This script can be scheduled to run daily before market open.
    run_live_strategy()
