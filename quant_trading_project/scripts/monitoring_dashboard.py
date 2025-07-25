# scripts/monitoring_dashboard.py

import sys
import os
os.environ['APCA_API_KEY_ID'] = 'PK118EYMNJDA4352MEXH'
os.environ['APCA_API_SECRET_KEY'] = '5WxwMbB9j2q3pBNms0Yur2DfA3vqSft6s9XwQwxR'
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets/v2'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
import time

# Import all our custom modules from the project structure
# Note: For this to run, your project structure must be set up correctly
# and you should run streamlit from the root of 'quant_trading_project'.
# Example: `streamlit run scripts/monitoring_dashboard.py`
from quant_trading_system.data.data_fetcher import DataFetcher
from quant_trading_system.data.data_preprocessor import DataPreprocessor
from quant_trading_system.features.feature_engineering import FeatureEngineering
from quant_trading_system.execution.execution_handler import ExecutionHandler
from quant_trading_system.risk.risk_manager import RiskManager

print('Python executable:', sys.executable)

def run_strategy_pipeline(ticker, start_date, end_date, user_api_key):
    """
    This is the refactored operational script. Instead of printing, it
    captures logs and returns a final result dictionary.
    """
    logs = []
    
    # --- Modules Initialization ---
    fetcher = DataFetcher(finnhub_api_key=user_api_key)
    preprocessor = DataPreprocessor()

    # --- 1. Data Pipeline ---
    logs.append(f"STEP 1: Running Data Pipeline for {ticker}...")
    market_data = fetcher.get_market_data(ticker, start_date=start_date, end_date=end_date)
    if market_data is None:
        logs.append("Error: Could not fetch market data. Exiting.")
        return {'logs': logs, 'status': 'Error'}

    clean_data = preprocessor.handle_missing_values(market_data, method='ffill')
    feature_generator = FeatureEngineering(clean_data)
    feature_generator.add_moving_averages().add_momentum_indicators().add_volatility_indicators().add_lagged_returns()
    data_with_features = feature_generator.get_feature_data()
    logs.append("Data pipeline completed successfully.")
    logs.append(f"Last available date in data: {data_with_features.index[-1]}")

    # --- 2. Model Training & Prediction ---
    logs.append("\nSTEP 2: Training Model and Generating Today's Signal...")
    data_with_features['Future_Return'] = data_with_features['Returns'].shift(-1)
    data_with_features['Target'] = (data_with_features['Future_Return'] > 0).astype(int)
    data_to_predict = data_with_features[data_with_features['Target'].isna()]
    training_data = data_with_features.dropna()

    if data_to_predict.empty:
        # Fallback: use the last row for prediction
        data_to_predict = data_with_features.iloc[[-1]]
        logs.append("No future row to predict. Using last available row for prediction.")

    y_train = training_data['Target']
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Future_Return', 'Target']
    X_train = training_data.drop(columns=cols_to_drop)
    X_predict = data_to_predict.drop(columns=cols_to_drop)

    if X_predict.empty:
        logs.append("Warning: No data available for today's prediction. Market may be closed.")
        return {'logs': logs, 'status': 'No Prediction'}

    # Flatten and sanitize feature names for LightGBM compatibility
    if isinstance(X_train.columns, pd.MultiIndex):
        X_train.columns = ['_'.join([str(i) for i in col if i]) for col in X_train.columns]
        X_predict.columns = X_train.columns  # Ensure prediction columns match
    X_train.columns = [col.replace('[','').replace(']','').replace('(','').replace(')','').replace(' ','_') for col in X_train.columns]
    X_predict.columns = X_train.columns

    logs.append(f"Training model on {len(X_train)} historical data points...")
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    prediction_proba = model.predict_proba(X_predict)
    probability_of_up = prediction_proba[0][1]
    logs.append(f"Prediction for {X_predict.index[0].date()}: Probability of UP = {probability_of_up:.2%}")

    # --- 3. Trade Execution Logic ---
    logs.append("\nSTEP 3: Applying Trading Logic...")
    CONFIDENCE_THRESHOLD = 0.55
    if probability_of_up > CONFIDENCE_THRESHOLD:
        signal = 'BUY'
        logs.append(f"Signal is BUY (Confidence: {probability_of_up:.2%})")
        return {'logs': logs, 'status': 'Trade', 'signal': signal, 'ticker': ticker, 'data_to_predict': data_to_predict}
    else:
        signal = 'HOLD'
        logs.append(f"Signal is HOLD (Confidence {probability_of_up:.2%} is below threshold)")
        return {'logs': logs, 'status': 'Hold', 'signal': signal}


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📈 Quantitative Trading System Dashboard")
st.write("This dashboard controls and monitors the live trading strategy pipeline.")

# --- Sidebar for Configuration ---
st.sidebar.header("Strategy Configuration")
ticker_input = st.sidebar.text_input("Ticker Symbol", "NVDA")
api_key_input = st.sidebar.text_input("Finnhub API Key", "d21bk3pr01qkdupiodggd21bk3pr01qkdupiodh0", type="password")

st.sidebar.header("Account Information")
# Placeholders for account info
equity_placeholder = st.sidebar.empty()
buying_power_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()

# --- Main Dashboard Area ---
if st.button("🚀 Run Live Strategy"):
    with st.spinner("Executing strategy pipeline... This may take a minute."):
        # --- Run the pipeline ---
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        results = run_strategy_pipeline(ticker_input, start_date, end_date, api_key_input)
        
        # --- Display Logs ---
        with st.expander("Show Strategy Execution Logs"):
            st.text_area("Logs", "".join(f"{log}\n" for log in results['logs']), height=300)

        # --- Handle Trade Execution ---
        if results.get('status') == 'Trade':
            st.subheader("📈 Trade Execution")
            with st.spinner("Placing risk-managed trade..."):
                try:
                    execution_handler = ExecutionHandler(paper_trading=True)
                    account_info = execution_handler.get_account_info()
                    
                    if account_info:
                        equity_placeholder.metric("Account Equity", f"${float(account_info.equity):,.2f}")
                        buying_power_placeholder.metric("Buying Power", f"${float(account_info.buying_power):,.2f}")
                        status_placeholder.success(f"Connected: {account_info.status}")

                        risk_manager = RiskManager(equity=float(account_info.equity))
                        
                        data_to_predict = results['data_to_predict']
                        current_price = data_to_predict['Close']
                        atr = data_to_predict['ATR']
                        if hasattr(current_price, 'iloc'):
                            current_price = current_price.iloc[0]
                        if hasattr(atr, 'iloc'):
                            atr = atr.iloc[0]
                        current_price = float(current_price)
                        atr = float(atr)
                        stop_loss_price = current_price - (2 * atr)
                        take_profit_price = current_price + (1.5 * (current_price - stop_loss_price))
                        stop_loss_price = float(stop_loss_price)
                        take_profit_price = float(take_profit_price)
                        
                        st.write(f"**Placing {results['signal']} order for {results['ticker']}**")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Entry Price", f"${current_price:,.2f}")
                        col2.metric("Take Profit", f"${take_profit_price:,.2f}")
                        col3.metric("Stop Loss", f"${stop_loss_price:,.2f}")

                        # This would place a real paper trade
                        # execution_handler.place_bracket_order(...)

                    else:
                        status_placeholder.error("Connection Failed")

                except Exception as e:
                    st.error(f"An error occurred during execution: {e}")
        else:
            st.info("No trade signal generated.")
