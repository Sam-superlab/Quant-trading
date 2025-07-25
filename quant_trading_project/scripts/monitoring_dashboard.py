# scripts/monitoring_dashboard.py

import streamlit as st
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
import os
import sys

# --- Path Fix ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------

# Import all our custom modules
from quant_trading_system.data.data_fetcher import DataFetcher
from quant_trading_system.data.data_preprocessor import DataPreprocessor
from quant_trading_system.features.feature_engineering import FeatureEngineering
from quant_trading_system.execution.execution_handler import ExecutionHandler
from quant_trading_system.risk.risk_manager import RiskManager
from quant_trading_system.models.backtester import run_backtest
from quant_trading_system.utils.config import config

def run_strategy_pipeline(ticker, start_date, end_date):
    """
    Main pipeline to get data and generate a prediction, now including fundamental data.
    """
    logs = []
    fetcher = DataFetcher(finnhub_api_key=config.FINNHUB_API_KEY)
    preprocessor = DataPreprocessor()

    # --- 1. Data Pipeline (Now fetches and aligns fundamental data) ---
    logs.append(f"STEP 1: Running Data Pipeline for {ticker}...")
    market_data = fetcher.get_market_data(ticker, start_date=start_date, end_date=end_date)
    income_statement = fetcher.get_fundamental_data(ticker, statement_type='ic', freq='quarterly')
    
    if market_data is None or income_statement is None:
        logs.append("Error: Could not fetch market or fundamental data. Exiting.")
        return {'logs': logs, 'status': 'Error'}

    # Align data to prevent lookahead bias
    concepts_to_align = ['us-gaap_Revenues', 'us-gaap_NetIncomeLoss']
    aligned_data = preprocessor.align_market_and_fundamental_data(
        market_df=market_data,
        fundamental_df=income_statement,
        fundamental_cols=concepts_to_align
    )
    aligned_data.rename(columns={'us-gaap_Revenues': 'QuarterlyRevenue', 'us-gaap_NetIncomeLoss': 'QuarterlyNetIncome'}, inplace=True)
    
    clean_data = preprocessor.handle_missing_values(aligned_data, method='ffill')
    
    # --- 2. Feature Engineering (Now includes fundamental features) ---
    feature_generator = FeatureEngineering(clean_data)
    feature_generator.add_technical_indicators()
    feature_generator.add_fundamental_features()
    
    data_with_features = feature_generator.get_feature_data()
    logs.append("Data and feature pipeline completed successfully.")

    # --- 3. Model Training & Prediction ---
    logs.append("\nSTEP 2: Training Model and Generating Today's Signal...")
    
    # Debug: Check the date range of our data
    logs.append(f"Data date range: {data_with_features.index.min()} to {data_with_features.index.max()}")
    logs.append(f"Today's date: {datetime.now().date()}")
    
    data_with_features['Future_Return'] = data_with_features['Returns'].shift(-1)
    data_with_features['Target'] = (data_with_features['Future_Return'] > 0).astype(int)
    
    # Check if we have data for today or the most recent trading day
    latest_date = data_with_features.index.max()
    
    # Handle NaT (Not a Time) values in the index
    if pd.isna(latest_date):
        logs.append("Error: Invalid dates found in data. Cleaning data...")
        # Remove rows with NaT index
        data_with_features = data_with_features.dropna()
        if data_with_features.empty:
            logs.append("Error: No valid data remaining after cleaning.")
            return {'logs': logs, 'status': 'Error', 'data': data_with_features}
        latest_date = data_with_features.index.max()
    
    logs.append(f"Latest available data date: {latest_date.date()}")
    
    # If we don't have today's data, use the most recent available data for prediction
    if latest_date.date() < datetime.now().date():
        logs.append(f"Using most recent available data ({latest_date.date()}) for prediction")
        # Use the last row for prediction instead of looking for NaN target
        data_to_predict = data_with_features.iloc[[-1]]  # Last row
        training_data = data_with_features.iloc[:-1]  # All but last row
    else:
        # We have today's data, use the standard approach
        data_to_predict = data_with_features[data_with_features['Target'].isna()]
        training_data = data_with_features.dropna()

    y_train = training_data['Target']
    
    # Define base columns to drop
    base_cols_to_drop = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Future_Return', 'Target']
    
    # Add fundamental columns only if they exist
    fundamental_cols = ['QuarterlyRevenue', 'QuarterlyNetIncome']
    cols_to_drop = base_cols_to_drop + [col for col in fundamental_cols if col in data_with_features.columns]
    
    # Drop columns that exist
    existing_cols_to_drop = [col for col in cols_to_drop if col in data_with_features.columns]
    X_train = training_data.drop(columns=existing_cols_to_drop)
    X_predict = data_to_predict.drop(columns=existing_cols_to_drop)

    # Sanitize feature names for LightGBM
    def sanitize_column(col):
        return (
            str(col)
            .replace('[', '_')
            .replace(']', '_')
            .replace('{', '_')
            .replace('}', '_')
            .replace('"', '_')
            .replace("'", '_')
            .replace(':', '_')
            .replace(',', '_')
            .replace(' ', '_')
        )
    X_train.columns = [sanitize_column(col) for col in X_train.columns]
    X_predict.columns = [sanitize_column(col) for col in X_predict.columns]

    if X_predict.empty:
        logs.append("Warning: No data available for today's prediction.")
        return {'logs': logs, 'status': 'No Prediction', 'data': data_with_features}

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    prediction_proba = model.predict_proba(X_predict)
    probability_of_up = prediction_proba[0][1]
    prediction_date = X_predict.index[0].date()
    logs.append(f"Prediction for {prediction_date}: Probability of UP = {probability_of_up:.2%}")
    
    # If we're predicting for a past date, add a note
    if prediction_date < datetime.now().date():
        logs.append(f"Note: Predicting for {prediction_date} (most recent available data)")

    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X_train.columns)), columns=['Value','Feature'])

    return {
        'logs': logs, 
        'status': 'Signal Generated', 
        'probability': probability_of_up, 
        'data_to_predict': data_to_predict,
        'feature_importance': feature_imp,
        'data': data_with_features
    }

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📈 Quantitative Trading System Dashboard")

# --- Sidebar for Configuration ---
st.sidebar.header("Strategy Configuration")
ticker_input = st.sidebar.text_input("Ticker Symbol", config.DEFAULT_TICKER)
confidence_input = st.sidebar.slider("Confidence Threshold", 0.50, 0.70, config.CONFIDENCE_THRESHOLD, 0.01)
risk_input = st.sidebar.slider("Risk Percentage", 0.005, 0.05, config.RISK_PERCENTAGE, 0.001, format="%.3f")

# Date range configuration
st.sidebar.header("Date Range Configuration")
use_custom_dates = st.sidebar.checkbox("Use Custom Date Range", value=False)

if use_custom_dates:
    # Allow user to set custom start and end dates
    default_start = (datetime.now() - timedelta(days=config.TRAINING_HISTORY_YEARS * 365)).strftime('%Y-%m-%d')
    default_end = datetime.now().strftime('%Y-%m-%d')
    
    start_date_input = st.sidebar.date_input(
        "Start Date", 
        value=datetime.strptime(default_start, '%Y-%m-%d').date(),
        max_value=datetime.now().date()
    )
    end_date_input = st.sidebar.date_input(
        "End Date", 
        value=datetime.strptime(default_end, '%Y-%m-%d').date(),
        max_value=datetime.now().date()
    )
    
    # Convert to string format
    start_date = start_date_input.strftime('%Y-%m-%d')
    end_date = end_date_input.strftime('%Y-%m-%d')
else:
    # Use default date range
    start_date = (datetime.now() - timedelta(days=config.TRAINING_HISTORY_YEARS * 365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

st.sidebar.header("Account Information")
equity_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()

# --- Main Dashboard Area ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Strategy Execution")
    if st.button("🚀 Run Live Strategy Analysis"):
        with st.spinner("Executing strategy pipeline..."):
            results = run_strategy_pipeline(ticker_input, start_date, end_date)
            st.session_state['results'] = results

with col2:
    st.header("Backtesting")
    if st.button("🔍 Run Backtest"):
        with st.spinner("Running historical backtest..."):
            stats, error = run_backtest(ticker_input, config.BACKTEST_CASH, config.BACKTEST_COMMISSION)
            if error:
                st.error(error)
            else:
                st.session_state['backtest_stats'] = stats

# --- Display Results ---
if 'results' in st.session_state:
    results = st.session_state['results']
    
    st.header("Analysis Results")
    
    # --- Display Logs and Charts in Columns ---
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Execution Logs")
        st.text_area("Logs", "".join(f"{log}\n" for log in results['logs']), height=250)
        
        st.subheader("Price vs. VWAP Chart")
        # Flatten MultiIndex columns if present
        if isinstance(results['data'].columns, pd.MultiIndex):
            results['data'].columns = ['_'.join([str(c) for c in col if c and c != '']) for col in results['data'].columns]
        
        # Find the correct column names after flattening
        close_col = None
        vwap_col = None
        for col in results['data'].columns:
            if 'Close' in col:
                close_col = col
            elif 'VWAP_14' in col:
                vwap_col = col
        
        # Plot Close price and the new VWAP indicator
        if close_col and vwap_col:
            chart_data = results['data'][[close_col, vwap_col]].tail(200) # Plot last 200 days
            st.line_chart(chart_data)
        else:
            st.warning("Could not find Close or VWAP_14 columns for plotting")
            st.write("Available columns:", list(results['data'].columns))

    with res_col2:
        if results['status'] == 'Signal Generated':
            st.subheader("Model's Reasoning (Feature Importance)")
            st.bar_chart(results['feature_importance'].sort_values('Value', ascending=False).head(10).set_index('Feature'))

            # --- Trade Execution Logic ---
            st.subheader("Trade Decision")
            if results['probability'] > confidence_input:
                st.success(f"Signal is BUY (Confidence: {results['probability']:.2%})")
                try:
                    execution_handler = ExecutionHandler(paper_trading=config.PAPER_TRADING)
                    account_info = execution_handler.get_account_info()
                    
                    if account_info:
                        equity_placeholder.metric("Account Equity", f"${float(account_info.equity):,.2f}")
                        status_placeholder.success(f"Connected: {account_info.status}")
                        risk_manager = RiskManager(equity=float(account_info.equity))
                        
                        data_to_predict = results['data_to_predict']
                        current_price = data_to_predict['Close'].iloc[0]
                        atr = data_to_predict['ATR'].iloc[0]
                        stop_loss_price = current_price - (2 * atr)
                        take_profit_price = current_price + (1.5 * (current_price - stop_loss_price))
                        
                        st.info("Live trading execution is commented out for safety.")
                    else:
                        status_placeholder.error("Connection Failed")
                except Exception as e:
                    st.error(f"Execution Error: {e}")
            else:
                st.warning(f"Signal is HOLD (Confidence is below threshold)")

if 'backtest_stats' in st.session_state:
    st.header("Backtest Results")
    st.dataframe(st.session_state['backtest_stats'])
