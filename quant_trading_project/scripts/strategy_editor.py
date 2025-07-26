import streamlit as st
from datetime import datetime

from quant_trading_system.strategies.runner import run_strategy

st.title("Strategy Visual Editor")

strategy = st.selectbox("Select Strategy", ["SMA Crossover", "RSI"])

ticker = st.text_input("Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=datetime(2024, 1, 1)).strftime("%Y-%m-%d")
end_date = st.date_input("End Date", value=datetime.now()).strftime("%Y-%m-%d")

params = {}
if strategy == "SMA Crossover":
    params["short_window"] = st.number_input("Short Window", min_value=5, max_value=50, value=10)
    params["long_window"] = st.number_input("Long Window", min_value=10, max_value=200, value=20)
elif strategy == "RSI":
    params["rsi_period"] = st.number_input("RSI Period", min_value=2, max_value=50, value=14)
    params["overbought"] = st.number_input("Overbought Level", min_value=50, max_value=90, value=70)
    params["oversold"] = st.number_input("Oversold Level", min_value=10, max_value=50, value=30)

run = st.button("Run Backtest")

if run:
    with st.spinner("Running backtest..."):
        try:
            bt, stats = run_strategy(strategy, ticker, start_date, end_date, params)
            st.success("Backtest completed")
            st.write(stats)
            st.plotly_chart(bt.plot(), use_container_width=True)
        except Exception as e:
            st.error(str(e))
