import streamlit as st
from datetime import datetime

from quant_trading_system.strategies.runner import run_strategy

st.title("Strategy Visual Editor")

strategy = st.selectbox("Select Strategy", ["SMA Crossover", "RSI", "ML Model"])

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

risk = st.slider("Risk %", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

run = st.button("Run Backtest")

if run:
    with st.spinner("Running backtest..."):
        try:
            result = run_strategy(strategy, ticker, start_date, end_date, risk / 100, params)
            st.success("Backtest completed")
            st.write(result['stats'])

            if result['stats'] is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Return %", f"{result['stats']['Return [%]']:.2f}")
                with col2:
                    st.metric("Win Rate", f"{result['stats']['Win Rate [%]']:.1f}%")
                with col3:
                    sr = result['stats'].get('Sharpe Ratio', 0.0)
                    st.metric("Sharpe Ratio", f"{sr:.2f}")

            if result['data'] is not None:
                import plotly.graph_objects as go
                price_fig = go.Figure()
                price_fig.add_trace(go.Scatter(x=result['data'].index, y=result['data']['Close'], mode='lines', name='Close'))
                if result['trades'] is not None and not result['trades'].empty:
                    buys = result['trades'][result['trades']['Size'] > 0]
                    sells = result['trades'][result['trades']['Size'] < 0]
                    price_fig.add_trace(go.Scatter(x=buys['EntryTime'], y=buys['EntryPrice'], mode='markers', marker_symbol='triangle-up', marker_color='green', name='Buy'))
                    price_fig.add_trace(go.Scatter(x=sells['EntryTime'], y=sells['EntryPrice'], mode='markers', marker_symbol='triangle-down', marker_color='red', name='Sell'))
                st.plotly_chart(price_fig, use_container_width=True)

            if result['pnl_curve'] is not None:
                import plotly.express as px
                fig = px.line(result['pnl_curve'], title='Equity Curve')
                st.plotly_chart(fig, use_container_width=True)

            if result.get('confusion_matrix') is not None:
                import plotly.express as px
                cm_fig = px.imshow(
                    result['confusion_matrix'],
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count")
                )
                st.plotly_chart(cm_fig, use_container_width=True)

            if result['feature_importance'] is not None:
                st.write("### Top Features")
                st.dataframe(result['feature_importance'].head())

            if result['trades'] is not None:
                with st.expander("Trade Log"):
                    st.dataframe(result['trades'])
        except Exception as e:
            st.error(str(e))
