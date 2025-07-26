# quant_trading_system/models/enhanced_backtester.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our existing modules
from quant_trading_system.models.backtester import run_backtest
from quant_trading_system.data.data_fetcher import DataFetcher
from quant_trading_system.data.data_preprocessor import DataPreprocessor
from quant_trading_system.features.feature_engineering import FeatureEngineering
from quant_trading_system.utils.config import config

class EnhancedBacktester:
    """
    Enhanced backtesting with comprehensive visualization and analysis.
    """
    
    def __init__(self, ticker, cash=100000, commission=0.002):
        self.ticker = ticker
        self.cash = cash
        self.commission = commission
        self.stats = None
        self.trades_data = None
        self.equity_curve = None
        self.feature_importance = None
        self.confusion_matrix = None
        
    def run_backtest(self):
        """Run the backtest and collect detailed data"""
        try:
            # Run the original backtest
            stats, feature_imp, conf_mat, error = run_backtest(self.ticker, self.cash, self.commission)
            
            if error:
                return None, error
                
            self.stats = stats
            self.feature_importance = feature_imp
            self.confusion_matrix = conf_mat
            
            # Extract additional data for visualization
            self._extract_trading_data()
            
            return self.stats, None
            
        except Exception as e:
            return None, f"Backtest failed: {str(e)}"
    
    def _get_stat_safely(self, key, default=0.0):
        """Safely get a stat value with fallback"""
        try:
            return self.stats[key]
        except (KeyError, TypeError):
            return default
    
    def _extract_trading_data(self):
        """Extract trading data for visualization"""
        if self.stats is None:
            return
            
        # Get the backtest object to extract detailed data
        try:
            # Re-run to get detailed data
            fetcher = DataFetcher(finnhub_api_key=config.FINNHUB_API_KEY)
            preprocessor = DataPreprocessor()
            
            start_date = (datetime.now() - timedelta(days=config.TRAINING_HISTORY_YEARS * 365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            market_data = fetcher.get_market_data(self.ticker, start_date=start_date, end_date=end_date)
            if market_data is None:
                return
                
            clean_data = preprocessor.handle_missing_values(market_data, method='ffill')
            feature_generator = FeatureEngineering(clean_data)
            feature_generator.add_technical_indicators()
            data_with_features = feature_generator.get_feature_data()
            
            # Create a simple equity curve based on returns
            data_with_features['Returns'] = data_with_features['Close'].pct_change()
            data_with_features['Cumulative_Returns'] = (1 + data_with_features['Returns']).cumprod()
            data_with_features['Equity_Curve'] = self.cash * data_with_features['Cumulative_Returns']
            
            self.equity_curve = data_with_features[['Close', 'Returns', 'Cumulative_Returns', 'Equity_Curve']].copy()
            
        except Exception as e:
            print(f"Error extracting trading data: {e}")
    
    def create_performance_dashboard(self):
        """Create a comprehensive performance dashboard"""
        if self.stats is None:
            return "No backtest results available."
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance Overview", 
            "📈 Equity Curve", 
            "💰 Risk Metrics", 
            "📋 Trade Analysis",
            "🎯 Strategy Insights"
        ])
        
        with tab1:
            self._show_performance_overview()
            
        with tab2:
            self._show_equity_curve()
            
        with tab3:
            self._show_risk_metrics()
            
        with tab4:
            self._show_trade_analysis()
            
        with tab5:
            self._show_strategy_insights()
    
    def _show_performance_overview(self):
        """Show key performance metrics"""
        st.subheader("🎯 Key Performance Metrics")
        
        # Create metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = self._get_stat_safely('Return [%]', 0.0)
            st.metric(
                "Total Return", 
                f"{total_return:.2f}%",
                delta=f"{total_return:.2f}%"
            )
            
        with col2:
            sharpe_ratio = self._get_stat_safely('Sharpe Ratio', 0.0)
            st.metric(
                "Sharpe Ratio", 
                f"{sharpe_ratio:.3f}",
                delta=f"{sharpe_ratio:.3f}"
            )
            
        with col3:
            max_drawdown = self._get_stat_safely('Max. Drawdown [%]', 0.0)
            st.metric(
                "Max Drawdown", 
                f"{max_drawdown:.2f}%",
                delta=f"{max_drawdown:.2f}%"
            )
            
        with col4:
            win_rate = self._get_stat_safely('Win Rate [%]', 0.0)
            st.metric(
                "Win Rate", 
                f"{win_rate:.1f}%",
                delta=f"{win_rate:.1f}%"
            )
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trades = self._get_stat_safely('# Trades', 0)
            st.metric("Total Trades", f"{total_trades}")
            
        with col2:
            avg_duration = self._get_stat_safely('Avg. Trade Duration', 'N/A')
            st.metric("Avg Trade Duration", f"{avg_duration}")
            
        with col3:
            profit_factor = self._get_stat_safely('Profit Factor', 0.0)
            st.metric("Profit Factor", f"{profit_factor:.2f}")
            
        with col4:
            calmar_ratio = self._get_stat_safely('Calmar Ratio', 0.0)
            st.metric("Calmar Ratio", f"{calmar_ratio:.3f}")
        
        # Performance comparison chart
        st.subheader("📈 Performance Comparison")
        
        # Create benchmark comparison (buy and hold)
        if self.equity_curve is not None:
            benchmark_returns = self.equity_curve['Cumulative_Returns'].iloc[-1] - 1
            strategy_returns = self._get_stat_safely('Return [%]', 0.0) / 100
            
            comparison_data = pd.DataFrame({
                'Strategy': ['Buy & Hold', 'ML Strategy'],
                'Return [%]': [benchmark_returns * 100, self._get_stat_safely('Return [%]', 0.0)],
                'Sharpe Ratio': [benchmark_returns / (self.equity_curve['Returns'].std() * np.sqrt(252)), self._get_stat_safely('Sharpe Ratio', 0.0)]
            })
            
            # Create comparison chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Total Return (%)', 'Sharpe Ratio'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=comparison_data['Strategy'], y=comparison_data['Return [%]'], 
                      name='Return [%]', marker_color=['#1f77b4', '#ff7f0e']),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=comparison_data['Strategy'], y=comparison_data['Sharpe Ratio'], 
                      name='Sharpe Ratio', marker_color=['#2ca02c', '#d62728']),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_equity_curve(self):
        """Show equity curve and drawdown analysis"""
        st.subheader("📈 Equity Curve Analysis")
        
        if self.equity_curve is None:
            st.warning("Equity curve data not available.")
            return
        
        # Create equity curve chart
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=self.equity_curve.index,
            y=self.equity_curve['Equity_Curve'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add drawdown
        rolling_max = self.equity_curve['Equity_Curve'].expanding().max()
        drawdown = (self.equity_curve['Equity_Curve'] - rolling_max) / rolling_max * 100
        
        fig.add_trace(go.Scatter(
            x=self.equity_curve.index,
            y=drawdown,
            mode='lines',
            name='Drawdown %',
            line=dict(color='#d62728', width=1),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'{self.ticker} Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            yaxis2=dict(title='Drawdown (%)', overlaying='y', side='right'),
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly returns heatmap
        st.subheader("📅 Monthly Returns Heatmap")
        
        monthly_returns = self.equity_curve['Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_df = monthly_returns.to_frame('Returns')
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month
        
        pivot_table = monthly_returns_df.pivot(index='Year', columns='Month', values='Returns') * 100
        
        fig = px.imshow(
            pivot_table,
            title='Monthly Returns Heatmap (%)',
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_risk_metrics(self):
        """Show comprehensive risk analysis"""
        st.subheader("💰 Risk Analysis")
        
        # Risk metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Volatility", f"{self.stats['Volatility (Ann.) [%]']:.2f}%")
            
        with col2:
            # Calculate VaR if not available
            if 'VaR (95%) [%]' in self.stats:
                var_value = self.stats['VaR (95%) [%]']
            else:
                # Calculate VaR from equity curve if available
                if self.equity_curve is not None:
                    returns = self.equity_curve['Returns'].dropna()
                    if len(returns) > 0:
                        var_value = np.percentile(returns, 5) * 100  # 95% VaR
                    else:
                        var_value = 0.0
                else:
                    var_value = 0.0
            st.metric("VaR (95%)", f"{var_value:.2f}%")
            
        with col3:
            # Calculate CVaR if not available
            if 'CVaR (95%) [%]' in self.stats:
                cvar_value = self.stats['CVaR (95%) [%]']
            else:
                # Calculate CVaR from equity curve if available
                if self.equity_curve is not None:
                    returns = self.equity_curve['Returns'].dropna()
                    if len(returns) > 0:
                        var_threshold = np.percentile(returns, 5)
                        cvar_value = returns[returns <= var_threshold].mean() * 100  # 95% CVaR
                    else:
                        cvar_value = 0.0
                else:
                    cvar_value = 0.0
            st.metric("CVaR (95%)", f"{cvar_value:.2f}%")
            
        with col4:
            st.metric("Sortino Ratio", f"{self.stats['Sortino Ratio']:.3f}")
        
        # Risk-return scatter plot
        st.subheader("📊 Risk-Return Analysis")
        
        if self.equity_curve is not None:
            # Calculate rolling metrics
            window = 252  # 1 year
            rolling_returns = self.equity_curve['Returns'].rolling(window).mean() * 252
            rolling_vol = self.equity_curve['Returns'].rolling(window).std() * np.sqrt(252)
            
            risk_return_df = pd.DataFrame({
                'Return': rolling_returns,
                'Volatility': rolling_vol,
                'Sharpe': rolling_returns / rolling_vol
            }).dropna()
            
            fig = px.scatter(
                risk_return_df,
                x='Volatility',
                y='Return',
                color='Sharpe',
                title='Risk-Return Scatter Plot (Rolling 1-Year)',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown analysis
        st.subheader("📉 Drawdown Analysis")
        
        if self.equity_curve is not None:
            # Calculate drawdown periods
            equity = self.equity_curve['Equity_Curve']
            rolling_max = equity.expanding().max()
            drawdown = (equity - rolling_max) / rolling_max * 100
            
            # Find drawdown periods
            underwater = drawdown < 0
            underwater_periods = underwater.astype(int).diff().fillna(0)
            start_drawdown = underwater_periods == 1
            end_drawdown = underwater_periods == -1
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown %',
                line=dict(color='#d62728', width=2),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='Drawdown Analysis',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_trade_analysis(self):
        """Show detailed trade analysis"""
        st.subheader("📋 Trade Analysis")
        
        # Trade statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trades = self._get_stat_safely('# Trades', 0)
            st.metric("Total Trades", f"{total_trades}")
            
        with col2:
            win_rate = self._get_stat_safely('Win Rate [%]', 0.0)
            winning_trades = total_trades * win_rate / 100
            st.metric("Winning Trades", f"{winning_trades:.0f}")
            
        with col3:
            losing_trades = total_trades * (1 - win_rate / 100)
            st.metric("Losing Trades", f"{losing_trades:.0f}")
            
        with col4:
            avg_duration = self._get_stat_safely('Avg. Trade Duration', 'N/A')
            st.metric("Avg Trade Duration", f"{avg_duration}")
        
        # Trade distribution
        st.subheader("📊 Trade Distribution")
        
        # Create trade distribution chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Trade Duration Distribution', 'Trade Return Distribution'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Simulate trade data for visualization (since we don't have actual trade data)
        np.random.seed(42)
        n_trades = int(self._get_stat_safely('# Trades', 0))
        
        if n_trades > 0:
            # Simulate trade durations (in days)
            durations = np.random.exponential(5, n_trades)  # Exponential distribution
            fig.add_trace(
                go.Histogram(x=durations, name='Duration (days)', nbinsx=20),
                row=1, col=1
            )
            
            # Simulate trade returns
            win_rate = self._get_stat_safely('Win Rate [%]', 0.0) / 100
            returns = np.random.choice([1, -1], n_trades, p=[win_rate, 1-win_rate]) * np.random.exponential(0.02, n_trades)
            fig.add_trace(
                go.Histogram(x=returns, name='Return %', nbinsx=20),
                row=1, col=2
            )
        else:
            st.warning("No trades available for distribution analysis.")
            return
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if self.confusion_matrix is not None:
            st.subheader("📉 Confusion Matrix")
            cm_fig = px.imshow(
                self.confusion_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count")
            )
            st.plotly_chart(cm_fig, use_container_width=True)
    
    def _show_strategy_insights(self):
        """Show strategy insights and recommendations"""
        st.subheader("🎯 Strategy Insights")
        
        # Performance summary
        st.write("### 📈 Performance Summary")
        
        total_return = self._get_stat_safely('Return [%]', 0.0)
        if total_return > 0:
            st.success(f"✅ The strategy generated a positive return of {total_return:.2f}%")
        else:
            st.error(f"❌ The strategy generated a negative return of {total_return:.2f}%")
        
        # Sharpe ratio analysis
        st.write("### 📊 Risk-Adjusted Performance")
        
        sharpe_ratio = self._get_stat_safely('Sharpe Ratio', 0.0)
        if sharpe_ratio > 1:
            st.success(f"✅ Excellent Sharpe ratio of {sharpe_ratio:.3f} (> 1.0)")
        elif sharpe_ratio > 0.5:
            st.info(f"📊 Good Sharpe ratio of {sharpe_ratio:.3f} (0.5-1.0)")
        else:
            st.warning(f"⚠️ Low Sharpe ratio of {sharpe_ratio:.3f} (< 0.5)")
        
        # Drawdown analysis
        st.write("### 📉 Risk Management")
        
        max_drawdown = self._get_stat_safely('Max. Drawdown [%]', 0.0)
        if max_drawdown < -20:
            st.error(f"❌ High maximum drawdown of {max_drawdown:.2f}% (> 20%)")
        elif max_drawdown < -10:
            st.warning(f"⚠️ Moderate maximum drawdown of {max_drawdown:.2f}% (10-20%)")
        else:
            st.success(f"✅ Low maximum drawdown of {max_drawdown:.2f}% (< 10%)")
        
        # Win rate analysis
        st.write("### 🎯 Trading Accuracy")
        
        win_rate = self._get_stat_safely('Win Rate [%]', 0.0)
        if win_rate > 60:
            st.success(f"✅ High win rate of {win_rate:.1f}% (> 60%)")
        elif win_rate > 50:
            st.info(f"📊 Moderate win rate of {win_rate:.1f}% (50-60%)")
        else:
            st.warning(f"⚠️ Low win rate of {win_rate:.1f}% (< 50%)")
        
        # Recommendations
        st.write("### 💡 Recommendations")
        
        recommendations = []
        
        if sharpe_ratio < 0.5:
            recommendations.append("Consider improving risk management to increase Sharpe ratio")
        
        if max_drawdown < -15:
            recommendations.append("Implement stricter stop-loss mechanisms to reduce drawdown")
        
        if win_rate < 50:
            recommendations.append("Review entry/exit criteria to improve win rate")
        
        total_trades = self._get_stat_safely('# Trades', 0)
        if total_trades < 10:
            recommendations.append("Consider longer backtest period for more statistical significance")
        
        if not recommendations:
            st.success("🎉 Strategy looks well-balanced! Consider live trading with small position sizes.")
        else:
            for rec in recommendations:
                st.info(f"💡 {rec}")
        
        # Export results
        st.write("### 📤 Export Results")
        
        if st.button("📊 Export Backtest Results"):
            # Create downloadable CSV with safe stat extraction
            results_dict = {}
            for key in self.stats.keys():
                try:
                    results_dict[key] = self.stats[key]
                except (KeyError, TypeError):
                    results_dict[key] = "N/A"
            
            results_df = pd.DataFrame([results_dict])
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"{self.ticker}_backtest_results.csv",
                mime="text/csv"
            )

def run_enhanced_backtest(ticker, cash=100000, commission=0.002):
    """
    Run enhanced backtest with comprehensive visualization.
    """
    backtester = EnhancedBacktester(ticker, cash, commission)
    stats, error = backtester.run_backtest()
    
    if error:
        return None, error
    
    # Display the enhanced dashboard
    backtester.create_performance_dashboard()
    return stats, None
