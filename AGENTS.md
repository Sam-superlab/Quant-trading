# AGENTS.md - AI Agent Guide for Quantitative Trading System

## Project Overview

This is a comprehensive quantitative trading system built in Python that combines machine learning, financial data analysis, and automated trading execution. The system uses LightGBM for predictive modeling and provides both traditional backtesting and enhanced visualization capabilities through a Streamlit dashboard.

### Core Purpose
- Fetch financial market and fundamental data
- Engineer predictive features using technical and fundamental analysis
- Train machine learning models for trading signal generation
- Execute automated trading strategies
- Provide comprehensive backtesting and performance analysis

## Architecture Overview

```
quant_trading_project/
├── quant_trading_system/          # Core trading system
│   ├── data/                      # Data fetching and preprocessing
│   ├── features/                  # Feature engineering
│   ├── models/                    # ML models and backtesting
│   ├── strategies/                # Trading strategies
│   ├── execution/                 # Trade execution
│   └── utils/                     # Configuration and utilities
├── scripts/                       # Streamlit dashboard
├── tests/                         # Test files (ignored in git)
└── notebooks/                     # Jupyter notebooks
```

## Key Components

### 1. Data Layer (`quant_trading_system/data/`)

**Primary Files:**
- `data_fetcher.py`: Fetches market data (yfinance) and fundamental data (Finnhub API)
- `data_preprocessor.py`: Cleans, standardizes, and preprocesses financial data

**Critical Notes:**
- Uses both yfinance and Finnhub APIs for comprehensive data coverage
- Handles missing data with forward-fill (`ffill()`) and backward-fill (`bfill()`)
- Implements `concept_to_statement` mapping for fundamental data extraction
- Column sanitization required for LightGBM compatibility

### 2. Feature Engineering (`quant_trading_system/features/`)

**Primary Files:**
- `feature_engineering.py`: Creates technical indicators and fundamental features

**Key Features:**
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Fundamental ratios and metrics
- Rolling window calculations
- Feature standardization and normalization

### 3. Models (`quant_trading_system/models/`)

**Primary Files:**
- `lightgbm_classifier.py`: LightGBM model for signal classification
- `backtester.py`: Traditional backtesting engine
- `enhanced_backtester.py`: Advanced backtesting with comprehensive visualizations

**Critical Implementation Details:**
- LightGBM requires sanitized column names (alphanumeric + underscores only)
- Use `verbose=-1` to suppress LightGBM warnings
- Enhanced backtester provides interactive Plotly visualizations
- Handles edge cases like empty returns arrays for risk calculations

### 4. Strategies (`quant_trading_system/strategies/`)

**Primary Files:**
- `basic.py`: Basic trading strategy implementation

### 5. Execution (`quant_trading_system/execution/`)

**Primary Files:**
- `execution_handler.py`: Handles trade execution via Alpaca API

**Configuration:**
- Uses Alpaca paper trading environment
- API credentials managed through environment variables and config.py fallback

### 6. Dashboard (`scripts/`)

**Primary Files:**
- `monitoring_dashboard.py`: Streamlit web interface

**Key Features:**
- Real-time monitoring and control
- Enhanced backtesting visualization
- Data compatibility handling for Streamlit's Arrow backend

## Critical Technical Requirements

### 1. Dependencies Management
- Use `requirements.txt` for package management
- Key dependencies: pandas, numpy, lightgbm, streamlit, plotly, backtesting, yfinance, finnhub-python
- Always include version constraints for stability

### 2. Data Handling Best Practices

**Pandas Compatibility:**
- Use `.loc` for DataFrame assignments to avoid `SettingWithCopyWarning`
- Use `df.ffill()` and `df.bfill()` instead of deprecated `fillna(method=)`
- Handle Timedelta objects for Streamlit compatibility

**Streamlit Data Display:**
- Convert `timedelta64[ns]` to total seconds for Arrow backend compatibility
- Use `clean_dataframe_for_streamlit()` helper function
- Handle both DataFrame and Series objects

### 3. LightGBM Model Requirements

**Column Sanitization:**
```python
def sanitize_feature_names(columns):
    """Remove quotes and non-alphanumeric characters except underscores"""
    sanitized = []
    for col in columns:
        col_str = str(col).strip("'\"")  # Remove quotes
        new_col = re.sub(r'[^0-9a-zA-Z]+', '_', col_str)
        new_col = new_col.strip('_')
        if not new_col or new_col[0].isdigit():
            new_col = 'f_' + new_col
        sanitized.append(new_col)
    return sanitized
```

**Model Configuration:**
- Use `verbose=-1` to suppress training warnings
- Validate training data before fitting
- Handle feature name mismatches between training and prediction

### 4. Enhanced Backtesting

**Critical Error Handling:**
- Check `if self.stats is None:` instead of `if not self.stats:` for _Stats objects
- Use `_get_stat_safely()` helper for robust stat access
- Handle empty returns arrays in risk calculations:
```python
if len(returns) > 0:
    var_value = np.percentile(returns, 5) * 100
else:
    var_value = 0.0
```

### 5. API Configuration

**Environment Variables:**
- Finnhub API: `FINNHUB_API_KEY`
- Alpaca API: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `APCA_API_BASE_URL`
- Fallback to hardcoded values in `config.py` if environment variables not set

## Common Issues and Solutions

### 1. LightGBM Training Failures
**Symptom:** "No further splits with positive gain"
**Solution:** 
- Debug column names for extra quotes or special characters
- Validate training data has sufficient samples and features
- Use column sanitization function

### 2. Streamlit Display Errors
**Symptom:** `pyarrow.lib.ArrowTypeError`
**Solution:**
- Use `clean_dataframe_for_streamlit()` before displaying data
- Convert Timedelta objects to numeric values

### 3. Enhanced Backtesting Errors
**Symptom:** KeyError for risk metrics or IndexError for empty arrays
**Solution:**
- Use `_get_stat_safely()` for all stat access
- Implement custom calculations for missing metrics
- Check array lengths before calling numpy functions

### 4. API Connection Issues
**Symptom:** "API keys not found"
**Solution:**
- Verify environment variables are set
- Check config.py fallback values
- Implement demo mode for development

## Development Workflow

### 1. Adding New Features
1. Update `requirements.txt` if new dependencies are needed
2. Follow existing code structure and naming conventions
3. Add appropriate error handling and logging
4. Test with both synthetic and real data
5. Update documentation

### 2. Modifying Existing Code
1. Understand data flow and dependencies
2. Maintain backward compatibility where possible
3. Test thoroughly with existing functionality
4. Update related documentation

### 3. Debugging
1. Use provided debug scripts (`debug_*.py`) as templates
2. Add logging at critical points
3. Validate data shapes and types at boundaries
4. Check for edge cases (empty data, missing keys, etc.)

## File Modification Guidelines

### DO:
- Use consistent error handling patterns
- Maintain existing code style and structure
- Add comprehensive docstrings for new functions
- Test changes with real data flows
- Update configuration files when adding new features

### DON'T:
- Modify core data structures without understanding downstream impact
- Remove error handling or safety checks
- Use deprecated pandas methods
- Hardcode values that should be configurable
- Skip testing with edge cases

## Testing Strategy

### Test Files (Git Ignored)
- `test_api_connection.py`: Verify API connectivity
- `test_alpaca_connection.py`: Test trading API
- `test_enhanced_backtest.py`: Validate backtesting functionality
- `debug_*.py`: Diagnostic scripts for troubleshooting

### Manual Testing Checklist
1. Data fetching for multiple tickers
2. Feature engineering pipeline
3. Model training and prediction
4. Backtesting execution
5. Dashboard functionality
6. API connections

## Configuration Management

### Key Configuration Files
- `quant_trading_system/utils/config.py`: Central configuration
- `requirements.txt`: Package dependencies
- `.env`: Environment variables (not tracked)
- `.gitignore`: Excludes test files and sensitive data

### Environment Setup
```bash
pip install -r requirements.txt
# Set environment variables or update config.py
streamlit run scripts/monitoring_dashboard.py
```

## Performance Considerations

### Data Processing
- Use vectorized operations with pandas/numpy
- Implement data caching where appropriate
- Monitor memory usage with large datasets

### Model Training
- Use appropriate sample sizes for training
- Implement incremental learning where possible
- Monitor training time and resource usage

### Dashboard Responsiveness
- Use Streamlit caching for expensive operations
- Optimize data queries and processing
- Consider asynchronous operations for long-running tasks

## Security Considerations

### API Keys
- Never commit API keys to version control
- Use environment variables or secure configuration
- Implement demo/sandbox modes for development

### Data Privacy
- Handle financial data according to regulations
- Implement appropriate access controls
- Consider data retention policies

## Maintenance and Updates

### Regular Tasks
- Update dependencies periodically
- Monitor API rate limits and usage
- Review and update trading strategies
- Backup important data and models

### Version Control
- Use meaningful commit messages
- Tag releases appropriately
- Document breaking changes
- Maintain changelog for major updates

## Support and Resources

### Documentation
- `BUG_FIXES_SUMMARY.md`: Historical bug fixes and solutions
- `ENHANCED_BACKTESTING_FEATURES.md`: Guide to visualization features
- Code comments and docstrings throughout

### External APIs
- [Finnhub API Documentation](https://finnhub.io/docs/api)
- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)

### Libraries
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Backtesting.py Documentation](https://kernc.github.io/backtesting.py/)
- [Plotly Documentation](https://plotly.com/python/)

---

**Last Updated:** Generated automatically for AI agent guidance
**Project Version:** Enhanced Backtesting with Full Visualization Suite
**Maintainer:** AI-Assisted Development Workflow 