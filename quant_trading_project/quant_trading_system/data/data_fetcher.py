import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime

class DataFetcher:
    """
    A module for fetching various types of financial data.

    This class serves as the primary interface for acquiring data required
    for the quantitative trading system, fulfilling Deliverable 2.1 of the project plan.
    It handles fetching of:
    1. Market Data (OHLCV) from Yahoo Finance.
    2. Fundamental Data from Finnhub.
    3. Alternative Data (News Sentiment) from Finnhub.
    """

    def __init__(self, finnhub_api_key=None):
        """
        Initializes the DataFetcher.

        Args:
            finnhub_api_key (str, optional): Your API key for Finnhub. 
                                             It's recommended to set this as an
                                             environment variable 'FINNHUB_API_KEY'.
                                             Defaults to None.
        """
        # The script will use the provided key, but prioritizing an environment variable is best practice.
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY') or finnhub_api_key
        if not self.finnhub_api_key:
            print("Warning: Finnhub API key not provided. Fundamental and alternative data fetching will not work.")
            print("Get a free key from https://finnhub.io and set it as an environment variable 'FINNHUB_API_KEY'.")
        self._finnhub_base_url = "https://finnhub.io/api/v1"

    def _make_finnhub_request(self, endpoint, params):
        """Helper function to make requests to the Finnhub API."""
        if not self.finnhub_api_key:
            print("Error: Finnhub API key is required to make this request.")
            return None
        
        params['token'] = self.finnhub_api_key
        try:
            response = requests.get(f"{self._finnhub_base_url}/{endpoint}", params=params)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Finnhub endpoint '{endpoint}': {e}")
            return None

    def get_market_data(self, tickers, start_date, end_date, interval="1d"):
        """
        Fetches historical market (OHLCV) data for a list of tickers.

        Args:
            tickers (list or str): A list of stock tickers or a single ticker string.
            start_date (str): The start date for the data in 'YYYY-MM-DD' format.
            end_date (str): The end date for the data in 'YYYY-MM-DD' format.
            interval (str, optional): Data interval. Valid intervals: 1d, 5d, 1wk, 1mo, 3mo. Defaults to "1d".

        Returns:
            pandas.DataFrame: A DataFrame containing the OHLCV data, or None if an error occurs.
        """
        print(f"Fetching market data for {tickers} from {start_date} to {end_date}...")
        try:
            # Using auto_adjust=True simplifies data by adjusting OHLC for splits/dividends
            data = yf.download(tickers, start=start_date, end=end_date, interval=interval, auto_adjust=True)
            if data.empty:
                print(f"Warning: No market data found for tickers {tickers} in the given date range.")
                return None
            print("Market data fetched successfully.")
            return data
        except Exception as e:
            print(f"An error occurred while fetching market data: {e}")
            return None

    def get_fundamental_data(self, ticker, statement_type='ic', freq='quarterly'):
        """
        Fetches fundamental data (financial statements) for a single ticker.

        Args:
            ticker (str): The stock ticker (e.g., 'AAPL').
            statement_type (str, optional): The type of statement. 'ic' (Income Statement), 
                                            'bs' (Balance Sheet), 'cf' (Cash Flow). Defaults to 'ic'.
            freq (str, optional): Frequency of the report. 'quarterly' or 'annual'. Defaults to 'quarterly'.

        Returns:
            pandas.DataFrame: A DataFrame containing the financial statement data, or None if an error occurs.
        """
        print(f"Fetching {freq} {statement_type} for {ticker}...")
        params = {'symbol': ticker, 'statement': statement_type, 'freq': freq}
        data = self._make_finnhub_request('stock/financials-reported', params)
        
        if data and data.get('data'):
            df = pd.DataFrame(data['data'])
            print("Fundamental data fetched successfully.")
            return df
        else:
            print(f"Warning: No fundamental data found for {ticker}.")
            return None

    def get_alternative_news_sentiment(self, ticker):
        """
        Fetches basic news sentiment data for a single ticker.

        Args:
            ticker (str): The stock ticker (e.g., 'TSLA').

        Returns:
            dict: A dictionary containing news sentiment data, or None if an error occurs.
        """
        print(f"Fetching news sentiment for {ticker}...")
        params = {'symbol': ticker}
        data = self._make_finnhub_request('news-sentiment', params)
        
        if data:
            print("News sentiment data fetched successfully.")
            return data
        else:
            print(f"Warning: No news sentiment data found for {ticker}.")
            return None

# --- Example Usage ---
if __name__ == '__main__':
    # The user-provided API key is used here for demonstration.
    # In a production environment, it is strongly recommended to set this
    # as an environment variable for security.
    # Example: export FINNHUB_API_KEY='your_key_here'
    user_api_key = 'd21bk3pr01qkdupiodggd21bk3pr01qkdupiodh0'
    
    fetcher = DataFetcher(finnhub_api_key=user_api_key)

    # --- 1. Fetch Market Data ---
    print("\n" + "="*50)
    print("1. DEMO: Fetching Market Data for AAPL and MSFT")
    print("="*50)
    market_data = fetcher.get_market_data(['AAPL', 'MSFT'], start_date='2023-01-01', end_date='2023-12-31')
    if market_data is not None:
        print("Sample of Market Data (AAPL):")
        # For multi-ticker downloads, yfinance creates a multi-level column index.
        # We can select a single ticker for cleaner display.
        print(market_data.xs('AAPL', level='Ticker', axis=1).head())


    # --- 2. Fetch Fundamental Data ---
    # This now works with the provided API key.
    if fetcher.finnhub_api_key:
        print("\n" + "="*50)
        print("2. DEMO: Fetching Fundamental Data for AAPL (Income Statement)")
        print("="*50)
        income_statement = fetcher.get_fundamental_data('AAPL', statement_type='ic', freq='quarterly')
        if income_statement is not None:
            print("Sample of Quarterly Income Statement Data for AAPL:")
            # Displaying a subset of columns for clarity
            if 'report' in income_statement.columns and 'fiscalDate' in income_statement.columns:
                 # Extracting 'revenue' and 'netIncome' for demonstration
                simplified_report = income_statement.apply(lambda row: {
                    item['label']: item['value'] for item in row['report']
                    if item['concept'] in ['us-gaap_Revenues', 'us-gaap_NetIncomeLoss']
                }, axis=1)
                
                display_df = pd.concat([income_statement['fiscalDate'], simplified_report.apply(pd.Series)], axis=1)
                print(display_df.head())
            else:
                print(income_statement.head())

    # --- 3. Fetch Alternative Data ---
    # This now works with the provided API key.
    if fetcher.finnhub_api_key:
        print("\n" + "="*50)
        print("3. DEMO: Fetching Alternative Data for TSLA (News Sentiment)")
        print("="*50)
        news_sentiment = fetcher.get_alternative_news_sentiment('TSLA')
        if news_sentiment:
            print("Sample of News Sentiment Data for TSLA:")
            # Print some high-level sentiment metrics
            print(f"Buzz (weekly articles): {news_sentiment.get('buzz', {}).get('weeklyAverage')}")
            print(f"Company News Score: {news_sentiment.get('sentiment', {}).get('companyNewsScore')}")
            print(f"Sector Average Bullish %: {news_sentiment.get('sectorAverageBullishPercent')}") 