# quant_trading_system/data/data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    A module for cleaning, transforming, and preparing financial data for analysis.
    """

    def handle_missing_values(self, df, method='ffill'):
        """
        Handles missing values in a DataFrame.
        """
        if method == 'ffill':
            return df.ffill()
        elif method == 'bfill':
            return df.bfill()
        elif method == 'drop':
            return df.dropna()
        elif method == 'mean':
            return df.fillna(df.mean())
        else:
            raise ValueError("Invalid method. Choose from 'ffill', 'bfill', 'drop', 'mean'.")

    def normalize_data(self, df, columns):
        """
        Normalizes specified columns in a DataFrame using StandardScaler.
        """
        scaler = StandardScaler()
        df_copy = df.copy()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        return df_copy

    def align_market_and_fundamental_data(self, market_df, fundamental_df, fundamental_cols):
        """
        Aligns daily market data with lower-frequency (e.g., quarterly) fundamental data.
        This is a critical step to avoid lookahead bias.
        """
        print("Aligning daily market data with fundamental data...")
        
        # Check if fundamental data is valid
        if fundamental_df is None or fundamental_df.empty:
            print("Warning: No fundamental data available. Using market data only.")
            # Return market data with standardized column names
            return self._standardize_market_columns(market_df)
        
        fundamental_df_copy = fundamental_df.copy()
        print("fundamental_df_copy columns:", fundamental_df_copy.columns)
        print("fundamental_df_copy head:", fundamental_df_copy.head())
        
        # Use 'fiscalDate' if present, otherwise 'endDate'
        date_col = None
        if 'fiscalDate' in fundamental_df_copy.columns:
            date_col = 'fiscalDate'
        elif 'endDate' in fundamental_df_copy.columns:
            date_col = 'endDate'
        else:
            print("Warning: No valid date column found in fundamental data. Using market data only.")
            return self._standardize_market_columns(market_df)
        
        try:
            fundamental_df_copy['date'] = pd.to_datetime(fundamental_df_copy[date_col])
            fundamental_df_copy = fundamental_df_copy.set_index('date').sort_index()
        except Exception as e:
            print(f"Warning: Error processing fundamental data dates: {e}. Using market data only.")
            return self._standardize_market_columns(market_df)
        
        # Helper to extract value from the correct statement type
        def extract_value(report, statement_type, concept):
            if not isinstance(report, dict):
                return np.nan
            items = report.get(statement_type, [])
            for item in items:
                if item.get('concept') == concept:
                    return item.get('value', np.nan)
            return np.nan

        # Try different statement types for different concepts
        concept_to_statement = {
            'QuarterlyRevenue': 'is',  # Income statement
            'QuarterlyNetIncome': 'is',  # Income statement
            'TotalAssets': 'bs',  # Balance sheet
            'TotalLiabilities': 'bs',  # Balance sheet
            'CashAndCashEquivalents': 'bs',  # Balance sheet
        }
        
        for col in fundamental_cols:
            statement_type = concept_to_statement.get(col, 'is')  # Default to income statement
            fundamental_df_copy[col] = fundamental_df_copy['report'].apply(
                lambda report: extract_value(report, statement_type, col)
            )
            
        # Debug: Print extracted values
        print("Extracted fundamental values:")
        for col in fundamental_cols:
            if col in fundamental_df_copy.columns:
                non_null_count = fundamental_df_copy[col].notna().sum()
                print(f"  {col}: {non_null_count} non-null values out of {len(fundamental_df_copy)}")
                if non_null_count > 0:
                    print(f"    Sample values: {fundamental_df_copy[col].dropna().head(3).tolist()}")
        
        fundamental_subset = fundamental_df_copy[fundamental_cols]
        
        # Check if fundamental subset has valid data
        if fundamental_subset.empty or fundamental_subset.isna().all().all():
            print("Warning: No valid fundamental data extracted. Using market data only.")
            return self._standardize_market_columns(market_df)

        # Use pandas merge_asof for precise point-in-time alignment
        market_df_sorted = market_df.sort_index()
        fundamental_subset_sorted = fundamental_subset.sort_index()

        # Ensure both DataFrames have proper date indices
        # If market_df already has a date index, we're good
        # If not, we need to find the date column and set it as index
        if not isinstance(market_df_sorted.index, pd.DatetimeIndex):
            # Reset index to get all columns
            market_df_sorted = market_df_sorted.reset_index()
            
            # Flatten MultiIndex columns if present
            if isinstance(market_df_sorted.columns, pd.MultiIndex):
                market_df_sorted.columns = ['_'.join([str(c) for c in col if c]) for col in market_df_sorted.columns]
            
            # Find date column (case-insensitive)
            market_date_col = None
            for col in market_df_sorted.columns:
                if str(col).lower() == 'date':
                    market_date_col = col
                    break
            
            if market_date_col:
                market_df_sorted = market_df_sorted.set_index(market_date_col)
            else:
                raise KeyError(f"Date column not found in market_df_sorted. Available columns: {list(market_df_sorted.columns)}")

        # Ensure fundamental data has proper date index
        if not isinstance(fundamental_subset_sorted.index, pd.DatetimeIndex):
            # Reset index to get all columns
            fundamental_subset_sorted = fundamental_subset_sorted.reset_index()
            
            # Flatten MultiIndex columns if present
            if isinstance(fundamental_subset_sorted.columns, pd.MultiIndex):
                fundamental_subset_sorted.columns = ['_'.join([str(c) for c in col if c]) for col in fundamental_subset_sorted.columns]
            
            # Find date column (case-insensitive)
            fundamental_date_col = None
            for col in fundamental_subset_sorted.columns:
                if str(col).lower() == 'date':
                    fundamental_date_col = col
                    break
            
            if fundamental_date_col:
                fundamental_subset_sorted = fundamental_subset_sorted.set_index(fundamental_date_col)
            else:
                raise KeyError(f"Date column not found in fundamental_subset_sorted. Available columns: {list(fundamental_subset_sorted.columns)}")

        print(f"After fixing - market_df_sorted index: {market_df_sorted.index}")
        print(f"After fixing - fundamental_subset_sorted index: {fundamental_subset_sorted.index}")

        # Debug: Check index types and levels
        print(f"Market DataFrame index type: {type(market_df_sorted.index)}")
        print(f"Market DataFrame index levels: {market_df_sorted.index.nlevels if hasattr(market_df_sorted.index, 'nlevels') else 'N/A'}")
        print(f"Fundamental DataFrame index type: {type(fundamental_subset_sorted.index)}")
        print(f"Fundamental DataFrame index levels: {fundamental_subset_sorted.index.nlevels if hasattr(fundamental_subset_sorted.index, 'nlevels') else 'N/A'}")

        # Force both DataFrames to have single-level indices by completely resetting
        print("Forcing single-level indices...")
        
        # Reset market DataFrame completely and set date as index
        market_df_sorted = market_df_sorted.reset_index()
        if isinstance(market_df_sorted.columns, pd.MultiIndex):
            market_df_sorted.columns = ['_'.join([str(c) for c in col if c]) for col in market_df_sorted.columns]
        
        # Find date column in market data
        market_date_col = None
        for col in market_df_sorted.columns:
            if str(col).lower() == 'date':
                market_date_col = col
                break
        
        if market_date_col:
            market_df_sorted = market_df_sorted.set_index(market_date_col)
        else:
            raise KeyError(f"Date column not found in market_df_sorted. Available columns: {list(market_df_sorted.columns)}")

        # Reset fundamental DataFrame completely and set date as index
        fundamental_subset_sorted = fundamental_subset_sorted.reset_index()
        if isinstance(fundamental_subset_sorted.columns, pd.MultiIndex):
            fundamental_subset_sorted.columns = ['_'.join([str(c) for c in col if c]) for col in fundamental_subset_sorted.columns]
        
        # Find date column in fundamental data
        fundamental_date_col = None
        for col in fundamental_subset_sorted.columns:
            if str(col).lower() == 'date':
                fundamental_date_col = col
                break
        
        if fundamental_date_col:
            fundamental_subset_sorted = fundamental_subset_sorted.set_index(fundamental_date_col)
        else:
            raise KeyError(f"Date column not found in fundamental_subset_sorted. Available columns: {list(fundamental_subset_sorted.columns)}")

        print(f"Final - market_df_sorted index: {market_df_sorted.index}")
        print(f"Final - fundamental_subset_sorted index: {fundamental_subset_sorted.index}")
        print(f"Final - market_df_sorted index type: {type(market_df_sorted.index)}")
        print(f"Final - fundamental_subset_sorted index type: {type(fundamental_subset_sorted.index)}")

        # Perform backward search merge
        try:
            aligned_data = pd.merge_asof(
                market_df_sorted.sort_index(),
                fundamental_subset_sorted.sort_index(),
                left_index=True, right_index=True, direction='backward'
            )
            print("Data alignment complete.")
        except Exception as e:
            print(f"Warning: Error during data alignment: {e}. Using market data only.")
            return self._standardize_market_columns(market_df)
        
        # Clean any NaT values from the index and ensure proper datetime index
        print(f"Before cleaning - aligned_data index: {aligned_data.index}")
        print(f"Before cleaning - aligned_data shape: {aligned_data.shape}")
        
        # Remove rows with NaT index values
        aligned_data = aligned_data[aligned_data.index.notna()]
        
        # Ensure the index is properly formatted as datetime
        if not isinstance(aligned_data.index, pd.DatetimeIndex):
            try:
                aligned_data.index = pd.to_datetime(aligned_data.index)
            except Exception as e:
                print(f"Error converting index to datetime: {e}")
                # If conversion fails, try to reset and set a proper date column
                aligned_data = aligned_data.reset_index()
                # Find the date column
                date_col = None
                for col in aligned_data.columns:
                    if str(col).lower() == 'date':
                        date_col = col
                        break
                if date_col:
                    aligned_data = aligned_data.set_index(date_col)
                    aligned_data.index = pd.to_datetime(aligned_data.index)
        
        print(f"After cleaning - aligned_data index: {aligned_data.index}")
        print(f"After cleaning - aligned_data shape: {aligned_data.shape}")
        
        if aligned_data.empty:
            raise ValueError("No valid data remaining after alignment. Check your date ranges and data sources.")
        
        # Standardize column names for market data
        aligned_data = self._standardize_market_columns(aligned_data)
        
        return aligned_data

    def _standardize_market_columns(self, df):
        """
        Standardizes market data column names by removing ticker suffixes.
        Converts columns like 'Open_NVDA' to 'Open', 'Close_NVDA' to 'Close', etc.
        """
        print("Standardizing market data column names...")
        
        # Define the mapping for standard market data columns
        column_mapping = {}
        
        for col in df.columns:
            # Check if column ends with a ticker suffix (e.g., _NVDA, _AAPL)
            if '_' in col:
                base_name = col.split('_')[0]
                # Only map if it's a standard market data column
                if base_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    column_mapping[col] = base_name
        
        # Rename columns if any mapping was found
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"Renamed columns: {column_mapping}")
        
        return df
