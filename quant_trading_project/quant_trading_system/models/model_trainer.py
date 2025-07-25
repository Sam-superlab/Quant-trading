# quant_trading_system/models/model_trainer.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import our existing modules to build the full pipeline
from data_fetcher import DataFetcher
from data_preprocessor import DataPreprocessor
from feature_engineering import FeatureEngineering
from quant_trading_system.utils.config import sanitize_feature_names

class ModelTrainer:
    """
    A module for training and evaluating a predictive machine learning model.

    This class fulfills the first part of Phase 4 of the project plan. It
    takes the feature-rich dataset, defines a target variable, trains a
    classifier, and evaluates its performance.
    """

    def __init__(self, feature_data):
        """
        Initializes the ModelTrainer.

        Args:
            feature_data (pd.DataFrame): A DataFrame containing the features (X)
                                         and the base data needed to create the
                                         target variable (y).
        """
        self.feature_data = feature_data.copy()
        self.model = None

    def create_target_variable(self, horizon=1):
        """
        Creates the target variable (y) for our classification model.

        The goal is to predict the direction of the price move in the future.
        Target = 1 if the price increases `horizon` days from now.
        Target = 0 if the price decreases or stays the same.

        Args:
            horizon (int, optional): The prediction horizon in days. Defaults to 1.
        """
        print(f"Creating target variable for a {horizon}-day future return...")
        # To avoid lookahead bias, we shift the returns backward.
        # The target for today is based on tomorrow's price change.
        self.feature_data['Future_Return'] = self.feature_data['Returns'].shift(-horizon)
        self.feature_data['Target'] = (self.feature_data['Future_Return'] > 0).astype(int)
        
        # Remove the last 'horizon' rows as they will have NaN targets
        self.feature_data.dropna(inplace=True)

    def prepare_data_for_modeling(self):
        """
        Prepares the final feature matrix (X) and target vector (y).
        """
        # The target is what we want to predict.
        y = self.feature_data['Target']

        # The features are all columns except for the ones we can't use.
        # We must drop columns that would "leak" future information into the model.
        cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Future_Return', 'Target']
        X = self.feature_data.drop(columns=cols_to_drop)
        
        print(f"Feature matrix (X) shape: {X.shape}")
        print(f"Target vector (y) shape: {y.shape}")
        
        return X, y

    def train_and_evaluate(self, X, y, test_size=0.2):
        """
        Splits the data, trains a LightGBM model, and evaluates it.

        IMPORTANT: For time-series data, we MUST NOT shuffle the data when splitting.
        This ensures the test set is always in the future relative to the train set.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector.
            test_size (float, optional): The proportion of the dataset to include in the
                                         test split. Defaults to 0.2.
        """
        print("\nSplitting data into training and testing sets (no shuffle)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        # Sanitize feature names for LightGBM compatibility
        X_train.columns = sanitize_feature_names(X_train.columns)
        X_test.columns = sanitize_feature_names(X_test.columns)

        print("Training LightGBM classifier...")
        self.model = lgb.LGBMClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        print("\nEvaluating model performance on the test set...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Down/Same', 'Up'])

        print(f"Model Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        # It's also useful to see the feature importances
        feature_imp = pd.DataFrame(sorted(zip(self.model.feature_importances_, X.columns)), columns=['Value','Feature'])
        print("\nTop 10 Most Important Features:")
        print(feature_imp.sort_values(by="Value", ascending=False).head(10))


# --- Example Usage ---
if __name__ == '__main__':
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
    
    if market_data is not None:
        clean_data = preprocessor.handle_missing_values(market_data, method='ffill')
        
        feature_generator = FeatureEngineering(clean_data)
        feature_generator.add_moving_averages()
        feature_generator.add_momentum_indicators()
        feature_generator.add_volatility_indicators()
        feature_generator.add_lagged_returns()
        
        data_with_features = feature_generator.get_feature_data()

        # --- 2. Train and Evaluate the Model ---
        print("\n" + "="*50)
        print("STEP 2: Model Training and Evaluation")
        print("="*50)
        
        model_pipeline = ModelTrainer(data_with_features)
        model_pipeline.create_target_variable(horizon=1)
        X, y = model_pipeline.prepare_data_for_modeling()
        model_pipeline.train_and_evaluate(X, y)

        print("\nNOTE: This simple train-test split is for initial model validation.")
        print("A robust system, as per our project plan, will require a full Walk-Forward Optimization backtester.")

