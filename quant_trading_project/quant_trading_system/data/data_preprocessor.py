class DataPreprocessor:
    def handle_missing_values(self, data, method='ffill'):
        """
        Handles missing values in the data using the specified method.
        Args:
            data (pd.DataFrame): The input data.
            method (str): The method to use for filling missing values ('ffill', 'bfill', etc.).
        Returns:
            pd.DataFrame: The data with missing values handled.
        """
        return data.fillna(method=method) 