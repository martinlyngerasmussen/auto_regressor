import unittest
import pandas as pd
import numpy as np
from auto_regressor import full_df, data_preparation_splits, regression_OLS

class TestAutoRegressor(unittest.TestCase):
    def setUp(self):
        self.file_location = "/path/to/dataset.csv"
        self.lags = 5
        self.splits = 5
        self.train_share = 0.8

    def test_full_df(self):
        # Generate a sample dataset
        dataset = pd.DataFrame({
            'date': pd.date_range(start='1/1/2022', periods=100),
            'y': np.random.randn(100),
            'X1': np.random.randn(100),
            'X2': np.random.randn(100)
        })

        # Save the sample dataset to a CSV file
        dataset.to_csv(self.file_location, index=False)

        # Call the full_df function
        df = full_df(self.file_location, self.lags)

        # Assert the expected number of columns in the dataframe
        self.assertEqual(len(df.columns), len(dataset.columns) * self.lags + 1)  # +1 for the constant column

    def test_data_preparation_splits(self):
        # Generate a sample dataset
        dataset = pd.DataFrame({
            'date': pd.date_range(start='1/1/2022', periods=100),
            'y': np.random.randn(100),
            'X1': np.random.randn(100),
            'X2': np.random.randn(100)
        })

        # Save the sample dataset to a CSV file
        dataset.to_csv(self.file_location, index=False)

        # Call the data_preparation_splits function
        splits_dict = data_preparation_splits(self.file_location, self.lags, self.splits, self.train_share)

        # Assert the expected number of splits in the dictionary
        self.assertEqual(len(splits_dict), self.splits)

        # Assert the expected number of train-test sets in each split
        for split in splits_dict:
            self.assertEqual(len(splits_dict[split]), 2)

    def test_regression_OLS(self):
        # Generate a sample dataset
        dataset = pd.DataFrame({
            'date': pd.date_range(start='1/1/2022', periods=100),
            'y': np.random.randn(100),
            'X1': np.random.randn(100),
            'X2': np.random.randn(100)
        })

        # Save the sample dataset to a CSV file
        dataset.to_csv(self.file_location, index=False)

        # Call the regression_OLS function
        results_dict = regression_OLS(self.file_location, self.lags, self.splits, self.train_share)

        # Assert the expected keys in the results dictionary
        self.assertIn('split_1_summary', results_dict)
        self.assertIn('split_2_summary', results_dict)
        self.assertIn('split_3_summary', results_dict)
        self.assertIn('split_4_summary', results_dict)
        self.assertIn('split_5_summary', results_dict)

    def tearDown(self):
        # Remove the sample dataset file
        os.remove(self.file_location)

if __name__ == '__main__':
    unittest.main()
