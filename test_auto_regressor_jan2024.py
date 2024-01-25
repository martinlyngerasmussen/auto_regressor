import unittest
import pandas as pd
import numpy as np
from auto_regressor import compiler_function

class TestCompilerFunction(unittest.TestCase):
    def setUp(self):
        self.file_location = "/path/to/dataset.csv"
        self.lags = 5
        self.splits = 5
        self.train_share = 0.8

    def test_compiler_function(self):
        # Generate a sample dataset
        dataset = pd.DataFrame({
            'date': pd.date_range(start='1/1/2022', periods=100),
            'y': np.random.randn(100),
            'X1': np.random.randn(100),
            'X2': np.random.randn(100)
        })

        # Save the sample dataset to a CSV file
        dataset.to_csv(self.file_location, index=False)

        # Call the compiler_function
        split_dfs, regression_models, predictions, oos_summary_stats_dict = compiler_function(self.file_location, self.lags, self.splits, self.train_share)

        # Assert the expected number of splits in split_dfs
        self.assertEqual(len(split_dfs), self.splits)

        # Assert the expected number of regression models
        self.assertEqual(len(regression_models), self.splits)

        # Assert the expected number of predictions
        self.assertEqual(len(predictions), self.splits + 1)  # +1 for the full sample prediction

        # Assert the expected number of oos summary stats
        self.assertEqual(len(oos_summary_stats_dict), self.splits)

    def tearDown(self):
        # Remove the sample dataset file
        os.remove(self.file_location)

if __name__ == '__main__':
    unittest.main()
class TestLoadDF(unittest.TestCase):
    def setUp(self):
        self.file_location = "/path/to/dataset.csv"

    def test_load_df_file_not_found(self):
        # Test if the file does not exist
        with self.assertRaises(FileNotFoundError):
            load_df(self.file_location)

    def test_load_df_file_not_readable(self):
        # Test if the file is not readable
        # Create a file with no read permissions
        open(self.file_location, 'a').close()
        os.chmod(self.file_location, 0o000)

        with self.assertRaises(PermissionError):
            load_df(self.file_location)

        # Restore the file permissions
        os.chmod(self.file_location, 0o644)

    def test_load_df_invalid_file_format(self):
        # Test if the file format is invalid
        self.file_location = "/path/to/dataset.txt"

        with self.assertRaises(ValueError):
            load_df(self.file_location)

    def test_load_df_datetime_column_found(self):
        # Test if a datetime column is found
        dataset = pd.DataFrame({
            'date': pd.date_range(start='1/1/2022', periods=100),
            'y': np.random.randn(100),
            'X1': np.random.randn(100),
            'X2': np.random.randn(100)
        })

        dataset.to_csv(self.file_location, index=False)

        df = load_df(self.file_location)

        self.assertTrue('date' in df.columns)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))

    def test_load_df_datetime_column_not_found(self):
        # Test if a datetime column is not found
        dataset = pd.DataFrame({
            'not_date': pd.date_range(start='1/1/2022', periods=100),
            'y': np.random.randn(100),
            'X1': np.random.randn(100),
            'X2': np.random.randn(100)
        })

        dataset.to_csv(self.file_location, index=False)

        with self.assertRaises(ValueError):
            load_df(self.file_location)

    def tearDown(self):
        # Remove the sample dataset file
        os.remove(self.file_location)
