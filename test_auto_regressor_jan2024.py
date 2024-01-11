import unittest
import pandas as pd
import numpy as np
from auto_regressor_jan2024 import compiler_function

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
