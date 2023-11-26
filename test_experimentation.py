import pandas as pd
import numpy as np
import unittest

from experimentation import data_preparation

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        self.file_location = "/path/to/dataset.csv"
        self.lags = 5
        self.splits = 5
        self.train_share = 0.8

    def test_data_preparation(self):
        # Generate a sample dataset
        dataset = pd.DataFrame({
            'date': pd.date_range(start='1/1/2022', periods=100),
            'y': np.random.randn(100),
            'X1': np.random.randn(100),
            'X2': np.random.randn(100)
        })

        # Save the sample dataset to a CSV file
        dataset.to_csv(self.file_location, index=False)

        # Call the data_preparation function
        splits_dict = data_preparation(self.file_location, self.lags, self.splits, self.train_share)

        # Assert the expected number of splits
        self.assertEqual(len(splits_dict), self.splits)

        # Assert the expected number of train-test sets in each split
        for split in splits_dict:
            self.assertEqual(len(splits_dict[split]), 2)

        # Assert the expected number of lagged features in each split
        for split in splits_dict:
            for key in splits_dict[split]:
                if "train" in key:
                    self.assertEqual(len(splits_dict[split][key].columns), len(dataset.columns) * self.lags)

        # Assert the expected number of rows in each train-test set
        for split in splits_dict:
            for key in splits_dict[split]:
                self.assertEqual(len(splits_dict[split][key]), int(len(dataset) * self.train_share))

    def tearDown(self):
        # Remove the sample dataset file
        os.remove(self.file_location)

if __name__ == '__main__':
    unittest.main()
