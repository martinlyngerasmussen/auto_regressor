
## import libraries
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def data_preparation(file_location, lags = 5, splits = 5, train_share = 0.8):
    ###### assumes that the variables are stationary, date column is called "date"
    ###### assumes that the dataset is in a csv file
    ###### assumes that y is the first column in the dataset
    ###### assumes that X is the rest of the columns in the dataset

    # Import the dataset
    dataset = pd.read_csv(file_location)

    ## convert date column to datetime format
    dataset['date'] = pd.to_datetime(dataset['date'], infer_datetime_format= True).dt.date

    ## make date column the index
    df = dataset.copy()
    df.set_index('date', inplace=True)

    ## create splits of the DataFrame, where splits = splits. Each split is a DataFrame. Name each split as df_split_i.
    # Split the DataFrame and store each split in a dictionary
    split_dfs = pd.DataFrame()
    split_dfs = {f"split_{i+1}": df for i, df in enumerate(np.array_split(df, splits))}

    # Creating a dictionary for each split
    splits_dict = {}

    for split in split_dfs:
        # Calculate the split point for 80-20 division
        split_point = int(len(split_dfs[split]) * 0.8)

        # Create and store the training and test sets in a new dictionary for each split
        splits_dict[split] = {
            f"{split}_train": split_dfs[split].iloc[:split_point],
            f"{split}_test": split_dfs[split].iloc[split_point:]
        }

    ## create lags of the variables in each split, divide each split into train and test sets.
    for split, split_df in split_dfs.items():
            # Store the original column names (excluding the date index)
            original_columns = split_df.columns

            # Create lagged features for each split, excluding the index (date)
            for lag in range(1, lags + 1):
                for col in original_columns:
                    split_df[f'{col}_lag{lag}'] = split_df[col].shift(lag)

            # Drop the initial rows with NaN values due to lagging
            split_df.dropna(inplace=True)

            # Calculate the split point for train-test division
            split_point = int(len(split_df) * train_share)

            # Split into training and testing sets
            splits_dict[split] = {
                f"{split}_train": split_df.iloc[:split_point],
                f"{split}_test": split_df.iloc[split_point:]
            }

    return splits_dict

def regression_OLS(file_location, lags, splits, train_share, p_cutoff = 0.05):
    df = data_preparation(file_location, lags, splits, train_share)

    ## write a code that loops over each dataframe in the df dictionary
    # Create empty dictionary to store the results
    results_dict = {}

    for split, split_df in df.items():
        if 'train' in split:  # fit the model only to the training set
            # Store the original column names (excluding the date index)
            original_columns = split_df.columns

            # Create the feature matrix (X) and target vector (y)
            X = split_df[original_columns[1:]]
            y = split_df[original_columns[0]]

            # Perform backward elimination until all p-values are smaller than the cut-off
            while True:
                # Fit the model
                model = sm.OLS(y, X).fit()

                # Store the results for each split
                results_dict[split] = {
                    'model': model,
                    'r2': model.rsquared,
                    'intercept': model.params[0],
                    'coefficients': model.params[1:],
                    'p_values': model.pvalues[0:],
                }

                # Find the variable with the highest p-value
                max_p_value = model.pvalues[1:].max()

                # Check if the highest p-value is greater than the cut-off
                if max_p_value > p_cutoff:
                    # Remove the variable with the highest p-value
                    max_p_value_index = model.pvalues[1:].idxmax()
                    X = X.drop(columns=max_p_value_index)
                else:
                    break

    return results_dict
