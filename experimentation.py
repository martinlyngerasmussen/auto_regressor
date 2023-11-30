
##### To do:
# 1. Format output of regression_OLS function
# 2. Make sure the dates for each split is clear
# 3. Save the in-sample and oos results in a csv file across the different splits
# 4. Visually summarize the results for the different splits
# 5. Model averaging: what to do? Average (simple or weighted) as well as min, max, median, etc.?


# # import libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from prettytable import PrettyTable
from datetime import datetime



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
        split_point = int(len(split_dfs[split]) * train_share)

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
    data = data_preparation(file_location, lags, splits, train_share)

    ## write a code that loops over each dataframe in the df dictionary
    # Create empty dictionary to store the results
    results_dict = {}

    ## import only y from the dataset
    dataset = pd.read_csv(file_location)
    dataset['date'] = pd.to_datetime(dataset['date'], infer_datetime_format= True).dt.date
    df = dataset.copy()
    df.set_index('date', inplace=True)
    oos_predictions = pd.DataFrame()
    oos_predictions["y"] = df.iloc[:, 0]



    for split in data:
        ## split is the name of the split (contains all the splits=splits dataframe), split_df is the dataframe (each split contains two split_dfs, and we only want to look at the "train" ones):
        split_df = data[split][f"{split}_train"]

        ## fit the OLS model on the split_df
        # Separate the target variable and the features
        y = split_df.iloc[:, 0]
        X = split_df.iloc[:, 1:]

        # Loop until all VIFs are smaller than the cut-off value
        vif_cut_off = 5
        vif_max = 10

        while vif_max > vif_cut_off:
            # Create a DataFrame with the features and their respective VIFs
            vif = pd.DataFrame()
            vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif["features"] = X.columns

            # Find the variable with the highest VIF
            vif_max = max(vif["VIF Factor"])
            feature_max_vif = vif[vif["VIF Factor"] == vif_max]["features"]

            # Remove the feature with the highest VIF
            X = X.drop(feature_max_vif, axis=1)

        # Add a constant to the features
        X = sm.add_constant(X)

        # Fit the OLS model
        model = sm.OLS(y, X).fit(cov_type = "HAC", cov_kwds={'maxlags': 4})


        # Perform backward elimination until all p-values are smaller than the cut-off value
        # Initialize the loop
        p_max = 1
        while p_max > p_cutoff:
            # Find the variable with the highest p-value
            p = model.pvalues
            p_max = max(p)
            feature_max_p = p.idxmax()

            # Remove the feature with the highest p-value
            X = X.drop(feature_max_p, axis=1)

            # Fit the model without the feature with the highest p-value
            model = sm.OLS(y, X).fit(cov_type = "HAC", cov_kwds={'maxlags': 4})

        ## store the results of the final model in a dictionary
        # results_dict[f'{split}_summary'] = model.summary()

        final_model = model

        ## add predictions to oos_predictions dataframe
        oos_predictions[f'{split}_y_fitted'] = final_model.predict(X)

        ## predict the values of the test set using the estimated model
        # Store the test set in a new variable
        test_set = data[split][f"{split}_test"]

        # Separate the target variable and the features
        y_test = test_set.iloc[:, 0]
        X_test = test_set.iloc[:, 1:]

        ## exclude columns from test_set that are not in train_set
        X_test = X_test[X.columns.intersection(X_test.columns)]


        # Add a constant to the features
        if "const" in X.columns:
            X_test = sm.add_constant(X_test)

        # Predict the target variable
        y_pred = final_model.predict(X_test)

        ## add to oos_predictions dataframe
        oos_predictions[f'{split}_y_pred'] = y_pred

        ## calculate the following for y_test and y_pred: R2, MAE, MSE, RMSE, MAPE
        # Calculate R2
        oos_r2 = r2_score(y_test, y_pred)
        results_dict[f'{split}_oos_r2'] = oos_r2

        # Calculate MAE
        oos_mae = np.mean(np.abs(y_test - y_pred))
        results_dict[f'{split}_oos_mae'] = oos_mae

        # Calculate MSE
        oos_mse = np.mean((y_test - y_pred)**2)
        results_dict[f'{split}_oos_mse'] = oos_mse

        # Calculate RMSE
        oos_rmse = np.sqrt(oos_mse)  # Fix the variable name to "oos_mse"
        results_dict[f'{split}_oos_rmse'] = oos_rmse

        ## find the min and max date of the test set, then convert to "dd-mmm-yyyy" format.

        # Find the min and max date of the test set
        start_date = min(y_test.index)
        end_date = max(y_test.index)

        # Convert the dates to the desired format
        results_dict[f'{split}_oos_start_date'] = start_date.strftime("%d/%m/%Y")
        results_dict[f'{split}_oos_end_date'] = end_date.strftime("%d/%m/%Y")

        # Correct the keys when storing start and end dates in the results_dict
        start_date_key = f'{split}_oos_start_date'  # Use 'start_date', not just 'start'
        end_date_key = f'{split}_oos_end_date'  # Use 'end_date', not just 'end'
        results_dict[start_date_key] = start_date.strftime("%d/%m/%Y")
        results_dict[end_date_key] = end_date.strftime("%d/%m/%Y")


    ########################################################################################################
    #### create a summary table the split-by-split results. Each column is a split, each row is a metric.
    ########################################################################################################

    final_table = PrettyTable()

    # Parse the dictionary and organize the data by metrics and splits
    nested_data = {}
    for key, value in results_dict.items():
        # Split the key into the split number and the metric name
        split_number, metric = key.split('_oos_')
        # Remove 'split_' from the split_number string and convert to int
        split_number = int(split_number.replace('split_', ''))
        # Populate the nested_data dictionary
        nested_data.setdefault(metric, {})[f"Sample {split_number}"] = value
    # Set up the header for the PrettyTable with the appropriate number of splits
    columns = [""] + [f"Sample {i}" for i in range(1, splits + 1)]
    final_table.field_names = columns

    # Add the rows to the PrettyTable for each metric
# Add the rows to the PrettyTable for each metric including the new start and end dates
    for metric in ['r2', 'mae', 'mse', 'rmse', 'start_date', 'end_date']:
        row = [metric.replace('_', ' ').title()]  # Format the metric name nicely
        # Add the data for each sample or an empty string if no data is available
        for i in range(1, splits + 1):
            row.append(nested_data.get(metric, {}).get(f'Sample {i}', ''))
        # Add the row to the PrettyTable
        final_table.add_row(row)

        ## reduce the number of decimals in the table to 4
        for i in range(1, splits + 1):
            final_table.align[f"Sample {i}"] = 'r'
            final_table.align[""] = 'l'
            final_table.float_format = '4.4'


    return final_table
