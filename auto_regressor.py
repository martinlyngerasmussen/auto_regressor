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
from stargazer.stargazer import Stargazer, LineLocation

def full_df(file_location, lags = 5):
    """
    Constructs a dataframe with lagged features for each split.

    Parameters:
    - file_location (str): The file location of the dataset.
    - lags (int): The number of lagged features to create.

    Returns:
    - df (pd.DataFrame): dataframe with 'y' and the set of features.
    """

    # Import the dataset
    dataset = pd.read_csv(file_location)

    ## convert date column to datetime format
    dataset['date'] = pd.to_datetime(dataset['date'], infer_datetime_format= True).dt.date

    ## make date column the index
    df = dataset.copy()
    df.set_index('date', inplace=True)

    # Assuming 'y' is the first column and should be excluded from VIF calculation
    y = df.iloc[:, 0]  # Store 'y' separately
    X = df.iloc[:, 1:]  # Consider only the predictor variables for VIF

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
        feature_max_vif = vif[vif["VIF Factor"] == vif_max]["features"].values[0]  # get the actual feature name

        # Remove the feature with the highest VIF from X
        X = X.drop(feature_max_vif, axis=1)

    # Reconstruct the dataframe with 'y' and the reduced set of features
    df = pd.concat([y, X], axis=1)

    original_columns = df.columns

    # Create lagged features for each split, excluding the index (date)
    for lag in range(1, lags + 1):
        for col in original_columns:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

            # Drop the initial rows with NaN values due to lagging BUT NOT IF ONLY Y IS missing.
            # List of all columns except the first one (assumed to be 'y')
            columns_except_first = df.columns[1:]

            # Drop rows where any of the columns except the first one have NaN values
            df = df.dropna(subset=columns_except_first)


    df = sm.add_constant(df)
    return df

def data_preparation_splits(file_location, lags = 5, splits = 5, train_share = 0.8):
    """
    Prepare the data for regression analysis by performing the following steps:
    1. Import the dataset from the specified file location.
    2. Convert the 'date' column to datetime format and set it as the index.
    3. Calculate the Variance Inflation Factor (VIF) for each predictor variable and remove variables with VIF greater than the cut-off value.
    4. Split the DataFrame into multiple splits, where each split is a DataFrame.
    5. Create lagged variables for each split and divide each split into train and test sets.

    Parameters:
    - file_location (str): The file location of the dataset.
    - lags (int): The number of lagged variables to create for each predictor variable (default: 5).
    - splits (int): The number of splits to create from the DataFrame (default: 5).
    - train_share (float): The proportion of data to use for training (default: 0.8).

    Returns:
    - splits_dict (dict): A dictionary containing the splits of the DataFrame, where each split is further divided into train and test sets.
    """

    # Import the dataset
    dataset = pd.read_csv(file_location)

    ## convert date column to datetime format
    dataset['date'] = pd.to_datetime(dataset['date'], infer_datetime_format= True).dt.date

    ## make date column the index
    df = dataset.copy()
    df.set_index('date', inplace=True)

    # Assuming 'y' is the first column and should be excluded from VIF calculation
    y = df.iloc[:, 0]  # Store 'y' separately
    X = df.iloc[:, 1:]  # Consider only the predictor variables for VIF

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
        feature_max_vif = vif[vif["VIF Factor"] == vif_max]["features"].values[0]  # get the actual feature name

        # Remove the feature with the highest VIF from X
        X = X.drop(feature_max_vif, axis=1)

    # Reconstruct the dataframe with 'y' and the reduced set of features
    df = pd.concat([y, X], axis=1)

    ## create splits of the DataFrame, where splits = splits. Each split is a DataFrame. Name each split as df_split_i.
    # Split the DataFrame and store each split in a dictionary
    split_dfs = {}
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

            # Drop if NaN or missing values
            split_df = split_df.dropna()

            # Calculate the split point for train-test division
            split_point = int(len(split_df) * train_share)

            # Split into training and testing sets
            splits_dict[split] = {
                f"{split}_train": split_df.iloc[:split_point],
                f"{split}_test": split_df.iloc[split_point:]
            }

    return splits_dict




def regression_OLS(file_location, lags, splits, train_share, p_cutoff = 0.05):
    """
    Perform Ordinary Least Squares (OLS) regression on a dataset using cross-validation.

    Parameters:
    - file_location (str): The file path of the dataset.
    - lags (int): The number of lagged variables to include in the regression.
    - splits (int): The number of splits for cross-validation.
    - train_share (float): The proportion of the dataset to use for training.
    - p_cutoff (float, optional): The p-value cutoff for backward elimination. Defaults to 0.05.

    Returns:
    - results_dict (dict): A dictionary containing the results of the regression, including out-of-sample performance metrics.
    """
    # Function code goes here
    pass
def regression_OLS(file_location, lags, splits, train_share, p_cutoff = 0.05):
    cv_data = data_preparation_splits(file_location, lags, splits, train_share) ## cv = cross-validation
    df_full = full_df(file_location, lags)

    ## write a code that loops over each dataframe in the df dictionary
    # Create empty dictionary to store the results
    results_dict = {}

    ####### import the full sample of data
    ## make X
    full_sample_X = df_full.iloc[:, 1:]

    ## make y
    full_sample = df_full.iloc[:, [0]]

    ## import only y from the dataset
    oos_predictions = pd.DataFrame()

    # Get the name of the first column in df_full
    first_column_name = df_full.columns[0]

    # Use this name to rename the 'y' column in oos_predictions
    oos_predictions.rename(columns={'y': first_column_name}, inplace=True)

    # Assign the values from the first column of df_full to the newly renamed column
    oos_predictions[first_column_name] = df_full.iloc[:, 0]

    ## empty dataframes to be used during loop
    model_list = []
    model_names_stargaze = []
    model_number = int(1)
    start_dates = {}
    end_dates = {}

    ###########################################################
    #### fit models to each split of data in cv_data       ####
    ###########################################################

    for split in cv_data:
        ## split is the name of the split (contains all the splits=splits dataframe), split_df is the dataframe (each split contains two split_dfs, and we only want to look at the "train" ones):
        split_df = cv_data[split][f"{split}_train"]

        full_sample_X_loop = full_sample_X.copy()

        ## fit the OLS model on the split_df
        # Separate the target variable and the features
        y = split_df.iloc[:, 0]
        X = split_df.iloc[:, 1:]

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

            if len(X.columns) == 0: ## proceed to next split if only constant is left
                break

            # Fit the model without the feature with the highest p-value
            model = sm.OLS(y, X).fit(cov_type = "HAC", cov_kwds={'maxlags': 4})

        ## store the results of the final model in a dictionary
        # results_dict[f'{split}_summary'] = model.summary()

        final_model = model
        model_list.append(final_model)  ## add model to list of models used by stargazer

        ########## fit model both in-sample and out-of-sample ##########
        # Add a constant to the features if necessary
        if 'const' in X.columns:
            full_sample_X_loop = sm.add_constant(full_sample_X_loop)

        if 'const' not in X.columns and 'const' in full_sample_X.columns:
            full_sample_X_loop = full_sample_X_loop.drop('const', axis=1)

        # Exclude columns from full_sample_X that are not in X
        full_sample_X_loop = full_sample_X_loop[X.columns]


        # Predict the target variable
        full_sample[f'{split}_y_fitted'] = final_model.predict(full_sample_X_loop)


        ########## collect OOS predictions for each split ##########
        ## add predictions to oos_predictions dataframe
        oos_predictions[f'{split}_y_fitted'] = final_model.predict(X)

        ## predict the values of the test set using the estimated model
        # Store the test set in a new variable
        test_set = cv_data[split][f"{split}_test"]

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

        start_dates[split] = min(X.index)
        end_dates[split] = max(X.index)

        # Convert the dates to the desired format
        results_dict[f'{split}_oos_start_date'] = start_date.strftime("%d/%m/%Y")
        results_dict[f'{split}_oos_end_date'] = end_date.strftime("%d/%m/%Y")

        # Correct the keys when storing start and end dates in the results_dict
        start_date_key = f'{split}_oos_start_date'  # Use 'start_date', not just 'start'
        end_date_key = f'{split}_oos_end_date'  # Use 'end_date', not just 'end'
        results_dict[start_date_key] = start_date.strftime("%d/%m/%Y")
        results_dict[end_date_key] = end_date.strftime("%d/%m/%Y")

        model_number += 1

    oos_predictions['y_actual'] = df_full.iloc[:, 0]

    # remove constant from full_sample
    full_sample = full_sample.drop('const', axis=1)

    # average the full_sample_fitted y's across splits
    full_sample['y_fitted_ave'] = full_sample.iloc[:, 1:].mean(axis=1)

    ## add back y to full_sample.
    full_sample['y'] = df_full.iloc[:, 1]

    ################################################
    #### fit model to full dataset, no test set ####
    ################################################
    df_full_reg = full_df(file_location, lags)
    df_full_reg = df_full_reg.drop('const', axis=1)

    # Separate the target variable and the features
    y_fullsample = df_full_reg.iloc[:, 0].dropna()
    X_fullsample = df_full_reg.iloc[:, 1:]
    X_fullsample_pred = X_fullsample

    ## make sure that there the dates intersect for X_fullsample and y_fullsample.
    X_fullsample = X_fullsample[X_fullsample.index.isin(y_fullsample.index)]

    # Add a constant to the features
    X_fullsample = sm.add_constant(X_fullsample).dropna()

    # Fit the OLS model
    model = sm.OLS(y_fullsample, X_fullsample).fit(cov_type="HAC", cov_kwds={'maxlags': 4})

    # Perform backward elimination until all p-values are smaller than the cut-off value
    # Initialize the loop
    p_max = 1
    while p_max > p_cutoff:
        # Find the variable with the highest p-value
        p = model.pvalues
        p_max = max(p)
        feature_max_p = p.idxmax()

        # Remove the feature with the highest p-value
        X_fullsample = X_fullsample.drop(feature_max_p, axis=1)

        if len(X_fullsample.columns) == 0: ## proceed to next split if only constant is left
            break

        # Fit the model without the feature with the highest p-value
        model = sm.OLS(y_fullsample, X_fullsample).fit(cov_type = "HAC", cov_kwds={'maxlags': 4})

    ## store the results of the final model in a dictionary
    # results_dict[f'{split}_summary'] = model.summary()

    final_model = model
    model_list.append(final_model)  ## add model to list of models used by stargazer

    ########## fit model both in-sample and out-of-sample ##########
    # select the columns in X_fullsample_pred that are also in X_fullsample.
    X_fullsample_pred = X_fullsample_pred[X_fullsample.columns]

    # Add a constant to the features if necessary
    if 'const' in X_fullsample.columns:
        X_fullsample_pred = sm.add_constant(X_fullsample_pred)

    # Predict the target variable


    full_sample[f'y_full_fitted'] = final_model.predict(X_fullsample_pred)

    ########################################################################################################
    #### create a table that summarizes the out-of-sample performance across the different splits       ####
    ########################################################################################################

    oos_metrics_table = PrettyTable()

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
    oos_metrics_table.field_names = columns

    ## set the title of the table to XYZ
    oos_metrics_table.title = "Out of sample performance across sub-periods"

    title = "Out of sample performance across sub-periods".upper()  # Uppercase title
    title = f"*** {title} ***"  # Add symbols for emphasis
    oos_metrics_table.title = title


    # Add the rows to the PrettyTable for each metric
    # Add the rows to the PrettyTable for each metric including the new start and end dates
    for metric in ['r2', 'mae', 'mse', 'rmse', 'start_date', 'end_date']:
        row = [metric.replace('_', ' ').title()]  # Format the metric name nicely
        # Add the data for each sample or an empty string if no data is available
        for i in range(1, splits + 1):
            row.append(nested_data.get(metric, {}).get(f'Sample {i}', ''))
        # Add the row to the PrettyTable
        oos_metrics_table.add_row(row)

        ## reduce the number of decimals in the table to 4
        for i in range(1, splits + 1):
            oos_metrics_table.align[f"Sample {i}"] = 'r'
            oos_metrics_table.align[""] = 'l'
            oos_metrics_table.float_format = '4.4'


    ############################################################
    #### use Stargazer to compare models ####
    ############################################################

    stargazer = Stargazer(model_list)
    ones_list = [1 for _ in model_list]

    # Initialize an empty list for model_names_stargaze
    model_names_stargaze = []

    # Iterate over each split
    model_names_stargaze = []
    for split in cv_data.keys():
        start_date = start_dates[split].strftime("%d/%m/%Y")
        end_date = end_dates[split].strftime("%d/%m/%Y")
        model_names_stargaze.append(f'{split}: {start_date} to {end_date}')

    model_names_stargaze.append('Full sample')
    stargazer.custom_columns(model_names_stargaze, ones_list)

    # for oss_predictions and full_sample, if 'cons' exists then drop it.
    if 'const' in oos_predictions.columns:
        oos_predictions = oos_predictions.drop('const', axis=1)

    if 'const' in full_sample.columns:
        full_sample = full_sample.drop('const', axis=1)

    return oos_metrics_table, oos_predictions, stargazer, full_sample
