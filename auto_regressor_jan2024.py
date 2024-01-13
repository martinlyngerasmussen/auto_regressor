# import libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor, reset_ramsey
from prettytable import PrettyTable
from datetime import datetime
from stargazer.stargazer import Stargazer
import os



##### Add to compiler function: calculate_residuals, diagnostics

def load_df(file_location):
    """
    Constructs a dataframe with data from a csv file and removes colinear features.

    Parameters:
    - file_location (str): The file location of the dataset.

    Returns:
    - df (pd.DataFrame): The dataframe with colinear features removed.
    """
    # Check if the file exists
    if not os.path.exists(file_location):
        raise FileNotFoundError(f"The file at {file_location} does not exist.")

    # Check if the file is readable
    if not os.access(file_location, os.R_OK):
        raise PermissionError(f"The file at {file_location} is not readable.")

    # Import the dataset with try-except to handle potential errors
    try:
        if file_location.endswith('.csv'):
            dataset = pd.read_csv(file_location)
        elif file_location.endswith('.xlsx'):
            dataset = pd.read_excel(file_location)
        else:
            raise ValueError("Invalid file format. Only csv and excel files are supported.")
    except Exception as e:
        raise IOError(f"Error reading file {file_location}: {e}")


    # Import the dataset
    if file_location.endswith('.csv'):
        dataset = pd.read_csv(file_location)
    elif file_location.endswith('.xlsx'):
        dataset = pd.read_excel(file_location)
    else:
        raise ValueError("Invalid file format. Only csv and excel files are supported.")

    # Convert date column to datetime format
    date_columns = ['date', 'Date']
    for col in date_columns:
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col], infer_datetime_format=True).dt.date
            dataset.set_index(col, inplace=True)
            break
    else:
        raise ValueError("No column named 'date' or 'Date' in dataset. Please change the name of the date column to 'date' or 'Date'.")

    # Assuming 'y' is the first column and should be excluded from VIF calculation
    y = dataset.iloc[:, 0]  # Store 'y' separately
    print(f"y is set to {y.name}, the first column of the dataset. To change this, please change the first column of the dataset.")

    X = dataset.iloc[:, 1:]  # Consider only the predictor variables for VIF

    # Loop until all VIFs are smaller than the cut-off value
    vif_cut_off = 5

    print("Removing colinear features...")
    while True:
        # Create a DataFrame with the features and their respective VIFs
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns

        # Find the variable with the highest VIF
        max_vif = vif["VIF Factor"].max()

        if max_vif <= vif_cut_off:
            break  # Exit the loop if all VIFs are below the threshold

        # Get the feature name with the highest VIF
        feature_with_max_vif = vif[vif["VIF Factor"] == max_vif]["features"].iloc[0]

        # Remove the feature with the highest VIF from X
        X = X.drop(feature_with_max_vif, axis=1)
        print(f"Variable '{feature_with_max_vif}' is being dropped due to high multicollinearity (VIF = {max_vif}).")

    print("Done removing colinear features.")
    # Reconstruct the dataframe with 'y' and the reduced set of features
    df = pd.concat([y, X], axis=1)

    return df

def create_lags(df, lags=5):
    """
    Create lagged features for a dataframe or a dictionary of dataframes.

    Parameters:
    df (pd.DataFrame or dict): The dataframe or dictionary of dataframes to create lagged features for.
    lags (int): The number of lagged features to create.

    Returns:
    pd.DataFrame or dict: The dataframe or dictionary of dataframes with lagged features created.
    """
    if not (isinstance(df, pd.DataFrame) or isinstance(df, dict)):
        raise TypeError("Input must be a pandas DataFrame or a dictionary of DataFrames.")

    # test if df is a dataframe or a dictionary of dataframe:
    if isinstance(df, dict):
        # for each dataframe in df, and each train and test set in each dataframe, create lags.
        # Each dataframe in df is a split, and each split has a train and test set.

        for split in df:
            for dataset in df[split]:
                # Create lagged features for each split, excluding the index (date)
                original_columns = df[split][dataset].columns

                for lag in range(1, lags + 1):
                    # print which columns are being lagged)
                    for col in original_columns:
                        df[split][dataset][f'{col}_lag{lag}'] = df[split][dataset][col].shift(lag)

                # Drop the initial rows with NaN values due to lagging BUT NOT IF ONLY Y IS missing.
                # List of all columns except the first one (assumed to be 'y')
                columns_except_first = df[split][dataset].columns[1:]

                # Drop rows where any of the columns except the first one have NaN values
                df[split][dataset] = df[split][dataset].dropna(subset=columns_except_first)


    elif isinstance(df, pd.DataFrame):
        # Create lagged features for each split, excluding the index (date)
        original_columns = df.columns

        for lag in range(1, lags + 1):
            # print which columns are being lagged)
            for col in original_columns:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # Drop the initial rows with NaN values due to lagging BUT NOT IF ONLY Y IS missing.
        # List of all columns except the first one (assumed to be 'y')
        columns_except_first = df.columns[1:]

        # Drop rows where any of the columns except the first one have NaN values
        df = df.dropna(subset=columns_except_first)

    return df

def create_splits(df, lags=5, splits=5, train_share=0.8):
    """
    Prepare the data for regression analysis by performing the following steps:
    1. Split the DataFrame into multiple splits, where each split is a DataFrame.
    2. Divide each split into a train and test set.
    3. Create lagged features for the train and test sets of each split.

    Parameters:
    - df (pd.DataFrame): The dataframe to be split.
    - lags (int): The number of lagged variables to create for each predictor variable (default: 5).
    - splits (int): The number of splits to create from the DataFrame (default: 5).
    - train_share (float): The proportion of data to use for training (default: 0.8).

    Returns:
    - splits_dict (dict): A dictionary containing the splits of the DataFrame, where each split is further divided into train and test sets.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if not (0 < train_share < 1):
        raise ValueError("train_share must be between 0 and 1.")


    # Split the DataFrame into multiple splits
    split_dfs = np.array_split(df, splits)

    # Create a dictionary to store the splits
    splits_dict = {}

    for i, split_df in enumerate(split_dfs):
        split_name = f"split_{i+1}"

        # Calculate the split point for train-test division
        split_point = int(len(split_df) * train_share)

        # Create train and test sets for the split
        train_df = split_df.iloc[:split_point]
        test_df = split_df.iloc[split_point:]

        # Create lagged features for train and test sets
        train_df = create_lags(train_df, lags)
        test_df = create_lags(test_df, lags)

        # Store the train and test sets in the splits dictionary
        splits_dict[split_name] = {
            f"{split_name}_train": train_df,
            f"{split_name}_test": test_df
        }

    return splits_dict

def regression_OLS(df, p_cutoff = 0.05):
    """
    This function performs backward elimination on the dataframe using OLS.

    Parameters:
    - df (pd.DataFrame): The dataframe to be fitted with OLS. y should be in the first column, all columns after that should be features.
    - p_cutoff (float): The p-value cut-off for backward elimination (default: 0.05).

    Returns:
    - model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted model
    """
    if df.empty or len(df.columns) < 2:
        raise ValueError("DataFrame must have at least one target and one feature column.")


    df = df.copy()
    X = df.iloc[:, 1:]
    X = sm.add_constant(X)
    y = df.iloc[:, [0]]

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
        try:
            model = sm.OLS(y, X).fit(cov_type = "HAC", cov_kwds={'maxlags': 4})

        except Exception as e:
            raise RuntimeError(f"Error fitting the OLS model: {e}")

    return model

def predict(df, model):
    """
    This function creates a dataframe with the actual and predicted values of y.

    Parameters:
    - df (pd.DataFrame): The dataframe to be used for prediction. y should be in the first column, all columns after that should be features.
    - model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted model

    Returns:
    - df_pred (pd.DataFrame): The dataframe with the actual and predicted values of y.

    """

    model = model
    df = df.copy()
    X = df.iloc[:, 1:]
    X = sm.add_constant(X)
    y = df.iloc[:, [0]]

    # Predict the target variable
    y_pred = model.predict(X)

    df_pred = pd.DataFrame()
    df_pred[['y_actual']] = y
    df_pred[['y_pred']] = y_pred

    return df_pred

def oos_summary_stats(df_pred):
    """
    Calculate out-of-sample summary statistics for a regression model. The idea is that the function is applied to out-of-sample data, though it can technically also be applied to in-sample data.

    Parameters:
    df_pred (DataFrame): DataFrame containing the actual and predicted values.

    Returns:
    dict: Dictionary containing the out-of-sample summary statistics.
        - oos_r2 (float): R-squared value.
        - oos_mae (float): Mean absolute error.
        - oos_mse (float): Mean squared error.
        - oos_rmse (float): Root mean squared error.
        - start_date (str): Start date of the test set in "dd/mm/yyyy" format.
        - end_date (str): End date of the test set in "dd/mm/yyyy" format.
    """

    y_test = df_pred['y_actual']
    y_pred = df_pred['y_pred']

    ## calculate the following for y_test and y_pred: R2, MAE, MSE, RMSE, MAPE
    oos_stats = {}

    # Calculate R2
    oos_stats['oos_r2'] = r2_score(y_test, y_pred)
    oos_stats['oos_mae'] = np.mean(np.abs(y_test - y_pred))
    oos_mse = np.mean((y_test - y_pred)**2)
    oos_stats['oos_mse'] = oos_mse
    oos_stats['oos_rmse'] = np.sqrt(oos_mse)  # Fix the variable name to "oos_mse"

    ## find the min and max date of the test set, then convert to "dd-mmm-yyyy" format.

    # Find the min and max date of the test set
    start_date = min(y_test.index)
    end_date = max(y_test.index)

    oos_stats['start_date'] = min(df_pred.index).strftime("%d/%m/%Y")
    oos_stats['end_date'] = max(df_pred.index).strftime("%d/%m/%Y")

    return oos_stats

def compare_fitted_models(models_and_data):
    """
    Compare the fitted models using Stargazer.

    Parameters:
    models_and_data: this is a dictionary with the following structure:
        models_and_data = {'model 1': {'fitted_model': model object,
                                        'dataset': df used to train model},
                            'model 2': {'fitted_model': model object,
                                        'dataset': df used to train model},
                            etc.
        }

    Returns:
    a table that compares the models provided in models_and_data.

    """

    ## loop over models_and_data to extract each fitted_model and attach it to the models list.

    models = []
    for model in models_and_data.keys():
        models.append(models_and_data[model]["fitted_model"])

    stargazer = Stargazer(models)
    ones_list = [1 for _ in models]

    # Initialize an empty list for model_names_stargaze
    model_names_stargaze = []

    ## loop over models_and_data to extract each the start_date and end_date of each dataset and attach it to model_names_stargaze.
    for model in models_and_data.keys():
        start_date = models_and_data[model]["dataset"].index[0].strftime("%d/%m/%Y")
        end_date = models_and_data[model]["dataset"].index[-1].strftime("%d/%m/%Y")
        model_names_stargaze.append(f'{model}: {start_date} to {end_date}')
        model_names_stargaze.append(('RESET test', reset_ramsey(model, degree=3)))

    stargazer.custom_columns(model_names_stargaze, ones_list)

    return stargazer

def calculate_residuals():
    """
    Compares residuals in-sample vs. out-of-sample.

    Parameters:
    df_pred (DataFrame): DataFrame containing the actual and predicted values.

    Returns:
    dict: Dictionary containing the in-sample and out-of-sample residuals.
        - in_sample_residuals (DataFrame): In-sample residuals.
        - oos_residuals (DataFrame): Out-of-sample residuals.
    """

    pass

def compiler_function(file_location, lags, splits, train_share):
    """
    Compiles and executes a series of steps for econometric analysis.

    Parameters:
    file_location (str): The file location of the dataset.
    lags (int): The number of lags to create for each variable.
    splits (int): The number of splits to create for cross-validation.
    train_share (float): The proportion of the dataset to use for training.

    Returns:
    tuple: A tuple containing the following dictionaries:
        - split_dfs: A dictionary of split datasets.
        - regression_models: A dictionary of regression models.
        - predictions: A dictionary of predictions.
        - oos_summary_stats_dict: A dictionary of out-of-sample summary statistics.
    """

    # Load the dataframe.
    df = load_df(file_location)

    # Create splits
    split_dfs = create_splits(df, lags = lags, splits = splits, train_share = train_share)

    # create lags for each split in split_dfs.
    for split in split_dfs:
        split_dfs[split][f"{split}_train"] = create_lags(split_dfs[split][f"{split}_train"], lags)
        split_dfs[split][f"{split}_test"] = create_lags(split_dfs[split][f"{split}_test"], lags)

    # do regression_OLS for each train set in split_dfs, then attach the model to the dictionary.
    regression_models = {}
    for split in split_dfs:
        regression_models[split] = regression_OLS(split_dfs[split][f"{split}_train"])

    # before doing predictions, subset the test set to only include the columns that are also in the train set.
    for split in split_dfs:
        split_dfs[split][f"{split}_test"] = split_dfs[split][f"{split}_test"][split_dfs[split][f"{split}_train"].columns]

        # make sure that 'const' is not in the test set if it is not in the train set.
        if 'const' in split_dfs[split][f"{split}_train"].columns and 'const' not in split_dfs[split][f"{split}_test"].columns:
            split_dfs[split][f"{split}_test"] = split_dfs[split][f"{split}_test"].drop('const', axis=1)

        # but if the train set has a constant but the test set doesn't, then add it to the test set
        elif 'const' in split_dfs[split][f"{split}_test"].columns and 'const' not in split_dfs[split][f"{split}_train"].columns:
            split_dfs[split][f"{split}_test"] = sm.add_constant(split_dfs[split][f"{split}_test"])

    ## test if the shape of the test set is the same as the train set, then print a warning if it is not.
    for split in split_dfs:
        if split_dfs[split][f"{split}_test"].shape[1] != split_dfs[split][f"{split}_train"].shape[1]:
            print(f"Warning: the number of columns in {split}_test is not the same as the number of columns in {split}_train.")

    # do prediction for each test set in split_dfs, then attach the prediction to the dictionary.
    predictions = {}
    for split in split_dfs:
        predictions[split] = predict(split_dfs[split][f"{split}_test"], regression_models[split])

    # do predictions based on the full dataset, then attach the prediction to the dictionary.
    df_full = df.copy()
    df_full = create_lags(df_full, lags)

    for split in split_dfs:
        predictions[f'full_sample_pred_{split}'] = predict(df_full, regression_models[split])

    # do oos_summary_stats for each prediction in predictions, then attach the summary stats to the dictionary.
    oos_summary_stats_dict = {}
    for split in split_dfs:
        oos_summary_stats_dict[split] = oos_summary_stats(predictions[split])

    return split_dfs, regression_models, predictions, oos_summary_stats_dict
