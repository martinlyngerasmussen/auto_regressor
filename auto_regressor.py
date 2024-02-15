# import libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor, reset_ramsey
from scipy.stats import spearmanr
from stargazer.stargazer import Stargazer
import string
import random
from IPython.display import display


def load_df(file_location):
    """
    Load a dataset from a file and perform necessary preprocessing.

    Parameters:
    file_location (str): The path to the file.

    Returns:
    pandas.DataFrame: The loaded dataset.

    Raises:
    FileNotFoundError: If the file does not exist.
    PermissionError: If the file is not readable.
    ValueError: If the file format is not supported or no datetime column is found.
    IOError: If there is an error reading the file.
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

    # Check for a datetime column first
    datetime_columns = [col for col in dataset.columns if pd.api.types.is_datetime64_any_dtype(dataset[col])]


    if len(datetime_columns) > 0:
        # If there's a datetime column, use the first one found
        date_column = datetime_columns[0]
        dataset[date_column] = pd.to_datetime(dataset[date_column], infer_datetime_format=True).dt.date
        dataset.set_index(date_column, inplace=True)
    else:
        # Fall back to 'date' or 'Date'
        date_columns = ['date', 'Date']
        for col in date_columns:
            if col in dataset.columns:
                dataset[col] = pd.to_datetime(dataset[col], infer_datetime_format=True).dt.date
                dataset.set_index(col, inplace=True)
                break
        else:
            raise ValueError("No datetime column or column named 'date' or 'Date' found in dataset.")

       # Assuming 'y' is the first column and should be excluded from VIF calculation
    y = dataset.iloc[:, 0]  # Store 'y' separately


    print("")
    print("##################")
    print("DATASET LOADED")
    print("##################")
    print("")

    # Print separately: y (dependent variable), X (independent variables), sample size, date range
    print(f"Dependent variable (y): {dataset.columns[0]}")
    print(f"Independent variables (X): {', '.join(dataset.columns[1:])}")
    print(f"Sample size: {len(dataset)}")
    print(f"Date range: {dataset.index.min()} to {dataset.index.max()}")

    # Infer the frequency of the dataset
    inferred_freq = pd.infer_freq(dataset.index)
    if inferred_freq is not None:
        # Set the frequency on the index
        dataset.index = pd.DatetimeIndex(dataset.index, freq=inferred_freq)

        # Now you can check the frequency and add dummies accordingly
        if inferred_freq.startswith('M'):
            # Create dummies for months if the data is monthly
            dataset['jan_2020'] = np.where((dataset.index.month == 1) & (dataset.index.year == 2020), 1, 0)
            dataset['feb_2020'] = np.where((dataset.index.month == 2) & (dataset.index.year == 2020), 1, 0)
            dataset['mar_2020'] = np.where((dataset.index.month == 3) & (dataset.index.year == 2020), 1, 0)
            dataset['apr_2020'] = np.where((dataset.index.month == 4) & (dataset.index.year == 2020), 1, 0)
            dataset['may_2020'] = np.where((dataset.index.month == 5) & (dataset.index.year == 2020), 1, 0)
            dataset['jun_2020'] = np.where((dataset.index.month == 6) & (dataset.index.year == 2020), 1, 0)
            dataset['jul_2020'] = np.where((dataset.index.month == 7) & (dataset.index.year == 2020), 1, 0)
            dataset['aug_2020'] = np.where((dataset.index.month == 8) & (dataset.index.year == 2020), 1, 0)
            dataset['sep_2020'] = np.where((dataset.index.month == 9) & (dataset.index.year == 2020), 1, 0)
            dataset['oct_2020'] = np.where((dataset.index.month == 10) & (dataset.index.year == 2020), 1, 0)
            dataset['nov_2020'] = np.where((dataset.index.month == 11) & (dataset.index.year == 2020), 1, 0)
            dataset['dec_2020'] = np.where((dataset.index.month == 12) & (dataset.index.year == 2020), 1, 0)
        elif inferred_freq.startswith('Q'):
            # Create dummies for quarters if the data is quarterly
            dataset['q1_2020'] = np.where((dataset.index.quarter == 1) & (dataset.index.year == 2020), 1, 0)
            dataset['q2_2020'] = np.where((dataset.index.quarter == 2) & (dataset.index.year == 2020), 1, 0)
            dataset['q3_2020'] = np.where((dataset.index.quarter == 3) & (dataset.index.year == 2020), 1, 0)
            dataset['q4_2020'] = np.where((dataset.index.quarter == 4) & (dataset.index.year == 2020), 1, 0)

    return dataset

def remove_colinear_features(df, vif_threshold=10):
    """
    Remove collinear features from a DataFrame based on the Variance Inflation Factor (VIF).

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the predictor variables.
    vif_threshold (float, optional): The threshold value for VIF. Default is 10.

    Returns:
    pandas.DataFrame: The modified DataFrame with collinear features removed.
    """

    # Assuming 'y' is the first column and should be excluded from VIF calculation
    y = df.iloc[:, 0]  # Store 'y' separately

    X = df.iloc[:, 1:]  # Consider only the predictor variables for VIF

    # Loop until all VIFs are smaller than the cut-off value
    vif_cut_off = vif_threshold

    # List to store names of removed features
    removed_features = []

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
        removed_features.append(feature_with_max_vif)  # Add the removed feature to the list

    # Print the names of removed features
    if removed_features:
        print("Removed features due to high collinearity:", ", ".join(removed_features))

    # Reconstruct the dataframe with 'y' and the reduced set of features
    df = pd.concat([y, X], axis=1)

    return df

def exploratory_analysis(df):
    """
    Perform exploratory analysis on a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None
    """

    # Assuming 'y' is the first column and is the target variable
    target_variable = df.columns[0]
    print(f"Target variable selected: {target_variable}")

    # Drop dummy variables and the target variable from the list of predictors
    variables_no_dummies = [var for var in df.columns[1:] if not var.startswith(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'q1', 'q2', 'q3', 'q4'))]

    # Create a correlation matrix with scatter plots for each pair of variables
    corr_matrix = df[variables_no_dummies].corr()
    print("")
    print("Correlation matrix:")
    display(corr_matrix.style.background_gradient(cmap='coolwarm'))

    # Define the number of rows and columns for the subplots based on the number of predictor variables
    num_predictors = len(variables_no_dummies)
    num_cols = 3  # for example, you can arrange the plots in 3 columns
    num_rows = (num_predictors + num_cols - 1) // num_cols  # calculate the number of rows needed

    # Create a figure with subplots
    plt.figure(figsize=(num_cols * 5, num_rows * 5))  # Adjust the size as needed

    for i, predictor in enumerate(variables_no_dummies, 1):
        plt.subplot(num_rows, num_cols, i)  # Create a subplot for each predictor
        plt.scatter(df[predictor], df[target_variable])  # Create the scatter plot
        plt.title(f"{predictor} (X) vs. {target_variable} (y)")  # Set the title
        plt.xlabel(predictor)
        plt.ylabel(target_variable)

    plt.tight_layout()  # Adjust the layout so the plots are neatly arranged
    plt.show()

def non_linearity(df):
    """
    Check for non-linearity between predictor variables and the target variable.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the predictor and target variables.

    Returns:
    pandas.DataFrame: The modified dataframe with additional columns for squared predictor variables.

    """
    df_final = df.copy()

    # Assuming 'y' is the first column and is the target variable
    target_variable = df.columns[0]

    # Create a correlation matrix with scatter plots for each pair of variables
    predictor_variables = df.columns.drop(target_variable)
    df_temp = df.copy().dropna()

    for predictor in predictor_variables:
        # Statistical test for non-linearity using Spearman's rank correlation
        correlation, p_value = spearmanr(df_temp[predictor], df_temp[target_variable])

        # Determine if transformation might be necessary
        if p_value < 0.05 and correlation not in [-1, 1]:
            ## Add a column with the square of the predictor variable (keep the sign, + or -, of the original variable. Also keep the original column)
            df_final[f"{predictor}_squared_sign_kept"] = np.sign(df_final[predictor]) * df_final[predictor] ** 2
            print(f"{predictor}_squared_sign_kept has been added as the relationship between {predictor} and {target_variable} is non-linear")

    return df_final

def generate_random_string():
    """
    Generate a random two-character string consisting of a random letter (either uppercase or lowercase)
    followed by a random digit.

    Returns:
        str: A random two-character string.
    """
    random_letter = random.choice(string.ascii_letters)
    random_digit = random.choice(string.digits)
    return random_letter + random_digit

def create_splits(df, lags=5, splits=5, train_share=0.8):
    """
    Split a pandas DataFrame into multiple train-test splits with lagged features.

    Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        lags (int): The number of lagged features to create.
        splits (int): The number of train-test splits to create.
        train_share (float): The proportion of data to use for training.

    Returns:
        dict: A dictionary containing the train and test sets for each split.
            The keys are the split names and the values are dictionaries
            with keys 'train_split_{split_id}' and 'test_split_{split_id}'.
    """

    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    if not isinstance(splits, int) or splits <= 0:
        raise ValueError("Parameter 'splits' must be an integer greater than 0.")

    if not isinstance(lags, int) or lags < 0:
        raise ValueError("Parameter 'lags' must be a non-negative integer.")

    if not isinstance(train_share, float) or not 0 < train_share <= 1:
        raise ValueError("Parameter 'train_share' must be a float between 0 and 1.")

    # Split the DataFrame into multiple splits
    split_dfs = np.array_split(df, splits)

    # Create a dictionary to store the splits
    splits_dict = {}

    for split_df in split_dfs:
        ## Create a random two-character string with one letter + one number.
        split_id = generate_random_string()

        split_name = f"split_{split_id}"

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
            f"train_split_{split_id}": train_df,
            f"test_split_{split_id}": test_df
        }

    return splits_dict

def create_lags(df, lags=5):
    """
    Create lagged features for a pandas DataFrame or a dictionary of DataFrames.

    Parameters:
    - df: pandas DataFrame or dictionary of DataFrames
        The input data to create lagged features for.
    - lags: int, optional (default=5)
        The number of lagged features to create.

    Returns:
    - lagged_df: pandas DataFrame or dictionary of DataFrames
        The DataFrame(s) with lagged features.

    Raises:
    - TypeError: If the input is not a pandas DataFrame or a dictionary of DataFrames.
    """

    if not (isinstance(df, pd.DataFrame) or isinstance(df, dict)):
        raise TypeError("Input must be a pandas DataFrame or a dictionary of DataFrames.")

    def create_lags_for_df(dataframe, lags):
        # Identify the dummy columns that should not be lagged
        dummy_columns = [col for col in dataframe.columns if col.startswith(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'q1', 'q2', 'q3', 'q4'))]

        # Select only the columns that are not dummies
        non_dummy_columns = [col for col in dataframe.columns if col not in dummy_columns]

        # Create a DataFrame to store lagged features
        lagged_df = dataframe.copy()

        # Create lags only for non-dummy variables
        for lag in range(1, lags + 1):
            shifted = dataframe[non_dummy_columns].shift(lag)
            shifted.columns = [f'{col}_lag{lag}' for col in non_dummy_columns]
            lagged_df = pd.concat([lagged_df, shifted], axis=1)

        # Drop rows with NaN values that occur at the start of the dataset for the first n = lags + 1 rows
        start_nan_rows = lagged_df.iloc[:lags+1].isna().any(axis=1)
        if start_nan_rows.any():
            lagged_df = lagged_df.iloc[start_nan_rows.sum():]

        return lagged_df

    if isinstance(df, dict):
        return {key: create_lags_for_df(df[key], lags) for key in df}
    else:
        return create_lags_for_df(df, lags)

def regression_OLS(splits_dict, p_cutoff=0.05, show_removed_features=False):
    """
    Perform Ordinary Least Squares (OLS) regression on the given data.

    Parameters:
    - splits_dict (dict): A dictionary containing the data splits for training and testing.
    - p_cutoff (float, optional): The p-value cutoff for feature elimination. Defaults to 0.05.
    - show_removed_features (bool, optional): Whether to print the removed features for each split. Defaults to False.

    Returns:
    - final_dict (dict): A dictionary containing the data splits, models, and removed features (if applicable).
    """

    # validate that p_cutoff is a float between 0 and 1
    if not isinstance(p_cutoff, float) or not 0 < p_cutoff < 1:
        raise ValueError("Parameter 'p_cutoff' must be a float between 0 and 1.")

    # raise caution if p_cutoff is above 0.1 or below 0.01
    if p_cutoff > 0.1:
        print("Warning: p_cutoff is above 0.1. This may result in a model with too many features.")
    elif p_cutoff < 0.01:
        print("Warning: p_cutoff is below 0.01. This may result in a model with too few features.")

    removed_features = []

    def fit_model(train_data):
        train_data = train_data.dropna()  # Drop rows with NaN values

        # Assuming 'y' is the first column
        y = train_data.iloc[:, 0]
        X = train_data.iloc[:, 1:]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        # Perform backward elimination
        while max(model.pvalues) > p_cutoff:
            if len(model.pvalues) == 1:  # Prevent removing all variables
                break
            highest_pval_feature = model.pvalues.idxmax()
            removed_features.append(highest_pval_feature)
            X.drop(highest_pval_feature, axis=1, inplace=True)
            model = sm.OLS(y, X).fit()

        return model

    final_dict = {}

    if isinstance(splits_dict, pd.DataFrame):
        fitted_model = fit_model(splits_dict)
        final_dict = {
            'data': splits_dict,
            'model': fitted_model
        }

        return final_dict

    for split_name, data in splits_dict.items():
        train_key = next((key for key in data if key.startswith('train')), None)
        test_key = next((key for key in data if key.startswith('test')), None)

        if train_key and test_key:
            train_data = data[train_key]
            test_data = data[test_key]

            fitted_model = fit_model(train_data)

            if show_removed_features:
                print("")
                print(f"{split_name}'s model due to p-values being above {p_cutoff}: {', '.join(removed_features)}")

            final_dict[split_name] = {
                'data': {
                    train_key: train_data,
                    test_key: test_data
                },
                'model': fitted_model
            }

    return final_dict

def fit_and_predict(ols_output):
    """
    Fits and predicts the values for each split in the OLS output.

    Args:
        ols_output (dict): A dictionary containing the OLS output for each split.

    Returns:
        pandas.DataFrame: A DataFrame containing the actual and predicted values for each split.
    """

    def prepare_data_for_prediction(df, model, model_features):
        """
        Prepares the data for prediction by aligning columns with model features.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            model (statsmodels.regression.linear_model.RegressionResultsWrapper): The regression model.
            model_features (list): The list of model features.

        Returns:
            pandas.DataFrame: The prepared DataFrame for prediction.
            pandas.Series: The target variable.
        """
        # Assuming 'y' is the first column
        y = df.iloc[:, 0]

        # Aligning columns with model features
        df_aligned = df.iloc[:, 1:].reindex(columns=model_features, fill_value=0)

        # Adding a constant if the model includes it
        if 'const' in model_features:
            df_aligned['const'] = 1

        return df_aligned, y

    all_splits_df = pd.DataFrame()

    for split_name, content in ols_output.items():
        model = content['model']
        model_features = model.model.exog_names

        # Predict for train dataset
        train_key = next((key for key in content['data'] if key.startswith('train')), None)
        if train_key:
            train_data, y_train = prepare_data_for_prediction(content['data'][train_key], model, model_features)
            y_fitted = model.predict(train_data)
            train_data = train_data.assign(y_actual=y_train, **{f'y_fitted_{split_name}': y_fitted})

        # Predict for test dataset
        test_key = next((key for key in content['data'] if key.startswith('test')), None)
        if test_key:
            test_data, y_test = prepare_data_for_prediction(content['data'][test_key], model, model_features)
            y_pred = model.predict(test_data)
            test_data = test_data.assign(y_actual=y_test, **{f'y_pred_{split_name}': y_pred})

        # Combine train and test datasets
        if train_key and test_key:
            combined_data = pd.concat([train_data, test_data])[['y_actual', f'y_fitted_{split_name}', f'y_pred_{split_name}']]
            all_splits_df = pd.concat([all_splits_df, combined_data])

    # Ensure the index is a datetime type if it's supposed to represent dates
    if not all_splits_df.index.empty and not isinstance(all_splits_df.index[0], pd.Timestamp):
        # This assumes that the index is meant to be a date and is in a format that pd.to_datetime can parse.
        all_splits_df.index = pd.to_datetime(all_splits_df.index, infer_datetime_format=True)

    return all_splits_df

def oos_summary_stats(fit_and_predict_output):
    """
    Calculate summary statistics for out-of-sample predictions.

    Parameters:
    fit_and_predict_output (DataFrame): DataFrame containing the predicted and actual values for each split.

    Returns:
    summary_stats (DataFrame): DataFrame containing the summary statistics for each split, including R2, MAE, MSE, RMSE,
                              sample period, sample length, and split ID.
    """

    # Initialize an empty DataFrame to store summary statistics
    summary_stats = pd.DataFrame()

    # Initialize an empty DataFrame to store summary statistics
    summary_stats = pd.DataFrame()

    # Extract split names based on the 'y_pred_' pattern in the column names
    split_names = [col for col in fit_and_predict_output.columns if 'y_pred_' in col]
    split_ids = [name.split('_')[-1] for name in split_names]

    # Initialize lists to collect stats and additional information for each split
    r2_list = []
    mae_list = []
    mse_list = []
    rmse_list = []
    time_period_list = []
    sample_size_list = []

    # Calculate statistics for each split
    for split_name, split_id in zip(split_names, split_ids):
        # Select the non-NaN predicted and actual values for the current split
        mask = ~fit_and_predict_output[split_name].isna() & ~fit_and_predict_output['y_actual'].isna()
        y_actual = fit_and_predict_output.loc[mask, 'y_actual']
        y_pred = fit_and_predict_output.loc[mask, split_name]

        # If there are NaN values in either the actual or predicted values, skip this split
        if y_actual.empty or y_actual.isna().any() or y_pred.isna().any():
            r2_list.append(np.nan)
            mae_list.append(np.nan)
            mse_list.append(np.nan)
            rmse_list.append(np.nan)
            time_period_list.append("NA to NA")
            sample_size_list.append(np.nan)
            continue

        # Calculate statistics and add them to their respective lists
        r2_list.append(r2_score(y_actual, y_pred))
        mae_list.append(np.mean(np.abs(y_actual - y_pred)))
        mse_list.append(np.mean((y_actual - y_pred) ** 2))
        rmse_list.append(np.sqrt(mse_list[-1]))

        # Format the time period and append to the list
        start_date_formatted = pd.to_datetime(y_actual.index.min()).strftime('%b-%y')
        end_date_formatted = pd.to_datetime(y_actual.index.max()).strftime('%b-%y')
        time_period_list.append(f"{start_date_formatted} to {end_date_formatted}")

        # Append the sample size for the split
        sample_size_list.append(mask.sum())

    # Create a DataFrame from the collected lists
    summary_stats = pd.DataFrame({
        'R2': r2_list,
        'MAE': mae_list,
        'MSE': mse_list,
        'RMSE': rmse_list,
        'Sample period': time_period_list,
        'Sample length': sample_size_list,
        'Split ID': split_ids  # Add Split ID as a column
    })

    # Transpose the DataFrame to have statistics as rows and splits as columns
    summary_stats = summary_stats.T

    # Rename the columns to "Split 1", "Split 2", etc.
    summary_stats.columns = [f"Split {i+1}" for i in range(summary_stats.shape[1])]

    return summary_stats

def compare_fitted_models(ols_output):
    """
    Compare the fitted models using the Stargazer library.

    Parameters:
    ols_output (dict): A dictionary containing the output of OLS regression for different models.

    Returns:
    stargazer (Stargazer): The Stargazer object containing the comparison table of the fitted models.
    """

    from stargazer.stargazer import Stargazer

    models = [content['model'] for content in ols_output.values()]
    stargazer = Stargazer(models)

    # Optional: Customize the Stargazer table further if needed
    # e.g., stargazer.title("Comparison of Models")

    return stargazer

def prediction_dummy(df):
    """
    Checks if the last date of the predictors is later than the last date of the target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target variable and predictors.

    Returns:
        bool: True if the last date of the predictors is later than the last date of the target variable, False otherwise.
    """

    # check that the index is a datetime index. If not, make the index a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Assuming 'y' is the first column and is the target variable
    target_variable = df.columns[0]

    # Get the last date of the target variable
    last_date_target = df[target_variable].dropna().index[-1] # Ensure to drop NA values to get the actual last date of observation

    # Get the last dates of all predictors and then find the maximum (latest) among them
    last_date_predictors = df.drop(target_variable, axis=1).apply(lambda x: x.dropna().index[-1]).min()

    # Create a dummy variable that is 1 if the last date of the predictors is later than the last date of the target variable
    prediction_dummy = int(last_date_predictors > last_date_target)

    if prediction_dummy == True:
        return True

def auto_regressor(file_location, lags=5, splits=1, train_share=0.9, vif_threshold=10, p_cutoff=0.05, show_removed_features = False):
    """
    Performs an automated regression analysis on a given dataset.

    Parameters:
    - file_location (str): The file path of the dataset.
    - lags (int): The number of lagged features to create.
    - splits (int): The number of splits to create for cross-validation.
    - train_share (float): The proportion of data to use for training.
    - vif_threshold (float): The threshold for removing collinear features based on VIF.
    - p_cutoff (float): The p-value threshold for feature selection.
    - show_removed_features (bool): Whether to display the removed features during regression.

    Returns:
    None
    """

    # Step 1: Load and preprocess the data
    df = load_df(file_location)
    df = remove_colinear_features(df, vif_threshold=vif_threshold)
    df = non_linearity(df)

    if prediction_dummy(df):
        print("")
        print("#############################################")
        print("PREDICTION")
        print("#############################################")
        print("")

        df_full = df.copy()
        df_full = create_splits(df_full, lags=lags, splits=1, train_share=float(1))
        ols_output_full = regression_OLS(df_full, p_cutoff=p_cutoff, show_removed_features = show_removed_features)
        model_comparison_table_full = compare_fitted_models(ols_output_full)
        fit_predict_output_full = fit_and_predict(ols_output_full)

        ## drop the "y_pred" columns
        fit_predict_output_full = fit_predict_output_full.drop(columns=[col for col in fit_predict_output_full.columns if 'y_pred' in col])

        fit_predict_output_full.tail(24).plot()
        plt.show()
        print("Model fitted using full dataset:")
        display(model_comparison_table_full)

        print("Predictions:")
        print(fit_predict_output_full.tail())


    print("")
    print("#############################################")
    print("EXPLORATORY ANALYSIS")
    print("#############################################")
    print("")

    exploratory_analysis(df)

    # Step 2: Create splits and lagged features
    splits_dict = create_splits(df, lags=lags, splits=splits, train_share=train_share)

    # Step 3: Perform OLS regression on each split
    ols_output = regression_OLS(splits_dict, p_cutoff=p_cutoff, show_removed_features = show_removed_features)

    # Step 4: Fit and predict using the models
    fit_predict_output = fit_and_predict(ols_output)

    # Step 5: Calculate out-of-sample summary statistics and display the results
    summary_stats_table = oos_summary_stats(fit_predict_output)


    print("")
    print("############################################################")
    print("ROBUSTNESS: OUT-OF-SAMPLE MODEL PERFORMANCE SPLITS")
    print("############################################################")

    display(summary_stats_table)  # Display summary statistics in Jupyter Notebook

    # Step 6: Compare fitted models and display the results
    model_comparison_table = compare_fitted_models(ols_output)
    print("")
    print("#############################################")
    print("PART 3: SIMILARITY OF MODELS ACROSS SPLITS")
    print("#############################################")
    print("")

    display(model_comparison_table)  # Display model comparison in Jupyter Notebook

    print("")
    print("#########################################################################################")
    print("PART 4: ROBUSTNESS OF PERFORMANCE: IN-SAMPLE VS. OUT-OF-SAMPLE PERFORMANCE ACROSS SPLITS")
    print("#########################################################################################")
    print("")

    fit_predict_output.plot()

    ## change the y axis such that the range covers 90% of the data
    plt.ylim([fit_predict_output.quantile(0.02).min(), fit_predict_output.quantile(0.98).max()])

    plt.show()
