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
from IPython.display import display  # For displaying DataFrame styles in Jupyter Notebook


## fix that the output table from oos_summary_stats shows the dates as integers

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
    print(f"y is set to {y.name}, the first column of the dataset. To change this, please change the first column of the dataset.")

    return dataset

def remove_colinear_features(df, vif_threshold=10):
    """
    Remove collinear features from a DataFrame based on the Variance Inflation Factor (VIF).

    Parameters:
    - df (DataFrame): The input DataFrame containing the target variable and predictor variables.
    - vif_threshold (float): The threshold value for VIF. Features with VIF greater than this threshold will be removed.

    Returns:
    - df (DataFrame): The modified DataFrame with collinear features removed.
    """

    # Assuming 'y' is the first column and should be excluded from VIF calculation
    y = df.iloc[:, 0]  # Store 'y' separately

    X = df.iloc[:, 1:]  # Consider only the predictor variables for VIF

    # Loop until all VIFs are smaller than the cut-off value
    vif_cut_off = vif_threshold

    # List to store names of removed features
    removed_features = []

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
        removed_features.append(feature_with_max_vif)  # Add the removed feature to the list

    print("Done removing colinear features.")

    # Print the names of removed features
    if removed_features:
        print("Removed features due to high collinearity:", ", ".join(removed_features))
    else:
        print("No features were removed due to high collinearity.")

    # Reconstruct the dataframe with 'y' and the reduced set of features
    df = pd.concat([y, X], axis=1)

    return df

def exploratory_analysis(df):
    """
    Perform exploratory analysis on the dataset with the first column as the target variable.

    Parameters:
    - df (DataFrame): The input DataFrame containing the target variable and predictor variables.

    Returns:
    None. Displays correlation and scatter plots.
    """
    # Assuming 'y' is the first column and is the target variable
    target_variable = df.columns[0]
    print(f"Target variable selected: {target_variable}")

    # Create a correlation matrix with scatter plots for each pair of variables
    corr_matrix = df.corr()
    display(corr_matrix.style.background_gradient(cmap='coolwarm'))

    predictor_variables = df.columns.drop(target_variable)

    for predictor in predictor_variables:
        # Visual inspection using scatter plot
        sns.scatterplot(x=df[predictor], y=df[target_variable])
        plt.title(f"Scatter plot of {predictor} vs {target_variable}")
        plt.show()

        # Statistical test for non-linearity using Spearman's rank correlation
        correlation, p_value = spearmanr(df[predictor], df[target_variable])

        print(f"Spearman's correlation between {predictor} and {target_variable}: {correlation:.2f}")
        print(f"P-value: {p_value:.3f}")

        # Determine if transformation might be necessary
        if p_value < 0.05 and correlation not in [-1, 1]:
            print(f"Potential non-linear relationship detected for {predictor}. Consider transformation.")
        else:
            print(f"No strong evidence of non-linear relationship for {predictor}.")
        print()


def generate_random_string():
    # Generate a random letter (either uppercase or lowercase)
    random_letter = random.choice(string.ascii_letters)

    # Generate a random digit
    random_digit = random.choice(string.digits)

    # Combine them to form a two-character string
    return random_letter + random_digit


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
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    if not isinstance(splits, int) or splits <= 0:
        raise ValueError("Parameter 'splits' must be an integer greater than 0.")

    if not isinstance(lags, int) or lags < 0:
        raise ValueError("Parameter 'lags' must be a non-negative integer.")

    if not isinstance(train_share, float) or not 0 < train_share < 1:
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

    print("Each split is assigned a unique ID. The ID is a random two-character string with one letter + one number.")

    return splits_dict

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

    def create_lags_for_df(dataframe, lags):
        original_columns = dataframe.columns  # Store the original columns
        lagged_df = dataframe.copy()
        for lag in range(1, lags + 1):
            shifted = dataframe[original_columns].shift(lag)  # Only shift original columns
            shifted.columns = [f'{col}_lag{lag}' for col in original_columns]
            lagged_df = pd.concat([lagged_df, shifted], axis=1)

        # Drop rows where any feature other than 'y' is NaN
        cols_except_y = [col for col in lagged_df.columns if col not in original_columns]
        lagged_df = lagged_df.dropna(subset=cols_except_y)
        return lagged_df

    if isinstance(df, pd.DataFrame):
        df = create_lags_for_df(df, lags)
    elif isinstance(df, dict):
        for split in df:
            train_key = next((key for key in df[split] if key.startswith('train')), None)
            test_key = next((key for key in df[split] if key.startswith('test')), None)
            if train_key:
                df[split][train_key] = create_lags_for_df(df[split][train_key], lags)
            if test_key:
                df[split][test_key] = create_lags_for_df(df[split][test_key], lags)

    return df

def regression_OLS(splits_dict, p_cutoff=0.05):
    """
    Perform OLS regression for each split in the splits_dict and return a structured dictionary.

    Parameters:
    - splits_dict (dict): Dictionary containing splits of the DataFrame.
    - p_cutoff (float): The p-value cutoff for feature elimination. Default is 0.05.

    Returns:
    - dict: A structured dictionary with each split's data and fitted model.
    """

    # validate that p_cutoff is a float between 0 and 1
    if not isinstance(p_cutoff, float) or not 0 < p_cutoff < 1:
        raise ValueError("Parameter 'p_cutoff' must be a float between 0 and 1.")

    # raise caution if p_cutoff is above 0.1 or below 0.01
    if p_cutoff > 0.1:
        print("Warning: p_cutoff is above 0.1. This may result in a model with too many features.")
    elif p_cutoff < 0.01:
        print("Warning: p_cutoff is below 0.01. This may result in a model with too few features.")

    def fit_model(train_data):
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
    Creates a single dataframe containing actual and predicted values of y for both train and test datasets across all splits.

    Parameters:
    - ols_output (dict): The output from regression_OLS function, containing data and models for each split.

    Returns:
    - pd.DataFrame: A combined dataframe with the actual and predicted values of y across all splits.
    """

    def prepare_data_for_prediction(df, model, model_features):
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
    if not all_splits_df.index.empty and isinstance(all_splits_df.index[0], (int, float)):
        # This assumes that the index is meant to be a date and is in a format that pd.to_datetime can parse.
        all_splits_df.index = pd.to_datetime(all_splits_df.index, format='%Y%m%d')  # Adjust the format as necessary

    return all_splits_df.reset_index(drop=True)

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def oos_summary_stats(fit_and_predict_output):
    """
    Calculate out-of-sample summary statistics for each split, including the unique ID and time period covered.

    Parameters:
    fit_and_predict_output (pd.DataFrame): DataFrame containing 'y_actual', 'y_fitted_{split_name}',
                                           and 'y_pred_{split_name}' for each split with a datetime index.

    Returns:
    pd.DataFrame: Summary statistics for each split, including split ID and time period covered.
    """
    # Initialize an empty DataFrame to store summary statistics
    summary_stats = pd.DataFrame()

    # Extract split names based on the 'y_pred_' pattern in the column names
    split_names = [col for col in fit_and_predict_output.columns if 'y_pred_' in col]
    split_ids = [name.split('_')[-1] for name in split_names]

    # Calculate statistics for each split
    for split_name, split_id in zip(split_names, split_ids):
        # Select the non-NaN predicted and actual values for the current split
        mask = ~fit_and_predict_output[split_name].isna()
        y_actual = fit_and_predict_output.loc[mask, 'y_actual']
        y_pred = fit_and_predict_output.loc[mask, split_name]

        if y_actual.empty:
            # If there are no non-NaN values for the current split, skip it
            continue

        if y_actual.isna().any() or y_pred.isna().any():
            # If there are NaN values in either the actual or predicted values, skip this split
            continue

        # Calculate R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE)
        r2 = r2_score(y_actual, y_pred)
        mae = np.mean(np.abs(y_actual - y_pred))
        mse = np.mean((y_actual - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # Determine the time period covered by this split
        # Assuming that the index is a datetime type or can be converted to one
        dates = fit_and_predict_output.index[mask]
        start_date = dates.min().strftime('%d/%b/%y') if hasattr(dates.min(), 'strftime') else str(dates.min())
        end_date = dates.max().strftime('%d/%b/%y') if hasattr(dates.max(), 'strftime') else str(dates.max())
        time_period = f"{start_date} to {end_date}"

        # Populate the summary DataFrame
        summary_stats[split_id] = [r2, mae, mse, rmse, time_period]

    # Define the index names for the summary DataFrame
    summary_stats.index = ['oos_r2', 'oos_mae', 'oos_mse', 'oos_rmse', 'time_period']

    return summary_stats



def compare_fitted_models(ols_output):
    """
    Modified to work with output from regression_OLS function.
    Compares fitted models using Stargazer.

    Parameters:
    ols_output (dict): Output from regression_OLS function.

    Returns:
    Stargazer object: Comparison table of models.
    """
    from stargazer.stargazer import Stargazer

    models = [content['model'] for content in ols_output.values()]
    stargazer = Stargazer(models)

    # Optional: Customize the Stargazer table further if needed
    # e.g., stargazer.title("Comparison of Models")

    return stargazer


def one_stop_analysis(file_location, lags=5, splits=5, train_share=0.8, vif_threshold=10, p_cutoff=0.05):
    """
    A comprehensive function that performs the entire process from data loading to model analysis and displays the results.

    Parameters:
    - file_location (str): File location of the dataset.
    - lags (int): Number of lagged variables to create for each predictor variable.
    - splits (int): Number of splits to create from the DataFrame.
    - train_share (float): Proportion of data to use for training.
    - vif_threshold (float): Threshold value for Variance Inflation Factor.
    - p_cutoff (float): P-value cutoff for feature elimination in OLS regression.

    Returns:
    None. Outputs are displayed within the function.
    """
    # Step 1: Load and preprocess the data
    df = load_df(file_location)
    df = remove_colinear_features(df, vif_threshold=vif_threshold)
    exploratory_analysis(df)

    # Step 2: Create splits and lagged features
    splits_dict = create_splits(df, lags=lags, splits=splits, train_share=train_share)

    # Step 3: Perform OLS regression on each split
    ols_output = regression_OLS(splits_dict, p_cutoff=p_cutoff)

    # Step 4: Fit and predict using the models
    fit_predict_output = fit_and_predict(ols_output)

    # Step 5: Calculate out-of-sample summary statistics and display the results
    summary_stats_table = oos_summary_stats(fit_predict_output)
    print("Out-of-Sample Summary Statistics:")
    display(summary_stats_table)  # Display summary statistics in Jupyter Notebook


    # Step 6: Compare fitted models and display the results
    model_comparison_table = compare_fitted_models(ols_output)
    print("Model Comparison:")
    display(model_comparison_table)  # Display model comparison in Jupyter Notebook
