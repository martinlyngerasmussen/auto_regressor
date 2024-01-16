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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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

def exploratory_analysis(df, target_variable):
    ## create a correlation matrix with scatter plots for each pair of variables.
    corr_matrix = df.corr()
    corr_matrix.style.background_gradient(cmap='coolwarm')

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
        lagged_df = dataframe.copy()
        for lag in range(1, lags + 1):
            shifted = dataframe.shift(lag)
            shifted.columns = [f'{col}_lag{lag}' for col in dataframe.columns]
            lagged_df = pd.concat([lagged_df, shifted], axis=1)
        return lagged_df

    if isinstance(df, dict):
        for split in df:
            for dataset in df[split]:
                df[split][dataset] = create_lags_for_df(df[split][dataset], lags)
    else:
        if isinstance(df, dict):
            print("Please create splits before creating lags to avoid data leakage.")
        # ask user if they want to create lags for the full dataset. Only proceed if user says yes. If no, stop the function.
        elif input("Do you want to create lags for the full dataset? (y/n) ") == 'y':
            print("Creating lags for the full dataset...")
            df = create_lags_for_df(df, lags)
            print("Done creating lags for the full dataset.")
        else:
            print("No lags were created.")

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

def regression_OLS(input_df, p_cutoff=0.05):
    """
    Perform Ordinary Least Squares (OLS) regression with backward elimination.

    Parameters:
    - input_df (pandas DataFrame or dict): The input data for regression. If a DataFrame is provided, a single model is fitted.
      If a dictionary of DataFrames is provided, multiple models are fitted for each split in the dictionary.
    - p_cutoff (float, optional): The p-value cutoff for feature elimination. Default is 0.05.

    Returns:
    - model (statsmodels.regression.linear_model.RegressionResultsWrapper or dict): The fitted OLS model(s).

    Raises:
    - ValueError: If p_cutoff is not a float between 0 and 1.
    - TypeError: If input_df is not a pandas DataFrame or a dictionary of DataFrames.
    - ValueError: If 'train' dataset is not found in any split of the input_df dictionary.
    """

    # validate that p_cutoff is a float between 0 and 1
    if not isinstance(p_cutoff, float) or not 0 < p_cutoff < 1:
        raise ValueError("Parameter 'p_cutoff' must be a float between 0 and 1.")

    # raise caution if p_cutoff is above 0.1 or below 0.01:
    if p_cutoff > 0.1:
        print("Warning: p_cutoff is above 0.1. This may result in a model with too many features.")
    elif p_cutoff < 0.01:
        print("Warning: p_cutoff is below 0.01. This may result in a model with too few features.")

    def fit_model(df):
        # Fit the OLS model
        df = df.copy()
        X = df.iloc[:, 1:]
        X = sm.add_constant(X)
        y = df.iloc[:, [0]]

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={'maxlags': 4})

        # Perform backward elimination
        p_max = 1
        while p_max > p_cutoff:
            p = model.pvalues
            p_max = max(p)
            feature_max_p = p.idxmax()

            if p_max > p_cutoff:
                X = X.drop(feature_max_p, axis=1)
                if len(X.columns) == 0:
                    break
                model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={'maxlags': 4})

        return model

    if isinstance(input_df, pd.DataFrame):
        return fit_model(input_df)

    elif isinstance(input_df, dict):
        models = {}
        for split, data in input_df.items():
            if 'train' in data:
                model_name = f"{split}_model"
                models[model_name] = fit_model(data['train'])
            else:
                raise ValueError(f"No 'train' dataset found in {split}.")
        return models

    else:
        raise TypeError("Input must be a pandas DataFrame or a dictionary of DataFrames.")

def predict(data, models):
    """
    Creates a dataframe or dictionary of dataframes with the actual and predicted values of y.

    Parameters:
    - data (pd.DataFrame or dict): The dataframe or dictionary of dataframes to be used for prediction.
    - models (statsmodels.regression.linear_model.RegressionResultsWrapper or dict): The fitted model(s).

    Returns:
    - A dataframe or dictionary of dataframes with the actual and predicted values of y.
    """

    def make_prediction(df, model):
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

    if isinstance(data, pd.DataFrame) and not isinstance(models, dict):
        return make_prediction(data, models)

    elif isinstance(data, dict) and isinstance(models, dict):
        predictions = {}
        for split, model in models.items():
            split_key = split.replace('_model', '')  # Get the original split key
            if 'test' in data[split_key]:
                predictions[split] = make_prediction(data[split_key]['test'], model)
            else:
                raise ValueError(f"No 'test' dataset found in {split_key}.")
        return predictions

    else:
        raise TypeError("Data and models must be either both pandas DataFrames or both dictionaries.")


def oos_summary_stats(data):
    """
    Calculate out-of-sample summary statistics for a regression model.

    Parameters:
    data (DataFrame or dict): DataFrame containing the actual and predicted values,
    or a dictionary of DataFrames with actual and predicted values.

    Returns:
    dict: Dictionary containing the out-of-sample summary statistics.
    """
    import numpy as np
    from sklearn.metrics import r2_score

    def calculate_stats(df_pred):
        y_test = df_pred['y_actual']
        y_pred = df_pred['y_pred']

        # Calculate statistics
        oos_stats = {
            'oos_r2': r2_score(y_test, y_pred),
            'oos_mae': np.mean(np.abs(y_test - y_pred)),
            'oos_mse': np.mean((y_test - y_pred) ** 2)
        }
        oos_stats['oos_rmse'] = np.sqrt(oos_stats['oos_mse'])

        # Find the min and max date of the test set
        oos_stats['start_date'] = df_pred.index.min().strftime("%d/%m/%Y")
        oos_stats['end_date'] = df_pred.index.max().strftime("%d/%m/%Y")

        return oos_stats

    if isinstance(data, pd.DataFrame):
        return calculate_stats(data)
    elif isinstance(data, dict):
        results = {}
        for split_name, split_data in data.items():
            test_df_key = f"{split_name}_test"
            if test_df_key in split_data:
                results[split_name] = calculate_stats(split_data[test_df_key])
            else:
                raise KeyError(f"'{test_df_key}' not found in '{split_name}'.")
        return results
    else:
        raise TypeError("Input must be a pandas DataFrame or a dictionary of DataFrames.")


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
