import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from prettytable import PrettyTable
from datetime import datetime
from stargazer.stargazer import Stargazer
import matplotlib.pyplot as plt

def import_dataset(file_location):
    """
    Imports the dataset and converts the 'date' column to datetime format.

    Parameters:
    - file_location (str): Path to the dataset file.

    Returns:
    - pd.DataFrame: Dataset with 'date' as datetime and set as index.
    """
    try:
        dataset = pd.read_csv(file_location)
        dataset['date'] = pd.to_datetime(dataset['date'], infer_datetime_format=True).dt.date
        dataset.set_index('date', inplace=True)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def calculate_vif_and_reduce_features(df, vif_cut_off=5):
    """
    Calculates the Variance Inflation Factor (VIF) and removes features with high VIF.

    Parameters:
    - df (pd.DataFrame): Dataset with features.
    - vif_cut_off (float): Threshold for the VIF.

    Returns:
    - pd.DataFrame: Reduced feature set.
    """
    X = df.iloc[:, 1:]  # Exclude target variable 'y'
    while True:
        vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        max_vif = max(vifs)
        if max_vif < vif_cut_off:
            break
        max_vif_index = vifs.index(max_vif)
        X = X.drop(X.columns[max_vif_index], axis=1)
    return pd.concat([df.iloc[:, 0], X], axis=1)  # Add 'y' back

def create_lagged_features(df, lags=5):
    """
    Creates lagged features for a given dataframe.

    Parameters:
    - df (pd.DataFrame): Dataset.
    - lags (int): Number of lags to create.

    Returns:
    - pd.DataFrame: Dataframe with lagged features.
    """
    columns = df.columns
    for lag in range(1, lags + 1):
        for col in columns:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df.dropna(inplace=True)  # Drop rows with NaN values after shifting
    return df

def prepare_data_for_splits(df, lags, splits, train_share):
    """
    Prepares the data for cross-validation splits.

    Parameters:
    - df (pd.DataFrame): Dataset.
    - lags (int): Number of lags.
    - splits (int): Number of splits for cross-validation.
    - train_share (float): Proportion of data for training.

    Returns:
    - dict: Dictionary of train-test splits.
    """
    split_dfs = np.array_split(df, splits)
    splits_dict = {}
    for i, split_df in enumerate(split_dfs):
        split_df = create_lagged_features(split_df, lags)
        train_size = int(len(split_df) * train_share)
        splits_dict[f'split_{i+1}'] = {
            'train': split_df[:train_size],
            'test': split_df[train_size:]
        }
    return splits_dict

def perform_regression_analysis(splits_dict, p_cutoff):
    results_dict = {}
    for split, data in splits_dict.items():
        train_df, test_df = data['train'], data['test']
        y_train, X_train = train_df.iloc[:, 0], train_df.iloc[:, 1:]
        X_train_const = sm.add_constant(X_train, has_constant='add')
        model = sm.OLS(y_train, X_train_const).fit(cov_type="HAC", cov_kwds={'maxlags': 4})

        # Backward elimination
        while max(model.pvalues) > p_cutoff:
            feature_to_remove = model.pvalues.idxmax()
            X_train_const = X_train_const.drop(feature_to_remove, axis=1)
            if len(X_train_const.columns) == 1:  # Stop if only the constant is left
                break
            model = sm.OLS(y_train, X_train_const).fit(cov_type="HAC", cov_kwds={'maxlags': 4})

        # Store results
        results_dict[split] = {
            'model': model,
            'metrics': calculate_metrics(model, test_df, X_train_const.columns)
        }
    return results_dict

def calculate_metrics(model, test_df, model_features):
    y_test = test_df.iloc[:, 0]

    # Create a DataFrame of zeros with the same structure as X_train_const
    X_test = test_df.loc[:, test_df.columns.isin(model_features)]

    print(X_test.columns)
    print(model_features)
    # Fill it with the values from test_df
    X_test.update(test_df)

    y_pred = model.predict(X_test)
    return {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }


def plot_cumulative_model_vs_actual(df, model, last_percent=0.2):
    """
    Plots a chart showing the cumulative factor model vs. actual for the last part of the data.

    Parameters:
    - df (pd.DataFrame): Dataset.
    - model (statsmodels OLS model): Trained model.
    - last_percent (float): Percent of data to use for the plot (default: 0.2).
    """
    last_n = int(len(df) * last_percent)
    actual = df.iloc[-last_n:, 0]
    predicted = model.predict(sm.add_constant(df.iloc[-last_n:, 1:]))
    plt.plot(actual.index, actual.cumsum(), label='Actual')
    plt.plot(predicted.index, predicted.cumsum(), label='Predicted')
    plt.legend()
    plt.title('Cumulative Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Value')
    plt.show()
