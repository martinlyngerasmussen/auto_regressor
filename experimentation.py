

import pandas as pd
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit¶



def data_preparation(file_location, lags):
    # assumes that the variables are stationary
    # Import the dataset
    dataset = pd.read_csv(file_location)
    X = dataset.iloc[:, :-1].values
    X = X.loc[~X.isna()]
    y = dataset.iloc[:, -1].values
    y = y.loc[~y.isna()]

    ### df['Date'] = pd.to_datetime(df['Date'], format = "%d/%m/%Y").dt.date



    // use TimeSeriesSplit to split the data
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #### REDO TO ADD LAGS TO THE SPLITTED DATA
    ## create lags of the variables
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    for i in range(1, lags+1):
        X = pd.concat([X, X.shift(i)], axis=1)
        y = pd.concat([y, y.shift(i)], axis=1)
    X = X.iloc[lags:, :]
    y = y.iloc[lags:, :]

    return X, y


def regression_OLS():
    ## Import the dataset
    df = read_data()
    X = df["X"]
    y = df["y"]

    ## Fit the model
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) # make the predictions by the model




## https://gist.github.com/vb100/177bad75b7506f93fbe12323353683a0
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
