# Autoregressor: simple and robust time series model selection
This notebook shows how to use auto_regressor.py, a very simple Python function that allows the user to fit an OLS with lagged variables.

The function does the following:
**Removes colinear variables** by removing regressors with high variance inflation factors (VIF).

**Model Selection:** the function automatically finds the best model available for the specified (lags of) variables. More specifically, for a given set of variables (y and set of X), the code does backward selection: remove the (lag of) variable with the highest p-value, then re-run the model, remove the least significant variable. This process is repeated until the p-values of all (lags of) variables are below the specified threshold (*p_cutoff* default is 0.05).

**...across multiple splits of the data:** to ensure that model selection is robust, the model is fit across multiple sub-samples ('splits') of the data. This means that the model is fitted separately for each split. Please note that the data is split before lags are added to avoid look-ahead bias.

Each split is divided into a 'training set', on which the model is fitted as well as a 'test set', on which the model is tested (but not fitted). *train_share* is 0.8 by default.

**The output of the function is the following (in order of output):**
1. Out-of-sample model performance across splits: R2, MAE, MSE, RMSE, start date and end date
2. A Pandas dataframe with the out-of-sample values across each split. This allows us to answer: how well does the model perform out-of-sample across different periods?
3. Model summary for each training split as well as the full sample: coefficients etc.
4. A Pandas dataframe that contains the model from each split, *fitted to the* **full dataset**. This answers the question: how well does a model fit on 2010-2015 data perform during 2021-2023?
