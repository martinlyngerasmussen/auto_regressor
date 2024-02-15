# Autoregressor: simple and robust time series model selection
auto_regressor.py automates the process of fitting and using a robust time series regression (OLS) model with lags. Plug in your data, run the script and the code will fit the best model + provide predictions.


## Autoregressor: Simple and Robust Time Series Model Selection

This is what the code does:

1. **Loading Your Dataset:**
  - The `load_df` function is employed to import your dataset. It's crucial that your dataset contains either a datetime column or a column explicitly named 'date' or 'Date'.

2. **Preprocessing and Analysis:**
  - Call `remove_colinear_features` to weed out collinear variables, ensuring model integrity.
  - Conduct an exploratory analysis to scrutinize relationships within your data and inspect for any non-linear dynamics between variables.

3. **Creating Splits and Fitting Models:**
  - Use `create_splits` to segment your data into distinct training and testing sets, integrating lagged features for thorough analysis.
  - Apply `regression_OLS` to fit models, selectively honing in on significant features through an automated process.

4. **Evaluation and Prediction:**
  - Gauge model performance using `oos_summary_stats`, allowing for comparison of models across different data splits.
  - Facilitate both in-sample and out-of-sample predictions with `fit_and_predict`, providing visual representation for in-depth evaluation.

Optional Features:
- Non-linearity Detection: Automatically detects and adjusts for non-linear relationships between variables, enhancing model accuracy.
- Customization Options: Offers flexibility in adjusting the number of lags, splits, training share, VIF threshold, and p-value cutoff for tailored analysis.
