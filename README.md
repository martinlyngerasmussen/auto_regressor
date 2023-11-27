This is a project I am working on in my spare time to help automate model selection for economic and financial time series. The idea is to automate the work flow from dataset to final model(s) used to explain or predict.

The end goal of the project is to fit optimal models automatically based on any given time series data:
1) Model selection: fit multiple models (OLS and, if selected, VAR) across multiple sub-sets of the data. Insignificant (lags) of variables are removed one-by-one for each sub-set of the data. This leaves one optimal model for each subset of the data.
2) Time series cross validation: for each sub-set (split) of the data, the model is trained on part of the data (e.g. 80%), and the out-of-sample performance of the fitted model is tested for the remainder of the sample (e.g. 20%). Lags are only added once the data has been split into the different slices and each slice has been divided into train and test datasets (to avoid look-ahead bias).
![image](https://github.com/martinlyngerasmussen/auto_econometrics/assets/103667557/93e230bc-075b-41a7-ad4b-61e9e0f6b49f)

Source: https://www.kaggle.com/code/cworsnup/backtesting-cross-validation-for-timeseries/notebook
  
3) Provide a summary of how the different models that have been fitted perform out-of-sample for the given time period. This should provide a quick and easily digestible overview of the models across the different time periods: significant (lags of) variables, summary stats (out of sample: R2, RMSE, etc)
4) Make the insights actionable: model averaging for optimal prediction. Automatically predict and/or show model-implied value for the latest available data point. 

To do list and thoughts:
- Work on #3-4
- Add VAR functionality to #1. Add test to determine whether VAR should be used. 
- Use heteroskedasticity and autocorrelation consistent standard errors for model selection in #1
- Turn into a package
- Update tests 
