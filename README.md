# masters_project

Scripts used to analyse geophysical dataset and produce a prediction map of gold.

## datavis

This script uses the seaborn package to visualize various attributes of the data to prepare for data cleaning and data preparation.

## data_eng

This script engineers new features by calculating statistics using a rolling window. The original features are made up geophysical data and rasterized geological map.

## ml_workspace_regional

This script encompasses the process of sampling the data (over and under sampling) to resolve the data imbalance. The predicition maps are produced using both Random Forest and extreme gradient boosting (XGBoost). The two models are compared for accuracy and tendency to overfit.

## param_opt

Using RandomizedSearchCV package of scikit-learn to optimize the hyperparameters of the XGBoost algorithm.
