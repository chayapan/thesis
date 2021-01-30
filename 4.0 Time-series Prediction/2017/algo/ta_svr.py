"""
Rewrite TA_SVR_1_ADVANC.

Predict Adj Close of tomorrow: Adj Close column.

1. Create features
2. Fit model
3. Evaluate
4. Use model to predict one day
5. Evaluation result on test set (0.4 train/test split)
6. Evaluation result on holdout set 1 - 2017
6. Evaluation result on holdout set 2

Use StandardScaler and pipeline. See https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from sklearn.metrics import max_error, median_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import os, os.path
import talib
from talib.abstract import *

def build_forecast_model(ticker, historical_data, train_start, train_end):
    """Build model using data in training period. Return model and metric.
    
    Ex. build_forecast_model(ticker, historical_data, train_start, train_end)
    build_forecast_model("ADVANC", dataset, '2014-01-01', '2016-12-31')
    """
    
    # Training period
    # df = data["ADVANC"]
    # daily = df['2014-01-01':'2016-12-31']
    df = historical_data[ticker]
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    daily = df[train_start:train_end]
    
    # Create input features
    inputs = {
        'open': daily["Open"],
        'high': daily["High"],
        'low': daily["Low"],
        'close': daily["Close"],
        'volume': daily["Volume"]
    }
    
    daily.loc[:,("SMA")] = SMA(inputs, timeperiod=25).tolist()
    daily.loc[:,("WMA")] = WMA(inputs).tolist()
    daily.loc[:,("ADOSC")] = ADOSC(inputs).tolist()
    daily.loc[:,("ATR")] = ATR(inputs).tolist()
    daily.loc[:,("RSI")] = RSI(inputs).tolist()
    
    # Create data label
    # We want X,y before splitting.
    data = daily
    # Target: Tomorrow's closing price
    data["Tomorrow"] = data["Close"].shift(-1)
    data = data.dropna()

    # Features: SMA, ATR, RSI AD

    features = data[["SMA", "WMA", "RSI", "ADOSC", "ATR"]].values
    target = data[["Tomorrow"]].values
    
    # Train/test split
    # Prepare data for training. Split train/test 60/40.
    # TODO: get rid of random_state to check when everything is stable?

    X = features
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Make model and fit
    svr = SVR(kernel='linear')
    model = make_pipeline(StandardScaler(), svr)
    model.fit(X_train, y_train)
    
    # Run validation test during training
    y_pred = model.predict(X_test)
    
    # Get metric: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    maxerr = max_error(y_test, y_pred)
    medaberr = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Metric to report
    metric = {
        'mse': mse, 'max_error': maxerr, 'r2': r2,
        'mae': mae, 'median_absolute_error': medaberr
    }
    
    return model, df, metric
    
def test_forecast_model_with_holdout_set(ticker, model, historical_data, test_start, test_end):
    """
    Ex. 
    test_forecast_model_with_holdout_set(ticker, historical_data, test_start, test_end)
    test_forecast_model_with_holdout_set("ADVANC", historical_data, "2017-01-01", "2017-12-31")
    """
    # Test on holdout set 1
    # holdout = df["2017-01-01":"2017-12-31"]
    holdout = historical_data[test_start:test_end]

    inputs = {
        'open': holdout["Open"],
        'high': holdout["High"],
        'low': holdout["Low"],
        'close': holdout["Close"],
        'volume': holdout["Volume"]
    }

    holdout.loc[:,("SMA")] = SMA(inputs, timeperiod=25).tolist()
    holdout.loc[:,("WMA")] = WMA(inputs).tolist()
    holdout.loc[:,("ADOSC")] = ADOSC(inputs).tolist()
    holdout.loc[:,("ATR")] = ATR(inputs).tolist()
    holdout.loc[:,("RSI")] = RSI(inputs).tolist()
    # Target: Tomorrow's closing price

    holdout["Actual"] = holdout["Close"].shift(-1) # tomorrow's price if known today.
    holdout.head()

    # Tomorrow predict
    holdout = holdout.dropna()
    holdout[0:1]
    # Test predicting on one sample.

    features = holdout[0:1][["SMA", "WMA", "RSI", "ADOSC", "ATR"]].values
    features

    y_pred = model.predict(features)
    y_pred

    # Prediction for the entire holdout set.

    holdout.loc[:,"Predict"] = model.predict(holdout[["SMA", "WMA", "RSI", "ADOSC", "ATR"]].values)

    holdout[["Actual", "Predict"]].plot()
    
    # Save plot
    fig = holdout[["Actual", "Predict"]].plot(title="%s %s - %s" % (ticker, test_start, test_end), 
                                                      figsize=(12, 8), fontsize=26).get_figure()
    fig.savefig("results/%s_%s_%s.png" % (ticker, test_start, test_end))
    
    # Get metric: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    mse = mean_squared_error(holdout[["Actual"]].values, holdout[["Predict"]].values)
    mae = mean_absolute_error(holdout[["Actual"]].values, holdout[["Predict"]].values)
    maxerr = max_error(holdout[["Actual"]].values, holdout[["Predict"]].values)
    medaberr = median_absolute_error(holdout[["Actual"]].values, holdout[["Predict"]].values)
    r2 = r2_score(holdout[["Actual"]].values, holdout[["Predict"]].values)
    
    # Metric to report
    metric = {
        'mse': mse, 'max_error': maxerr, 'r2': r2,
        'mae': mae, 'median_absolute_error': medaberr
    }
    
    return holdout, metric
