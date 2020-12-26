import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os, os.path
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Load data series and clean-up missing values
def prep_series_for_XGBoost(df):
    data = df
    # Target: Tomorrow's closing price
    data.loc[:,("Tomorrow")] = data["Price"].shift(-1)
    # data.dropna(inplace=True)

    # Past: Historical data



    # Target: Future time
    data.loc[:,("1-day-ahead")] = data["Price"].shift(-1)
    data.loc[:,("3-day-ahead")] = data["Price"].shift(-3)
    data.loc[:,("5-day-ahead")] = data["Price"].shift(-5)
    data.loc[:,("10-day-ahead")] = data["Price"].shift(-10)
    data.loc[:,("15-day-ahead")] = data["Price"].shift(-15)
    data.loc[:,("30-day-ahead")] = data["Price"].shift(-30)

    # Initial Setup XGBoost.
    # Run with 5 TI: SMA WMA ADOSC ATR RSI
    # Only use Closing price
    df = data[['SMA','WMA','ADOSC','ATR','RSI','1-day-ahead']]
    df
