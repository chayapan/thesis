import numpy as np 
from numpy.linalg import LinAlgError
import pandas as pd 
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

import warnings
warnings.filterwarnings("ignore") ##### Dangerous!!!

import matplotlib.pylab
from statsmodels.tsa import stattools
# from pandas_datareader import data

# Load data series and clean-up missing values

csv_path = 'https://raw.githubusercontent.com/chayapan/thesis/master/historical/ADVANC.csv'
df = pd.read_csv(csv_path)
date_time = pd.to_datetime(df.pop('Date'), format='%Y-%m-%d')
df['Daily Return'] = np.log(df['Adj Close']/df['Adj Close'].shift())
df[['Adj Close', 'Daily Return']]
df.dropna(inplace=True)
df.head() # Starting data without date column
df['t'] = df.index
df['y_true'] = df['Adj Close'] ## Too reconsider feature for model

# Check Autocorrelation
plt.figure()
lag_plot(df['y_true'], lag=3)
plt.title('HS-1 - Autocorrelation plot with lag = 3')
# plt.show()


# Number of observations in time-series.
all_obs = len(df)

# Requires: all_obs, df

def collect_forecast_perf(df, all_obs, ndays=1, window=100, arima_order=(3,1,0)):
  # window = 100  # window of history looking back
  # ndays = 1     # 1-day horizon. Look 1-days into the future.
  eval_window = window + ndays # the total period need for evaluation: history + future horizon.

  #############################
  # Roll - generate batches
  # ws = Window Start

  prediction_batch = {}

  for ws in range(0, all_obs - eval_window ):
    sample_window = (ws, ws + eval_window)
    # print(sample_window)

    # Batch to test prediction
    prediction_batch[sample_window] = df[sample_window[0]:sample_window[1]]

  idx = list(prediction_batch.keys())

  ########################

  ## Run batch

  rcParams['figure.figsize'] = 15, 6
  result = pd.DataFrame()
  pred_result = pd.DataFrame()

  # To do quick check, use 50-100 here.
  # Or use range that is random from index to prediction batch
  # 2. Sampling with count
  # for k in random.sample(idx,200):  # Choose 200 batches from series by sampling the index

  # 1. Go thorough all series
  for c in range(0, all_obs - eval_window ): # Chose 0 - 1000 of all obs in time series
    k = idx[c]
    sw = prediction_batch[k]
    sw
    #######
    actual_window = window + ndays

    actual = sw[:actual_window]['y_true'] # For comparison
    history = sw[:window]['y_true']

    try:
    
        model_predictions = [] # TODO: fill-me
        model_error = []
        model = ARIMA(history, order=arima_order) # (3,1,0) by default
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(ndays)
        output # 1. forecasts 2. std.error 3. confident intervals
        ######
        yhat = output[0]
        err = output[1]
        con_inter = output[2]
        yhat[ndays-1] # Prediction of n days in the future

        est_val = np.append(history.values, yhat)
        est_val # Series of estimated values -- actual plus forecast

        window_range = actual.index
        window_range
        ######
        plt.ylabel('Adj. Close Price')
        plt.xlabel("Day")
        plt.plot(window_range, est_val, color='blue', marker='o', linestyle='dashed',label='Predicted Value')
        plt.plot(window_range, actual, color='red', label='Actual Value')
        plt.plot([k[0] + i+1 for i in range(window, actual_window)], yhat, color='green', marker='x', linestyle='dashed',label='Forecast')
        plt.title("%s-day Forecast" % ndays)
        # _ =  plt.legend()

        price_p = est_val[-1] # Predicted value from ARIMA(3,1,0).
        price_a = actual.to_numpy()[-1] # Actual value.
        err = ((price_p - price_a) / price_a) # Error (percentage)


        # Defitiion of percentage error
        # https://www.mathsisfun.com/numbers/percentage-error.html
        # By definition, divide the difference between estimate and actual by the actual value.

        # print("Study Window",k)
        # Predict, Actual, Error
        # print(price_p, price_a, err)


        pred_result = pred_result.append({'Period': str(k), 
                    'Predict':price_p, 'Actual': price_a,
                    'Present Date': k[0]+window,
                    'Error (Pct)': err}, 
                    ignore_index=True)
    
    except Error as e:
        print(str(e))
        pass

  # Concluding

  # MAPE
  metric = abs(pred_result['Actual'] - pred_result['Predict']) / pred_result['Actual']
  mape = metric.sum() / metric.count()
  mape
  print("MAPE: %0.3f" % mape)
  print("Batches Tried: %0d" % pred_1d['Predict'].count())
  
  return pred_result

HPARAM = [(3,1,0), (4,1,0)]
HISTORY = [50, 100] # 10, 15, 30, 
HORIZON = [1, 3, 5, 7, 14, 30]

def collect_error_data():
    
    for hist_window in HISTORY:
        for forecast_days in HORIZON:
            print("History: %s days Forecast: %s days" % (hist_window, forecast_days))
            pred_res = collect_forecast_perf(df, all_obs, ndays=forecast_days, window=hist_window)

if __name__ == '__main__':
    collect_error_data()