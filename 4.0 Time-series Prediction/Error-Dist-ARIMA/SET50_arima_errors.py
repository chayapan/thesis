import time
from multiprocessing import Pool
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
import logging
logging.basicConfig(filename='collect_perf.log', level=logging.DEBUG)
rcParams['figure.figsize'] = 15, 6

import warnings
warnings.filterwarnings("ignore") ##### Dangerous!!!

import matplotlib.pylab
from statsmodels.tsa import stattools
# from pandas_datareader import data

# Load data series and clean-up missing values
def get_series(ticker):
    # csv_path = 'https://raw.githubusercontent.com/chayapan/thesis/master/historical/ADVANC.csv'
    logging.info("Downloading %s" % ticker)
    csv_path = 'https://raw.githubusercontent.com/chayapan/thesis/master/historical/%s.csv' % ticker
    df = pd.read_csv(csv_path)
    date_time = pd.to_datetime(df.pop('Date'), format='%Y-%m-%d')
    df['Date'] = date_time
    df['Daily Return'] = np.log(df['Adj Close']/df['Adj Close'].shift())
    df[['Adj Close', 'Daily Return']]
    df.dropna(inplace=True)
    df.head() # Starting data without date column
    df['t'] = df.index
    df['y_true'] = df['Adj Close'] ## Too reconsider feature for model

    # Check Autocorrelation
    # plt.figure()
    # lag_plot(df['y_true'], lag=3)
    # plt.title('HS-1 - Autocorrelation plot with lag = 3')
    # plt.show()

    return df


def summarize_experiment_batch(batch_output):
    """Summarize data from each treatment into a row to add to model selection dataset."""
    
    # TODO/FIXME
    
    # store report data
    exprData = pd.DataFrame()
    pred_res = batch_output
    dat = {
        'series_id': ticker,
        # 'forecast_setting': 'X_X_X_X_X_X',
        'history_window': hist_window,
        'forecast_days': forecast_days,
        'arima_order': str(morder),
        # Result from dataframe
        'MAPE': pred_res['MAPE'][0],
        'MSE': pred_res['MSE'][0],
        'batches_tried': pred_res['batches_tried'][0],
        'observations': pred_res['observations'][0],
        'error_mean': pred_res[['Error (Pct)']].describe().T['mean'].values[0],
        'error_std': pred_res[['Error (Pct)']].describe().T['std'].values[0],
        'error_min': pred_res[['Error (Pct)']].describe().T['min'].values[0],
        'error_max': pred_res[['Error (Pct)']].describe().T['max'].values[0]
    }
    # How log did it take to run this trial
    elapse = time.time() - start_time
    print("Treatment took %0.2f sec." % elapse)
    dat['experiment_took_sec'] = elapse
    # Add the descriptive stats for error percentage column
    exprData = exprData.append(dat, ignore_index=True)


# Requires: all_obs, df

def collect_forecast_perf(series_id, df, ndays=1, window=100, arima_order=(3,1,0)):
  """Measure ARIMA model performance by running the model against historical data. Returns dataframe.  

    Parameters
    ----------
    series_id: str
        The name or ID of time-series to apply forcast and collect accuracy metric.
    df : pd.DataFrame
        The time-series data with columns approprite for use: High, Low, Open, Close, Volume, Adj Close, Daily Return, t, y_true.
    ndays : int, optional
        Days in the future to forecast, the forecast horizon.
    window: int, optional
        Historical data to use, the historical window. Default is 100 days. 
    arima_order: tuple, optimal
        The (p,d,q) order for ARIMA model. Default is (3,1,0).

    Returns
    -------
    list
        a list of strings used that are the header columns
  
  """
  # window = 100  # window of history looking back
  # ndays = 1     # 1-day horizon. Look 1-days into the future.
  eval_window = window + ndays # the total period need for evaluation: history + future horizon.

  # Number of observations in time-series.
  all_obs = len(df)
  logging.info("%s - ARIMA%s with %s-day history; %s-day Forecast" % (series_id, str(arima_order), window, ndays))

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
    #####
    # k[0] index the start of history window. 
    # k[1] points to the end of history window.
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
        plt.title("%s - ARIMA%s with %s-day history; %s-day Forecast" % (series_id, str(arima_order), window, ndays))
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
        
        # History window start, end. Prediction made on. Date prediction was made for.
        history_start_day = str(pd.to_datetime(df[['Date']][k[0]:k[0]+1].values[0]).strftime('%Y-%m-%d'))
        history_end_day = str(pd.to_datetime(df[['Date']][k[1]:k[1]+1].values[0]).strftime('%Y-%m-%d'))
        prediction_date = str(pd.to_datetime(df[['Date']][k[1]+1:k[1]+2].values[0]).strftime('%Y-%m-%d'))
        prediction_for_day = str(pd.to_datetime(df[['Date']][k[1]+ndays:k[1]+ndays+1].values[0]).strftime('%Y-%m-%d'))
        
        pred_result = pred_result.append({'Period': str(k),
                    'Predict':price_p, 'Actual': price_a,
                    'Present Date': k[0]+window, # The day we make the prediction
                    'Prediction Made For': k[0]+window+ndays, # The target date
                    'history_window_start': history_start_day, # history window start date
                    'history_window_end': history_end_day, # history window end date
                    'prediction_made_on': prediction_date, # Prediction made on date
                    'prediction_for_day': prediction_for_day, # Prediction for date
                    'Error (Pct)': err},
                    ignore_index=True)

    except Exception as e:
        print(str(e))
        pass

  # 2. Record result
  # Save image: file name: order, history window, forecast horizon
  plt.savefig('img/%s_ARIMA_%s_h%s_f%s.png' % (series_id, str(arima_order), window, ndays))

  # Concluding

  # Distribution of error
  plt.clf()
  ax = pred_result['Error (Pct)'].hist(bins=100)
  plt.title("Distribution of error %s-day forecast %s using %s-day window." % (ndays, series_id, window))
  plt.savefig('img/%s_ARIMA_%s_h%s_f%s_dist.png' % (series_id, str(arima_order), window, ndays))

  # MAPE
  metric = abs(pred_result['Actual'] - pred_result['Predict']) / pred_result['Actual']
  mape = metric.sum() / metric.count()
  mape
  print("MAPE: %0.5f" % mape)
  print("Batches Tried: %0d" % pred_result['Predict'].count())

  # Save out data
  pred_result['MAPE'] = mape
  pred_result['MSE'] = mean_squared_error(pred_result['Actual'], pred_result['Predict'])
  pred_result['batches_tried'] = pred_result['Predict'].count()
  pred_result['observations'] = all_obs


  return pred_result

HPARAM = [(3,1,0), (4,1,0), (3,0,0), (1,0,0), (1,0,1)]
HISTORY = [10, 15, 30, 50, 100] # 10, 15, 30,
HORIZON = [1, 3, 5, 7, 14, 30]



def collect_error_data(ticker='AOT'):
    # ticker = 'AOT'
    df = get_series(ticker)

    # store report data
    exprData = pd.DataFrame()

    for morder in HPARAM:
        for hist_window in HISTORY:
            for forecast_days in HORIZON:
                print("%s :: ARIMA%s History: %s days Forecast: %s days" % (ticker, str(morder), hist_window, forecast_days))
                start_time = time.time()
                pred_res = collect_forecast_perf(ticker, df, ndays=forecast_days, window=hist_window, arima_order=morder)

                dat = {
                    'series_id': ticker,
                    # 'forecast_setting': 'X_X_X_X_X_X',
                    'history_window': hist_window,
                    'forecast_days': forecast_days,
                    'arima_order': str(morder),
                    # Result from dataframe
                    'MAPE': pred_res['MAPE'][0],
                    'MSE': pred_res['MSE'][0],
                    'batches_tried': pred_res['batches_tried'][0],
                    'observations': pred_res['observations'][0],
                    'error_mean': pred_res[['Error (Pct)']].describe().T['mean'].values[0],
                    'error_std': pred_res[['Error (Pct)']].describe().T['std'].values[0],
                    'error_min': pred_res[['Error (Pct)']].describe().T['min'].values[0],
                    'error_max': pred_res[['Error (Pct)']].describe().T['max'].values[0]
                }
                # How log did it take to run this trial
                elapse = time.time() - start_time
                print("Treatment took %0.2f sec." % elapse)
                dat['experiment_took_sec'] = elapse
                # Add the descriptive stats for error percentage column
                exprData = exprData.append(dat, ignore_index=True)
    print("Save collected experimental data for %s." % ticker)
    exprData.to_csv("output/arima_errors_%s.csv" % ticker)

if __name__ == '__main__':
    # collect_error_data('SCB')

    SET50 = ['ADVANC','AOT','BANPU','BAY','BBL','BCP','BEC','BGH', 
             'BH','BIGC','BJC','BLA','BTS','CENTEL','CK','CPALL',
             'CPF','CPN','DELTA','DTAC','EGCO','GLOBAL','GLOW','HMPRO',
             'INTUCH','IRPC','IVL','JAS','KBANK','KTB','LH','MINT',
             'PS','PTT','PTTEP','PTTGC','RATCH','ROBINS','SCB','SCC',
             'SCCC','TCAP','THAI','THCOM','TMB','TOP','TRUE','TTW','TUF','VGI']

    with Pool(16) as p:
        try:
            p.map(collect_error_data, SET50)
        except Exception as e:
            print("Job worker error %s" % str(e))
