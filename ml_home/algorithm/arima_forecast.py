import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# statsmodels.tsa.arima_model.ARMA and statsmodels.tsa.arima_model.ARIMA have
# been removed in favor of statsmodels.tsa.arima.model.ARIMA (note the .
# between arima and model) and statsmodels.tsa.SARIMAX.


# Load data series and clean-up missing values
def prep_series_for_ARIMA(df):
    """Returns dataframe specific to ARIMA model"""
    # date_time = pd.to_datetime(df.pop('Date'), format='%Y-%m-%d')
    # df['Daily Return'] = np.log(df['Adj Close']/df['Adj Close'].shift())
    # df[['Adj Close', 'Daily Return']]
    # df.dropna(inplace=True)
    # df.head() # Starting data without date column
    if df['Type'][0] == 'StockIndex':
        df['Daily Return'] = df['Change %']
        df['y_true'] = df['Close']
    else:
        df['Daily Return'] = df['DailyReturn']
        df['y_true'] = df['Close']
    df['t'] = df.index

    return df

class Prediction:
    def fit_and_predict(self, df, today, ndays=5, window=100, arima_order=(3,1,0)):
        series_id = df['Ticker'][0]
        # Dataframe supply as argument can be sliced by date index.
        # Prediction starts from first day to the total length of window evaluate.

        idx = df[today:].index[1:1+ndays]
        df_future = df[idx[0]:idx[-1]] # Future. Not include today
        idx = df[:today].index[-1-window:-1] #
        df_past = df[idx[0]:idx[-1]]  # Past. The past not include today.

        df_today = df[today:][0:1]

        df_past = pd.concat([df_past, df_today]) # Past up to today

        # df.sort_index(inplace=True) # TODO: No need. But would be good to enforce check here.

        # Forecast one series

        # ndays=5 # Number of days into the future to predict.
        # 1-day horizon. Look 1-days into the future.
        # window=100 # Days of historical observations to use.
        # window of history looking back
        # arima_order=(3,1,0)

        # the total period need for evaluation: history + future horizon.
        eval_window = window + ndays

        # Number of observations in time-series.
        all_obs = len(df)

        # fit_df = df[0:eval_window]
        fit_df = df # All values

        # history = fit_df[:window]['y_true']
        history = df_past['y_true']
        model_predictions = [] # TODO: fill-me
        model_error = []
        model = ARIMA(history, order=arima_order) # (3,1,0) by default
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(ndays)
        # output = 1. forecasts 2. std.error 3. confident intervals

        window, ndays, output

        # pred_df = fit_df[['y_true']][window:window+ndays]
        pred_df = df_future[['y_true']]
        pred_df['y_pred'] = output[0]
        pred_df['std_err'] = output[1]

        # pred_df['confident_lower'] = output[2][0][0]  # TODO
        # pred_df['confident_upper'] = output[2][0][1]  #

        # Save parameters
        self.series_id = series_id
        self.arima_order = arima_order
        self.fit_df = fit_df
        self.df_past = df_past
        self.today = today
        self.df_future = df_future
        self.pred_df = pred_df
        self.window = window
        self.ndays = ndays

    @property
    def forecast_horizon(self):
        """The day of forecast value."""
        return self.pred_df.last('1d').index.strftime('%Y-%m-%d').values[0]

    @property
    def prediction_error(self):
        # Calculate total error.
        res = self.pred_df.last('1d')
        # The error is the difference between prediction and actual value in absolute term.
        err = abs(res['y_pred'] - res['y_true']) .values[0]

        # The percentage error is the absolute error divide by the actual error.
        pct_error = err / res['y_true'].values[0]

        predict = res['y_pred'].values[0]
        actual = res['y_true'].values[0]

        out = {'predict_error': err, 'pct_error': pct_error, 'predict': predict, 'actual': actual}
        return out

    def plot(self):
        window, ndays = self.window, self.ndays
        fit_df, pred_df = self.fit_df, self.pred_df
        df_past, df_future = self.df_past, self.df_future
        fig, ax = plt.subplots()
        fit_df[['y_true']].plot(ax=ax, figsize=(18,8), color='Blue')
        # fit_df[['y_true']][window:window+ndays].plot(ax=ax, marker='o', color='Orange')
        df_future[['y_true']].plot(ax=ax, marker='o', color='Orange')
        # fit_df[['y_true']][0:window].plot(ax=ax, marker='x', linestyle='dashed', color='Green')
        df_past[['y_true']].plot(ax=ax, marker='x', linestyle='dashed', color='Green')
        pred_df[['y_pred']].plot(ax=ax, marker='x', color='Red')

        ax.legend(['Truth', 'Test', 'Train', 'Predict'])

        train_from = fit_df.first('1d').index.strftime('%Y-%m-%d').values[0]
        train_to = fit_df[window:window+1].index.strftime('%Y-%m-%d').values[0]
        pred_from = pred_df.first('1d').index.strftime('%Y-%m-%d').values[0]
        pred_to = pred_df.last('1d').index.strftime('%Y-%m-%d').values[0]

        error = self.prediction_error
        err, pct_err = error['predict_error'], error['pct_error']
        pred, actual = error['predict'], error['actual']

        title = """{series_id} :: ARIMA{order} history={history}d forecast={ndays}d :: pred={pred:.2f} actual={actual:.2f} err={err:.4f} %err={pcterr:.4f} \n
                    Train({train_from}:{train_to}) Predict({pred_from}:{pred_to})""".format(
                        series_id=self.series_id, order=self.arima_order, history=window, ndays=ndays,
                        pred=pred, actual=actual, err=err, pcterr=pct_err,
                        train_from=train_from, train_to=train_to, pred_from=pred_from, pred_to=pred_to
                    )

        plt.title(title)
