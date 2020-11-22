"""ARIMA prediction accuracy estimate for SET50 stocks"""


import multiprocessing as mp
import itertools
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from SET50_arima_errors import get_series, collect_forecast_perf


HPARAM = [(3,1,0), (4,1,0), (3,0,0), (1,0,0), (0,0,3)]
HISTORY = [10, 15, 30, 50, 100]
HORIZON = [1, 3, 5, 7, 14, 30]
DTRANSFORM = ['none', 'minmax', 'standard']


def get_treatment_factors():
    """Returns tuple of (arima_order, history_window, forecast_horizon, data_transform)"""
    treatments = []
    for tfactor in itertools.product(HPARAM, HISTORY, HORIZON, DTRANSFORM):
        treatments.append(tfactor)
    return treatments



def collect_accuracy_metric(work):
    ticker = work[0] # unpack job
    control_factor = work[1] 
    print("Collect metric for series %s. Variation: %s." % (ticker, control_factor))
    
    df = get_series(ticker)
    # store report data
    exprData = pd.DataFrame()
    
    # unpack factors
    morder = control_factor[0]
    hist_window = control_factor[1]
    forecast_days = control_factor[2]
    data_transform = control_factor[3]
    
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
    exprData.to_csv("output/arima_errors_%s.csv" % ticker, mode='a')

if __name__ == '__main__':
    # collect_error_data('SCB')

    SET50 = ['ADVANC','AOT','BANPU','BAY','BBL','BCP','BEC','BGH', 
             'BH','BIGC','BJC','BLA','BTS','CENTEL','CK','CPALL',
             'CPF','CPN','DELTA','DTAC','EGCO','GLOBAL','GLOW','HMPRO',
             'INTUCH','IRPC','IVL','JAS','KBANK','KTB','LH','MINT',
             'PS','PTT','PTTEP','PTTGC','RATCH','ROBINS','SCB','SCC',
             'SCCC','TCAP','THAI','THCOM','TMB','TOP','TRUE','TTW','TUF','VGI']

    control_factors = get_treatment_factors()
    
    # Note about which multiprocessing to use:
    #. https://stackoverflow.com/questions/8533318/multiprocessing-pool-when-to-use-apply-apply-async-or-map
    with mp.Pool(16) as p:
        work = []
        for job in itertools.product(SET50, control_factors):
            work.append(job)
        try:
            p.map(collect_accuracy_metric, work)
        except Exception as e:
            print("Job worker error %s" % str(e))
        p.close()
        p.join()
        
