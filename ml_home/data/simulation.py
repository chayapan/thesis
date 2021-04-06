import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.generator import dgf10, dgf11
from data.generator import plot_line, gd2df, add_noise

SIM_DAYS = 1200  # 200 history + 250 + 3*250

class TimeSeries():
    def __init__(self, days, generator, g_params={}):
        self.time_start = datetime.now()
        self.time_end = self.time_start + timedelta(days=days)  # Fixed two dates.
        
        self.days = days
        self.g = generator
        self.x , self.y = self.g(days=days, with_noise=True, **g_params)
        
        self.val_start = self.y[0]
        self.val_end = self.y[-1]
    def get_return(self):
        """Logarithmic return """
        return np.log(self.val_end/self.val_start) * 100
    def get_return_pct(self):
        return (self.val_end - self.val_start) / self.val_start * 100
    @property
    def df(self):
        return gd2df(self.x, self.y)
    def plot(self):
        plot_line(self.x, self.y, xlim=(-10, self.days+10)) # xlim, ylim

        
def make_dataset_linear(dataset, days=SIM_DAYS, plot=True):
    """Returns dictionary with generated data set and statistics data frame."""
    assert type({}) == type(dataset) # Must be a dictionary
    if days:
        SIM_DAYS = days
    
    # b is the starting value. This normally can't be zero.
    b = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
         100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
         100, 100, 100, 100, 100, 100, 100, 100, 100, 100 ]

    # Slope, shift horizon, shift vertical
    m = [0.0, 0.01, 0.15, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 
         0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.2 , 0.09, 0.095,
         0.1, 0.101, 0.103, 0.105, 0.107, 0.11, 0.115, 0.117, 0.12, 0.13]

    # Shift horizontal (along x-axis)
    h = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    # h = [0,  50,  100,  0,  50, 100] 
    # h = [0,  50,  100,  0,  50, 100]

    # Shift vertical (along y-axis)
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    # v = [0, 0, 0, 0, 10, 10, 10]
    # v = [0,  0,  0,  0,  0, 0]

    df = pd.DataFrame()

    for i in range(len(m)):
        b_i=b[i]
        m_i=m[i]
        h_i=h[i]
        v_i=v[i]
        tsi = TimeSeries(days=SIM_DAYS, generator=dgf10, 
                         g_params={'b':b_i, 'm':m_i, 'h':h_i, 'v':v_i})
        if plot:
            tsi.plot()
        
        r = tsi.get_return_pct()
        rl = tsi.get_return()

        data = (tsi.g.__name__, tsi.df, r)
        series_id = "linear_%s" % str(i+1)
        dataset[series_id] = data

        df = df.append({'series_id': series_id, 'slope': m_i, 'shift_x': h_i, 'shift_y': v_i,
                   'return': r, 'return_pct': rl, 'days': tsi.days,
                   'val_start': tsi.val_start, 'val_end':tsi.val_end,
                   'variance': np.var(tsi.y) }, 
                       ignore_index=True)
    if plot:
        df[['return', 'return_pct']].plot.bar()
    df['return_annualized_pct'] = (df['val_end'] - df['val_start']) / df['val_start'] * 100 / SIM_DAYS * 250
    return dataset, df

def make_dataset_exponential(dataset, days=SIM_DAYS, plot=True):
    """Returns dictionary with generated data set and statistics data frame."""
    assert type({}) == type(dataset) # Must be a dictionary
    if days:
        SIM_DAYS = days
    
    # b is the starting value. This normally can't be zero.
    b = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
         100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
         100, 100, 100, 100, 100, 100, 100, 100, 100, 100 ]

    # Growth, the 'a' param: default 1.5 or 1.8
    a = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2,
         1.22, 1.25, 1.28, 1.3, 1.32, 1.34, 1.35, 1.36, 1.37, 1.38,  
         1.39, 1.4, 1.41, 1.42, 1.43, 1.44, 1.45, 1.47, 1.5, 1.55]

    # Shift horizontal (along x-axis)
    h = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    # h = [0,  50,  100,  0,  50, 100] 
    # h = [0,  50,  100,  0,  50, 100]

    # Shift vertical (along y-axis)
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    # v = [0, 0, 0, 0, 10, 10, 10]
    # v = [0,  0,  0,  0,  0, 0]

    df = pd.DataFrame()

    for i in range(len(a)):
        b_i=b[i]
        a_i=a[i]
        h_i=h[i]
        v_i=v[i]
        tsi = TimeSeries(days=SIM_DAYS, generator=dgf11, 
                         g_params={'a': a_i, 'b':b_i, 'h':h_i, 'v':v_i})
        if plot:
            tsi.plot()
        r = tsi.get_return_pct()
        rl = tsi.get_return()

        data = (tsi.g.__name__, tsi.df, r)
        series_id = "growth_%s" % str(i+1)
        dataset[series_id] = data

        df = df.append({'series_id': series_id, 'a': a_i, 'shift_x': h_i, 'shift_y': v_i,
                   'return': r, 'return_pct': rl, 'days': tsi.days,
                   'val_start': tsi.val_start, 'val_end':tsi.val_end,
                   'variance': np.var(tsi.y) }, 
                       ignore_index=True)
    if plot:
        df[['return', 'return_pct']].plot.bar()
    df['return_annualized_pct'] = (df['val_end'] - df['val_start']) / df['val_start'] * 100 / SIM_DAYS * 250
    return dataset, df