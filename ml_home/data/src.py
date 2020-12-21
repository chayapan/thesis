import os
import numpy as np
import pandas as pd
from data.snapshot import industries, sectors

if not 'DATA_HOME' in os.environ:
    raise Exception("DATA_HOME not defined.")
data_home = os.environ["DATA_HOME"]

"""Example:

# 2015
period_start='2015-01-01'
period_end='2015-12-31'
data1 = stockdata_fetch_local(period_start, period_end)

data1.pivot_table(values='Close', index='Date', columns=['Industry', 'Sector', 'Ticker']).tail()


Example 2:

from dataset import Yr2014
from data.src import stockdata_fetch_local
df = stockdata_fetch_local(Yr2014.dt_start, Yr2014.dt_end, ["SCB","BBL","KBANK"])
tbl = df.pivot_table(values='Close', index='Date', columns=['Industry', 'Sector', 'Ticker'])
df[df['Ticker']=='SCB']
df[df['Ticker']=='KBANK']
tbl.plot()

"""

def read_csv_from_historical_folder(symbol, index_col=0):
    return pd.read_csv(os.path.join(data_home,"historical", "%s.csv" % symbol), index_col=0)

def to_float(x):
    """Formating for index data."""
    if isinstance(x, str):
        x = x.replace(',','')
        if 'K' in x:
            p1, p2 = x.split('K')
            return float(p1) * 1000
        if 'M' in x:
            p1, p2 = x.split('M')
            return float(p1) * 1000000
        if 'B' in x:
            p1, p2 = x.split('B')
            return float(p1) * 1000000000
    return float(x) # already float!

def stockdata_fetch_local(period_start, period_end, symbols, preprocessing=[]):
    data = {}
    for s in symbols:
        # Read series
        df = read_csv_from_historical_folder(s, index_col=0)
        # Not just using index_col=0. Also set index manually.
        df.index = pd.to_datetime(df.index)
        # Add column ticker
        df.loc[:,('Date')] = df.index
        df['Ticker'] = s

        # Add sector column
        for k, sect in sectors.items():
            if s in sect:
                df['Sector'] = k

        # Add industry column
        for k, indus in industries.items():
            if s in indus:
                df['Industry'] = k

        # Add return column: Return from yesterday's closing.
        df["DailyReturn"] = np.log(df["Close"].shift(1)/df["Close"])

        # TODO
        #  Add feature TI from shared library.

        # FF-24 Daily Price Variation
        df["DailyPriceVariation"] = (df["High"] - df["Low"]) / df["Close"]

        # FF-25 ~ use VWAP?
        df['PriceStDev-5d'] = df['Close'].rolling(5).std() # Short cut available from pandas

        # Select only defined range
        df = df[period_start:period_end]

        # Add to dictionary
        # df.dropna(inplace=True) # FIXME: Check drop na location. Or shoud we raise error?
        data[s] = df

    rows = []
    for k, d in data.items():
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    return df

def indexdata_fetch_local(period_start, period_end, symbols):
    """

    df[df['Ticker']=='SET50']
    """
    data = {}
    for s in symbols:
        # Read series
        dataset = read_csv_from_historical_folder(s) # Notice we didn't use index_col here.
        # Format data frame
        dataset = dataset.set_index(pd.to_datetime(dataset.index))
        dataset.sort_index(inplace=True)

        # Set column name. Also use same convention as stock
        dataset['Open'] = dataset['Open'].apply(to_float)
        dataset['High'] = dataset['High'].apply(to_float)
        dataset['Low'] = dataset['Low'].apply(to_float)
        dataset['Close'] = dataset['Price'].apply(to_float)
        dataset['Volume'] = dataset['Vol.'].apply(to_float)
        dataset['Change %'] = dataset['Change %'].apply(lambda x: float(x.replace('%', '')))

        # Add column ticker
        dataset.loc[:,('Date')] = dataset.index
        dataset['Ticker'] = s

        # Select only defined range
        dataset = dataset[period_start:period_end]

        # Add to dictionary
        data[s] = dataset
    rows = []
    for k, d in data.items():
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    return df


def stockdata_fetch_historical(period_start, period_end, symbols, local=True):
    """Fetch historical data. Use DATA_HOME folder if local is True."""
    return stockdata_fetch_local(symbols)
