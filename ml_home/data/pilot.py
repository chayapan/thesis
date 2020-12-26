from data.src import stockdata_fetch_local, indexdata_fetch_local
from dataset import Yr2014, Yr2019
import os, os.path
import pandas as pd

DEBUG = True

sectors = {
    "Financials" : ["SCB", "KBANK"],
    "Services" : ["AOT", "BTS"],
    "Property & Construction" : ["AP", "LH"],
    "Agro & Food Industry" : ["CPF", "KSL"],
    "Resources" : ["PTT", "RATCH"]
}

industries = {
    "Banking" : ["SCB", "KBANK"],
    "Transportation & Logistics" : ["AOT", "BTS"],
    "Property Development" : ["AP", "LH"],
    "Food & Beverage" : ["CPF", "KSL"],
    "Energy & Utilities" : ["PTT", "RATCH"]
}

stock_indices = ["SET", "SET50", "SET100"]


def get_pilot_stocks():
    stocks = ["SCB","KBANK","AOT","BTS","AP","LH","CPF","KSL","PTT","RATCH"]
    df = stockdata_fetch_local(Yr2014.dt_end, Yr2019.dt_end, stocks)
    return df

def get_pilot_indices():
    df = indexdata_fetch_local(Yr2014.dt_end, Yr2019.dt_end, ["SET","SET50","SET100"])
    return df

DATA_HOME = os.environ['DATA_HOME']
PILOT_DATA_HOME = os.path.join(DATA_HOME, 'pilot.data')

def write_piot_series(ticker, dataframe, folder=PILOT_DATA_HOME):
    dataframe.to_csv(os.path.join(folder,'%s.csv' % ticker))
    if DEBUG:
        print("Wrote: %s pilot data series with %s observations." % (ticker, len(dataframe)))
    return dataframe

def build_pilot_dataset():
    """See PilotData.ipynb"""
    all_tickers = []
    df_stocks = get_pilot_stocks()
    df_indices = get_pilot_indices()

    tbl_stock = df_stocks.pivot_table(values='Close', index='Date', columns=['Ticker'])
    tbl_index = df_indices.pivot_table(values='Close', index='Date', columns=['Ticker'])

    for ticker in tbl_stock.columns:
        df_series = df_stocks[df_stocks['Ticker']==ticker]
        df_series.index = pd.to_datetime(df_series['Date'])
        with pd.option_context('mode.chained_assignment', None):
            df_series['Type'] = 'Stock'
        write_piot_series(ticker, df_series)
        all_tickers.append(ticker)

    for ticker in tbl_index.columns:
        df_series = df_indices[df_indices['Ticker']==ticker]
        df_series.index = pd.to_datetime(df_series['Date'])
        with pd.option_context('mode.chained_assignment', None):
            df_series['Type'] = 'StockIndex'
        write_piot_series(ticker, df_series)
        all_tickers.append(ticker)

    # Update ticker index file
    ticker_list = pd.DataFrame(all_tickers, columns=['Ticker'])
    ticker_list.to_csv(os.path.join(PILOT_DATA_HOME,'ticker.list'), header=False)

    return df_stocks, df_indices


# Useful variable

def list_pilot_series():
    PILOT_DATA_HOME = os.path.join(DATA_HOME, 'pilot.data')
    csvlist = os.listdir(PILOT_DATA_HOME)
    symbols = [f.replace('.csv','') for f in csvlist if f.endswith('.csv')]
    return symbols

def load_pilot_series():
    """Returns a dictionary with ticker as key."""
    dataset = {}
    for symbol in list_pilot_series():
        if DEBUG:
            print("Loading: %s" % symbol)
        series = pd.read_csv(os.path.join(PILOT_DATA_HOME, '%s.csv' % symbol))
        series.index = pd.to_datetime(series['Date'])
        series.asfreq('d')
        dataset[symbol] = series
    return dataset
