import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import os, os.path

ticker_list = "stock_ticker.csv"
stocks = pd.read_csv(ticker_list)
start = dt.datetime(2014,1,1)

if not os.path.exists("historical"):
    os.mkdir('historical')
    print("Data directory created.")



for s in stocks['symbol'].values:
    print(s)
    try:
        data = pdr.get_data_yahoo('%s.BK' % s, start=start)
        outfile = "historical/%s.csv" % s
        data.to_csv(outfile)
    except Exception as e:
        print("Error: %s %s" % (s, str(e)))
