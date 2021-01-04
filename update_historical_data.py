import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import sys, os, os.path
ML_HOME = os.path.abspath(os.path.join(".", "ml_home")) # ML workspace
EXPERIMENT_HOME = os.path.abspath(os.path.join(ML_HOME, "..")) # Experiment workspace
DATA_HOME = os.path.abspath(os.path.join(EXPERIMENT_HOME,"dataset")) # Dataset location
sys.path.insert(0, EXPERIMENT_HOME)
sys.path.insert(0, ML_HOME) # Add to path so can load our library
os.chdir(DATA_HOME) # Change working directory to experiment workspace
print("Current Path: ", os.path.abspath(os.curdir), "; Data Home:", DATA_HOME)


def update_SET100():
    ticker_list = "set100.data/_directory.csv"
    stocks = pd.read_csv(ticker_list)
    start = dt.datetime(2014,1,1)
    
    if not os.path.exists("historical"):
        os.mkdir('historical')
        print("Data directory created.")
    
    for s in stocks['ticker'].values:
        print(s)
        try:
            data = pdr.get_data_yahoo('%s.BK' % s, start=start)
            outfile = "historical/%s.csv" % s
            data.to_csv(outfile)
        except Exception as e:
            print("Error: %s %s" % (s, str(e)))

