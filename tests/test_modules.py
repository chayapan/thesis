import pytest

def test_load_ml_home_modules():
    """Load functions and methods from some of the modules."""
    # from algorithm import *
    from algorithm.clustering import dendrogram_with_extra_info
    assert callable(dendrogram_with_extra_info)

    from data.src import to_float, read_csv_from_historical_folder

    # Test reading CSV price
    from data.src import read_csv_from_historical_folder
    import numpy as np
    import pandas as pd
    df = read_csv_from_historical_folder("SCC")
    type(df) == pd.DataFrame
    type(df['High'].values) == np.ndarray
    assert len(df) > 0 # pd.DataFrame
    assert len(df['High'].values) > 0

    # Functions for reading data sources
    from data.src import stockdata_fetch_local, indexdata_fetch_local, stockdata_fetch_historical

    # Functions for generating data for simulation study
    from data.generator import make_gbm_series, dgf10, dgf11
    from data.simulation import TimeSeries
    from data.simulation import make_dataset_linear, make_dataset_exponential

    # Functions for reading historical data from main collection (files and database)
    from data.snapshot import set100, SET100, SET100_db_engine
    from data.snapshot import stockdb_viewbystock, stockdb_viewbydate

    from experiment.design import Experiment, TimeSeriesClustering
    from learn.hypothesis import H, Hypothesis
    from train.performance import Evaluation

def test_local_stockdata():
    """Read several stocks into pivot table. Also get price standard deviation and daily return."""
    import numpy as np
    import pandas as pd
    from dataset import Yr2014
    from data.src import stockdata_fetch_local
    df = stockdata_fetch_local(Yr2014.dt_start, Yr2014.dt_end, ["SCB","BBL","KBANK"])
    tbl = df.pivot_table(values='Close', index='Date', columns=['Industry', 'Sector', 'Ticker'])
    df[df['Ticker']=='SCB']
    df[df['Ticker']=='KBANK']

    type(df) == pd.DataFrame
    type(df['High'].values) == np.ndarray

    assert len(df) > 0 # pd.DataFrame
    assert len(df['High'].values) > 0

    assert list(df.columns) == ["Adj Close", "Close", "DailyPriceVariation",
                                "DailyReturn", "Date", "High", "Industry", "Low", "Open",
                                "PriceStDev-5d", "Sector", "Ticker", "Volume"]

def test_experiment_design():
    """Initialize experiment instance from base class."""
    from experiment.design import TimeSeriesClustering, Experiment
    TimeSeriesClustering()

def func(x):
    from data.snapshot import SET100, SET100_db_engine, make_index
    engine = SET100_db_engine()
    print(engine)
    return x + 1

def test_load_data_snapshot():
    assert func(4) == 5

def func_data_generators():
    from data.generator import plot_line, gd2df, add_noise, dgf10, dgf11, dgf1, dgf2, dgf3, dgf4, dgf5, dgf6, dgf7, dgf8, dgf9
    for g in [dgf10, dgf11, dgf1, dgf2, dgf3, dgf4, dgf5, dgf6, dgf7, dgf8, dgf9]:
        X,y = g()
    return 1

def test_data_generators():
    assert func_data_generators() == 1
