"""Test dataset for experiment: data schema and data warehouse functionality."""

import pytest


def test_load_universe_of_companies():
    """Test reading data from Excel file that contain company directory and quantitave variables in different sheets.
    
    Universe of Companies
The information about companies are in the first sheet. This is loaded into data frame and is inserted into table.

The file SET100_Data.xlsm is the master list of all companies in the universe.
    """
    from experiment import os, np, pd, pdr, plt, datetime
    import datetime as dt
    import xlrd
    # First sheet list all companies
    # VO sheet contains VO data

    # Row 3  LOC;  Row 4  Datatype  Row 5 Name
    os.chdir(os.environ['DATA_HOME'] + '/Datastream')

    sheets = pd.read_excel('SET100_Data.xlsm', sheet_name=[0,'VO','MV','P','MACD']) 
    sheets.keys()

    
def test_set100_company_dim():
    """The company dimension table should contain 160 companies. There are 163 symbols and 3 were duplicates."""
    from experiment import os, np, pd
    os.chdir(os.environ['DATA_HOME'] + '/Datastream')
    
    sheets = pd.read_excel('SET100_Data.xlsm', sheet_name=[0])
    
    # List of ticker symbols
    symbols = list(sheets[0]['Symbol In SET100 Constituent'].values)
    
    # Data frame
    df_tickers = sheets[0][['Symbol In SET100 Constituent', 'Company Name', 'Datastream Mnemonic', 'Remark']]
    df_tickers # All 163 stock symbols
    
    # The 160 companies
    df_companies = df_tickers[df_tickers['Datastream Mnemonic'].notnull()]
    df_companies
    
    # The SET100 companies with the sector
    sector_lookup = os.path.join(os.environ['EXPERIMENT_HOME'],"""1.0 Data Acquisition/stock_ticker.csv""")
    sector_lookup = pd.read_csv(sector_lookup)

    set100_companies = df_companies[['Symbol In SET100 Constituent', 'Company Name', 'Datastream Mnemonic']]
    set100_companies

    # Join
    df = set100_companies.merge(sector_lookup, left_on='Symbol In SET100 Constituent', right_on='symbol')
    df['localCode'] = 'TH:'+df['symbol']
    df
    
    # This last dataframe is the 'set100_company_dim' table.
    ### END TEST ###
    
    
def test_load_dataset_for_experiment():
    """Check datababse connection. Load stock symbols. Get sample data frame for experiment."""
    
    from dataset import get_dataset_db
    
    from dataset import shapshot
    the_conn = shapshot.SET100_db_engine()
    dataset_db = get_dataset_db()
    
    
    from sqlalchemy import create_engine
    engine = create_engine('postgresql://datauser:1234@172.18.0.1:5432/stockdb', echo=False)
    assert the_conn == engine
    dataset_db = engine
    
    