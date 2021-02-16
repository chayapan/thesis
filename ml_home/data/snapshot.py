#
# NOTICE: This snapshot is to be replaced with data from database system.
# The pilot data is in data/pilot.py
import os
import pandas as pd
from sqlalchemy import create_engine

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

pilot10stock = ["SCB","KBANK","AOT","BTS","AP","LH","CPF","KSL","PTT","RATCH"]

######### Real Data ###########
# https://studentmahidolac-my.sharepoint.com/:x:/g/personal/chayapan_kha_student_mahidol_ac_th/EarTa03YiaJGhot_-I-a2rcBHEYwgJK-2troRO5FlT3sMw?e=RH7B0C

class SET100:
    """SET100_Data.xlsm"""
    dataFile = os.path.join(os.environ['DATA_HOME'], 'Datastream', 'SET100_Data.xlsm')
    def __init__(self):
        self.sheets = self.get_sheets()
        ##### VO #####
        VO = self.sheets['VO']
        localCode = VO[2:3]  # Stock symbol
        companyName = VO[4:5]  # Name
        bDate = VO[5:6] # dataAvailableFrom
        dbEntityCode = VO[6:7] # internal database code 
        df_VO = VO[7:]   # Data
        df_VO.columns = companyName.values[0] # Set local code as column header
        df_VO = df_VO.set_index(df_VO.columns[0]) # Make index on date column
        self.sheets['VO'] = df_VO
        ##### MV #####
        MV = self.sheets['MV']
        localCode = MV[2:3]  # Stock symbol
        companyName = MV[4:5]  # Name
        bDate = MV[5:6] # dataAvailableFrom
        df_MV = MV[7:]   # Data
        df_MV.columns = companyName.values[0] # Set local code as column header
        df_MV = df_MV.set_index(df_MV.columns[0]) # Make index on date column
        self.sheets['MV'] = df_MV
        ##### P #####
        P = self.sheets['P']
        localCode = P[2:3]  # Stock symbol
        companyName = P[4:5]  # Name
        bDate = P[5:6] # dataAvailableFrom 
        df_P = P[7:]   # Data
        df_P.columns = companyName.values[0] # Set local code as column header
        df_P = df_P.set_index(df_P.columns[0]) # Make index on date column
        self.sheets['P'] = df_P
        ##### MACD #####
        MACD = self.sheets['MACD']
        localCode = MACD[2:3]  # Stock symbol
        companyName = MACD[4:5]  # Name
        bDate = MACD[5:6] # dataAvailableFrom 
        df_MACD = MACD[7:]   # Data
        df_MACD.columns = companyName.values[0] # Set local code as column header
        df_MACD = df_MACD.set_index(df_MACD.columns[0]) # Make index on date column
        self.sheets['MACD'] = df_MACD
    @property
    def symbols(self):
        return self.sheets[0]['Symbol In SET100 Constituent'].values
    @property
    def VO(self):
        return self.sheets['VO']
    @property
    def MV(self):
        return self.sheets['MV']
    @property
    def P(self):
        return self.sheets['P']
    @property
    def MACD(self):
        return self.sheets['MACD']    
    @classmethod
    def get_sheets(cls):
            _sheets = pd.read_excel(cls.dataFile, sheet_name=[0,'VO','MV','P','MACD'])
            return _sheets

## TODO: Replace with the list from SET100_Data.xlsm
# .replace('\n', '')
set100 = ['AAV', 'ADVANC', 'AEONTS', 'AMATA', 'ANAN', 'AOT', 'AP', 'ASP', 'AWC', 'BA', 'BANPU', 'BAY', 'BBL', 'BCH', 'BCP', 'BCPG', 'BDMS', 'BEAUTY', 'BEC', 'BECL', 'BEM', 'BGH', 'BGRIM', 'BH', 'BIG', 'BIGC', 'BJC', 'BJCHI', 'BLA', 'BLAND', 'BMCL', 'BPP', 'BTS', 'CBG', 'CENTEL', 'CHG', 'CK', 'CKP', 'COM7', 'CPALL', 'CPF', 'CPN', 'DELTA', 'DEMCO', 'DTAC', 'EA', 'EARTH', 'EGCO', 'EPG', 'ERW', 'ESSO', 'GFPT', 'GGC', 'GL', 'GLOBAL', 'GLOW', 'GOLD', 'GPSC', 'GULF', 'GUNKUL', 'HANA', 'HEMRAJ', 'HMPRO', 'ICHI', 'IFEC', 'INTUCH', 'IRPC', 'ITD', 'IVL', 'JAS', 'JMART', 'JMT', 'JWD', 'KAMART', 'KBANK', 'KCE', 'KKP', 'KTB', 'KTC', 'KTIS', 'LH', 'LHBANK', 'LOXLEY', 'LPN', 'M', 'MAJOR', 'MALEE', 'MBK', 'MC', 'MEGA', 'MINT', 'MONO', 'MTLS', 'NOK', 'ORI', 'OSP', 'PLANB', 'PLAT', 'PRM', 'PS', 'PSH', 'PSL', 'PTG', 'PTL', 'PTT', 'PTTEP', 'PTTGC', 'QH', 'RATCH', 'ROBINS', 'RS', 'S', 'SAMART', 'SAMTEL', 'SAPPE', 'SAWAD', 'SCB', 'SCC', 'SCCC', 'SCN', 'SF', 'SGP', 'SIM', 'SIRI', 'SPALI', 'SPCG', 'SPRC', 'STA', 'STEC', 'STPI', 'SUPER', 'SVI', 'TASCO', 'TCAP', 'THAI', 'THANI', 'THCOM', 'THREL', 'TICON', 'TISCO', 'TKN', 'TMB', 'TOA', 'TOP', 'TPIPL', 'TPIPP', 'TRC', 'TRUE', 'TTA', 'TTCL', 'TTW', 'TU', 'TUF', 'TVO', 'U', 'UNIQ', 'UV', 'VGI', 'VIBHA', 'VNG', 'WHA', 'WHAUP', 'WORK']

def stockdb_viewbydate(date1, date2):
    """Date or two date."""
    # os.chdir("/home/jovyan/dataset/Datastream/LBNGKSET")
    dataDir = os.path.join(os.environ['DATA_HOME'], 'Datastream', 'LBNGKSET')
    attr = [os.path.join(dataDir,f) for f in os.listdir(dataDir) if f.endswith('.csv')]
    files = attr
    data = {}
    # Load every thing!!!
    for f in attr:
        df_attr = pd.read_csv(f, index_col='Date')
        # print("opening %s " % f)
        df_attr.index = pd.to_datetime(df_attr.index)
        df_attr.sort_index(inplace=True)
        data[f] = df_attr
    
    df = pd.DataFrame()
    # Pick date
    # date_range = '2015-03-01':'2015-03-02'
    for attr in data.keys():
        try:
            col = attr.replace('.csv','')
            col = col.split('_')[1]
            df[col] = data[attr][date1:date2][set100].T[date1]
        except Exception as e:
            print("Error %s %s" % (attr, str(e)))  # Error 4_P.csv 
    return df

def stockdb_viewbystock(stock):
    # os.chdir("/home/jovyan/dataset/Datastream/LBNGKSET")
    # attr = [f for f in os.listdir() if f.endswith('.csv')]
    dataDir = os.path.join(os.environ['DATA_HOME'], 'Datastream', 'LBNGKSET')
    attr = [os.path.join(dataDir,f) for f in os.listdir(dataDir) if f.endswith('.csv')]
    files = attr
    data = {}
    # Load every thing!!!
    for f in attr:
        df_attr = pd.read_csv(f, index_col='Date')
        # print("opening %s " % f)
        df_attr.index = pd.to_datetime(df_attr.index)
        df_attr.sort_index(inplace=True)
        data[f] = df_attr
    df = pd.DataFrame()
    ds = {}
    # Pick stock
    # date_range = '2015-01-01':'2019-12-31'
    for attr in data.keys():
        try:
            ds[attr] =  data[attr]['2015-01-01':'2019-12-31'][stock]
            col = attr.replace('.csv','')
            col = col.split('_')[1]
            df[col] = ds[attr]
        except Exception as e:
            print("Error %s %s" % (attr, str(e)))  # Error 4_P.csv 
    return df

### From BuildDataSetV4
# Prepare Dimensional Model (Data Cube)
# Get measure for each stock on each date. Null value will be droped here.
def get_measure(df, col_name):
    for c in df.columns:
        rows = df[c] # c = company name
        keyx = rows.index.values # time index
        vals = rows.values # value is the measures
        measure = pd.DataFrame(data={'stock':c, 'date':keyx, col_name: vals})
        yield measure

def reduce_fetch_frame(df, col):
    g = get_measure(df, col) # yield chunk: one stock per chunk
    s_df = next(g)  #  start iterator
    for s in g:  # run remaining
        s_df = s_df.append(s)
    return s_df # finish with the data frame that appended all chunks
        
def make_index(df):
    arrays = [df['stock'].values, df['date'].values]
    tuples = list(zip(*arrays))
    # tuples
    index = pd.MultiIndex.from_tuples(tuples, names=['stock','date']) # create index. See pandas doc.
    df.index = index
    return df

def SET100_db_engine(networked=True):
    # Or stock.db sqlite3
    engine = create_engine('postgresql://datauser:1234@172.18.0.1:5432/stockdb', echo=False)
    return engine