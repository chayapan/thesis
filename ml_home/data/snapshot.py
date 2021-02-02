#
# NOTICE: This snapshot is to be replaced with data from database system.
# The pilot data is in data/pilot.py
import os
import pandas as pd

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

set100 = """GFPT
STA
CBG
CPF
ICHI
KTIS
M
MALEE
MINT
SAPPE
TU
TVO
BAY
BBL
KBANK
KKP
KTB
LHFG
SCB
TCAP
TISCO
TMB
AEONTS
ASP
JMT
KTC
MTC
SAWAD
THANI
BLA
THREL
PTL
IVL
PTTGC
EPG
SCC
SCCC
TASCO
TPIPL
VNG
BJCHI
CK
ITD
STEC
STPI
TRC
TTCL
UNIQ
AMATA
ANAN
AP
BLAND
CPN
LH
LPN
MBK
PSH
QH
S
SF
SIRI
SPALI
U
UV
WHA
BANPU
BCP
CKP
DEMCO
EA
EGCO
ESSO
GUNKUL
IRPC
PTG
PTT
PTTEP
RATCH
SGP
SPCG
SUPER
TOP
TTW
BEAUTY
BIG
BJC
CPALL
GLOBAL
HMPRO
KAMART
LOXLEY
MC
MEGA
RS
BCH
BDMS
BH
CHG
VIBHA
BEC
MAJOR
MONO
VGI
WORK
CENTEL
ERW
AAV
AOT
BA
BTS
NOK
PSL
THAI
TTA
DELTA
HANA
KCE
SVI
ADVANC
DTAC
INTUCH
JAS
SAMART
SAMTEL
THCOM
""" # From FinalList in Data.xlsx


set100 = set100.split()

def stockdb_viewbydate(date1, date2):
    """Date or two date."""
    os.chdir("/home/jovyan/dataset/Datastream")
    attr = [f for f in os.listdir() if f.endswith('.csv')]
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
    os.chdir("/home/jovyan/dataset/Datastream")
    attr = [f for f in os.listdir() if f.endswith('.csv')]
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