import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
import talib
from talib.abstract import *
from data.transform import FeatureSet

def add_technical_indicator_features(daily):
    # Note the use of Close instead of Adj Close for TI calculation.
    inputs = {
        'open': daily["Open"],
        'high': daily["High"],
        'low': daily["Low"],
        'close': daily["Close"],
        'volume': daily["Volume"]
    }

    n = 5
    daily_sma = SMA(inputs, timeperiod=n)
    daily.loc[:,("SMA")] = daily_sma.tolist()    

    n = 5
    daily_wma = WMA(inputs, timeperiod=n)
    daily.loc[:,("WMA")] = daily_wma.tolist()  

    n = 5
    daily_ema = EMA(inputs, timeperiod=n)
    daily.loc[:,("EMA")] = daily_sma.tolist()

    n = 5
    rsi_5d = RSI(inputs, timeperiod=n)
    n = 10
    rsi_10d = RSI(inputs, timeperiod=n)
    n = 15
    rsi_15d = RSI(inputs, timeperiod=n)
    daily.loc[:,("RSI-5")] = rsi_5d.tolist()
    daily.loc[:,("RSI-10")] = rsi_10d.tolist()
    daily.loc[:,("RSI-15")] = rsi_15d.tolist()

    n = 10
    mom_t10 = MOM(inputs, timeperiod=n)
    daily.loc[:,("MOM-10")] = mom_t10.tolist()


    # STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # See documentation in C code.
    slowk, slowd = STOCH(inputs, prices=['high', 'low', 'close']) # No need for period.!! Check out the C documentation.
    daily.loc[:,("Stoch-K")] = slowk.tolist()
    daily.loc[:,("Stoch-D")] = slowd.tolist()

    return daily


def add_feature_set_1(df):
    # Note the use of Close instead of Adj Close for TI calculation.
    inputs = {
        'open': df["Open"],
        'high': df["High"],
        'low': df["Low"],
        'close': df["Close"],
        'volume': df["Volume"]
    }
    
    # TODO: put in class
    # stock_fs = FeatureSet()
    # stock_fs.compile(df)
    # df = stock_fs.df
    
    # Features compute from ta.
    df_ta = df.copy()
    df_ta = add_all_ta_features(df_ta, open="Open", high="High", low="Low", close="Close", volume="Volume")

    # FF-01 Yesterday's price
    # Add return column: Return from yesterday's closing.
    df.loc[:,("FF-01")]  = df["Close"].shift(1)
    
    # FF-02 Period return (yesterday's)
    # Add return column: Return from yesterday's closing.
    df.loc[:,("FF-02")]  = np.log(df["Close"].shift(1)/df["Close"])
    
    # FF-3	Simple n-day moving average
    df.loc[:,("FF-03")] = SMA(inputs, timeperiod=5).tolist()    
    # FF-4	Weighted 14-day moving average
    df.loc[:,("FF-04")] = WMA(inputs, timeperiod=5).tolist()  
    # FF-5	Exponential Moving Average (EMA)
    df.loc[:,("FF-05")] = EMA(inputs, timeperiod=5).tolist()
    # FF-6	Relative strength index (RSI)
    rsi_15d = RSI(inputs, timeperiod=15)
    df.loc[:,("FF-06")] = rsi_15d.tolist()
    # FF-7	Momentum MOM
    n = 10
    mom_t10 = MOM(inputs, timeperiod=n)
    df.loc[:,("FF-07")] = mom_t10.tolist()
    # FF-8	Stochastic K%
    # FF-9	Stochastic D%
    # STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # See documentation in C code.
    slowk, slowd = STOCH(inputs, prices=['high', 'low', 'close']) # No need for period.!! Check out the C documentation.
    df.loc[:,("FF-08")] = slowk.tolist()
    df.loc[:,("FF-09")] = slowd.tolist()
    
    # FF-10	Signal(n) ~ MACD
    
    # FF-11	Larry William’s R%
    willr_14 = WILLR(inputs)
    df.loc[:,("FF-11")] = willr_14.tolist()
    
    # FF-12	Accumulation/Distribution (A/D) oscillator
    df.loc[:,("FF-12")] = df_ta[['volume_adi']]
    # FF-13	CCI (Commodity channel index)
    # FF-14	Average True Range (ATR)
    # FF-15	Average Directional Movement Index (ADMI)
    # FF-16	Price rate-of-change (ROC)
    # FF-17	CHO (Chaikin Oscillator, which measures the change of the average range of prices in a certain period)
    # FF-18	MFI (money flow index, which evaluates the selling and buying pressure with the help of trading price and volume)
    # FF-19	%B Indicator
    # FF-20	Relative Strength (it is used to compare the stock price with the whole market in a certain period)
    # FF-21	Parabolic SAR = high, low, acceleration=0.02, maximum=0
    # FF-22	True range (TR) = high, low, close
    # FF-23	OBV
    # FF-24	Daily Variaty (Daily Price Variation) จาก ReturnsAndVolatility_SingleStock_ADVANC.ipynb
    # FF-25	5-day price St.Dev (Volatility) จาก Eikon_Volatility_StDev_5d.ipynb use VWAP?
    # FF-26	MACD w/ Buy and Sell signal from crossover event.
    # FF-27	Regression Line Slope 200,400 days
    
    return df
    

def add_feature_set_2(df):
    """See documentation for ta https://technical-analysis-library-in-python.readthedocs.io/en/latest/"""
    # Note the use of Close instead of Adj Close for TI calculation.
    inputs = {
        'open': df["Open"],
        'high': df["High"],
        'low': df["Low"],
        'close': df["Close"],
        'volume': df["Volume"]
    }
    # Features compute from ta.
    df_ta = df.copy()
    df_ta = dropna(df_ta)
    df_ta = add_all_ta_features(df_ta, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return df_ta