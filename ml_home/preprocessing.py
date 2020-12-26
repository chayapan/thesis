
import talib
from talib.abstract import *

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