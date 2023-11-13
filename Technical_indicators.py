import pandas as pd
import numpy as np 

def money_flow_index(stock,n=14):
    typical_price = (stock["High"] + stock["Low"]  + stock["Close"] ) / 3
    money_flow = typical_price * stock["Volume"]
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign

    # Calculate gain and loss using vectorized operations
    positive_mf = np.where(signed_mf > 0, signed_mf, 0)
    negative_mf = np.where(signed_mf < 0, -signed_mf, 0)

    mf_avg_gain = pd.Series(positive_mf).rolling(n, min_periods=1).sum()
    mf_avg_loss = pd.Series(negative_mf).rolling(n, min_periods=1).sum()

    return (100 - 100 / (1 + mf_avg_gain / mf_avg_loss)).to_numpy()

def get_moving_average(stock,avg):
    return stock["Close"].rolling(window=avg).mean()

def get_macd(ticker):
    k = ticker["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    # Get the 12-day EMA of the closing price
    d = ticker['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
    macd = k - d
    # Add all of our new values for the MACD to the dataframe
    return macd

def get_aroon(stock, lb=25):

    up = 100 * stock.High.rolling(lb + 1).apply(lambda x: x.argmax()) / lb
    dn = 100 * stock.Low.rolling(lb + 1).apply(lambda x: x.argmin()) / lb
    return up,dn

def price_and_volume(stock):
    chg = stock["Close"].pct_change
    print(stock.iloc[1:]["Volume"])
    x = np.ndarray(chg) * stock.iloc[1:]["Volume"].values
    return x.cumsum()
