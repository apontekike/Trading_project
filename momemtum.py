import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime,date,timedelta
import scipy.optimize as opt

def cal_momentum_stocks(fname):

    tickers = pd.read_csv(fname)
    tickers = tickers["Ticker"]
    tickers = list(tickers.T.values)

    momemtum = pd.DataFrame([],columns=["ticker","Momemtum","ID"])
    today = date.today().strftime("%Y-%m-%d")
    year_from_today = (date.today() - timedelta(days=366)).strftime("%Y-%m-%d")

    for ticker in tickers:
        try: 
            stock = yf.Ticker(ticker)
            stock_data = stock.history(start=year_from_today, end=today, interval="1mo")
        except AttributeError:
            continue
        
        stock_data = stock_data["Close"]

        if len(stock_data) < 11:
            continue

        mom = (stock_data.iloc[-1]/stock_data.iloc[0]) - 1

        stock = yf.Ticker(ticker)
        ID_data = stock.history(period="1y", interval="1d")
        ID_data = ID_data["Close"]

        ID = ID_data.pct_change()
        perc = list(ID < 0)
        ID = np.sign(mom)*((perc.count(True)/len(perc))-(perc.count(False))/len(perc))

        momemtum.loc[-1] = [ticker,mom,ID]  
        momemtum.index = momemtum.index + 1  
        momemtum = momemtum.sort_index() 

    momemtum.to_csv("Momemtum.csv") 
    
    return momemtum

def sort_momemtum_stocks(stocks):

    stocks.sort_values(by=['Momemtum'],ascending=False,inplace=True)

    long_portfolio = stocks.iloc[:50]
    short_portfolio = stocks.iloc[-50:]

    long_portfolio.sort_values(by=['ID'],ascending=False,inplace=True)
    short_portfolio.sort_values(by=['ID'],ascending=False,inplace=True)

    long_portfolio = long_portfolio.iloc[-7:].reset_index(drop=True)
    short_portfolio = short_portfolio.iloc[-7:].reset_index(drop=True)

    return list(long_portfolio["ticker"].values), list(short_portfolio["ticker"].values)


def historic_returns(data):

    stock = yf.Ticker(data[0])
    stocks_data = stock.history(period="max", interval="1mo")
    stocks_data = stocks_data["Close"]
    stocks_data = pd.DataFrame(stocks_data.pct_change().values,columns=[data[0]],index = stocks_data.index)
    stocks_data.dropna(inplace=True)
    stocks_data.columns = [data[0]]

    for ticker in data[1:]:
        stock = yf.Ticker(ticker)
        stock = stock.history(period="max", interval="1mo")
        stock = stock["Close"]
        stock = stock.pct_change().to_frame()
        stock.columns = [ticker]
        stock.dropna(inplace=True)
        stocks_data = pd.concat([stocks_data, stock], axis=1, join="inner")
    
    return stocks_data

def variance_of_portfolio(weights,cov_matrix):

    cov_sum = []
    for x in range(len(weights)):
        for y in range(x,len(weights)):
            if x == y:
                cov_sum.append(weights[x]*weights[y]*cov_matrix.iloc[y,x])
            else:
                cov_sum.append(2*weights[x]*weights[y]*cov_matrix.iloc[y,x])
    
    return np.sqrt(sum(cov_sum))

def optimize_portfolio(long,short):

    long_pw = np.array([1/(len(long)) for _ in range((len(long)))])
    short_pw = np.array([1/(len(short))for _ in range((len(short)))])

    long_data = historic_returns(long)
    long_cov = long_data.cov()

    short_data = historic_returns(short)
    short_cov = short_data.cov()

    #Says one minus the sum of all variables must be zero
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})

    #Required to have non negative values
    bnds = tuple((0.05,0.2) for x in range(len(long_pw)))

    long_op_weigths = opt.minimize(variance_of_portfolio,x0=long_pw,args=(long_cov),
                                    method='SLSQP', bounds=bnds ,constraints=cons)

    short_op_weigths = opt.minimize(variance_of_portfolio,x0=short_pw,args=(short_cov),
                                    method='SLSQP', bounds=bnds ,constraints=cons)

    long_portfolio = long_op_weigths.x
    short_portfolio = short_op_weigths.x

    positions = {"long" : {}, "short" : {}}

    for x in range(len(long)+len(short)):
        if x < len(long):
            positions["long"][long[x]] = long_portfolio[x]
        else:
            positions["short"][short[x-len(long)]] = short_portfolio[x-len(long)]
        
    return positions

def calculate_amount(dollars,fname,long_ratio):

    positions = json.load(open("positions.json","r"))

    long_amount = dollars*long_ratio
    short_amount = dollars - long_amount

    Trades =  {"long":{},"short":{}}
    for stock in positions["long"]:
        Trades["long"][stock] = round(positions["long"][stock]*long_amount,2)

    for stock in positions["short"]:
        Trades["short"][stock] = round(positions["short"][stock]*short_amount,2)
    
    json_object = json.dumps(Trades, indent=4)

    with open("Trades.json", "w") as outfile:
        outfile.write(json_object)

    return Trades

def momemntum(fname):

    cal_momemtum = cal_momentum_stocks(fname)

    #cal_momemtum.drop(stocks.columns[0], axis=1,inplace=True)
    #cal_momemtum = pd.read_csv("Momemtum.csv")
    #cal_momemtum.drop(cal_momemtum.columns[0], axis=1,inplace=True)

    long, short = sort_momemtum_stocks(cal_momemtum)

    positions = optimize_portfolio(long,short)

    json_object = json.dumps(positions, indent=4)
 
    with open("positions.json", "w") as outfile:
        outfile.write(json_object)
    
    trades = calculate_amount(500000,"positions.json",0.60)

    return trades

if __name__ == "__main__":
    momemntum("Ticker.csv")
