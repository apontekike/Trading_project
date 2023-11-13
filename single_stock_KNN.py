import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib.pyplot as plt
import scipy.stats as stats
import math

from Technical_indicators import get_macd,get_moving_average,money_flow_index,get_aroon
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def get_historical(ticker,period,interval):
    stock = yf.Ticker(ticker)
    stock = stock.history(period=period, interval=interval)
    return stock

def KknClass_model_training(stock):

    X = stock.iloc[:,stock.columns != "Signal"]
    y = stock["Signal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_neighbors = list(range(1,20))

    #Convert to dictionary
    hyperparameters = dict(n_neighbors=n_neighbors)
    
    #Create new KNN object
    knn = KNeighborsClassifier()

    #Use GridSearch
    clf = GridSearchCV(knn, hyperparameters, cv=10)
    
    #Fit the model
    best_model = clf.fit(X_train,y_train)
    
    #Print The value of best Hyperparameters
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

    predict_y_test = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predict_y_test)
    classification_report_output = classification_report(y_test, predict_y_test)

    print("Classification Report:\n", classification_report_output)

    return best_model

def backtest_model(model,stock):

    prices = list(stock["Close"])

    signals = model.predict(stock.drop(columns=["Close"]))

    returns = []

    returns.append(0)

    last_price = 0
    position = 0

    for x in range(len(signals)):

        today_p = prices[x]

        if position == 1:
            re = (today_p / last_p)*(returns[x-1]+1)-1
            returns.append(re)
        elif position == -1:
            re = (-((today_p / last_p)-1)+1)*(returns[x-1]+1)-1
            returns.append(re)

        last_p = today_p
        position = signals[x]
 
    return returns[len(returns)-1]
    

def main():

    #GET DATA 
    ###########################################################################################

    tickers = pd.read_csv("Tickers.csv")
    number_of_stocks = len(tickers)-400

    x = 0
    for ticker in tickers["Ticker"].iloc[:number_of_stocks]:
        
        #Get stock info

        stock = get_historical(ticker,"5y","1d")

        if not(stock.empty):

            #Choose feature variables
            stock["MACDI"] = get_macd(stock)
            stock["MA50-MA20"] = get_moving_average(stock,50) - get_moving_average(stock,20)
            stock["MFI"] = money_flow_index(stock)
            up,dn = get_aroon(stock)
            stock["up"] = up
            stock["dn"] = dn
            stock["H-L"] = stock["High"] - stock["Low"]


            #Chose target variable
            stock["Signal"] = np.where(stock['Close'].shift(-1) > stock['Close'], 1, -1)

            stock.drop(columns=["Dividends","Stock Splits","Open","Low"],inplace=True)

            #get rid of the NA
            stock.dropna(inplace=True)

            if x == 0:
                stocks = stock
                x = 1
            else:
                stocks = pd.concat([stocks,stock],copy=True)
                

    stock_train = stocks.iloc[:len(stocks)-50]
    stock_test = stocks.iloc[len(stocks)-50:]

    # BUILD MODEL
    #################################################################

    stock_train.drop(columns=["Close"],inplace=True)
    model = KknClass_model_training(stock_train)

    # BACKTEST
    #########################################################################

    returns = []
    test = 300
    final = pd.DataFrame(data=None,columns=["Ticker","CumRe"])

    for ticker in tickers["Ticker"].iloc[number_of_stocks:]:

        stock = get_historical(ticker,"3mo","1d")

        if stock.empty:
            continue

 
        #Choose feature variables
        stock["MACDI"] = get_macd(stock)
        stock["MA50-MA20"] = get_moving_average(stock,50) - get_moving_average(stock,20)
        stock["MFI"] = money_flow_index(stock)
        up,dn = get_aroon(stock)
        stock["up"] = up
        stock["dn"] = dn
        stock["H-L"] = stock["High"] - stock["Low"]

        # get rid of unneccesary columns
        stock.drop(columns=["Dividends","Stock Splits","Open","Low"],inplace=True)
        stock.dropna(inplace=True)

        if stock.empty:
            continue

        re = backtest_model(model,stock)
        final.loc[len(final)] = [ticker,re]
        returns.append(re)

    # PLOT DISTRIBUTION OF RETURNS
    ###########################################################################################

    returns = np.array(returns)
    returns = np.sort(returns)
    mu = np.mean(returns)
    variance = np.var(returns)
    sigma = math.sqrt(variance)
    plt.plot(returns, stats.norm.pdf(returns, mu, sigma))
    plt.show()

    # SAVE OUTPUT
    ##########################################################################################
    
    final.sort_values(by=['CUmRe'],inplace=True,ascending=True)
    final.to_csv("Stocks_cumRe.csv")


if __name__ == "__main__":
    main()


