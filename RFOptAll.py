"""
Created on Sun Mar 3 2024

@author: kevinlochbihler
RFOpt.py

Portfolio optimization problem for Undergrad Summer Research
"""

import pandas as pd
import numpy as np
import yfinance as yf
import cvxopt
from cvxopt import matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def getDailyReturns(tickers, start_date, end_date):
    adjCl = pd.DataFrame()
    for ticker in tickers:
        tick = yf.Ticker(ticker)
        hist = tick.history(start=start_date, end=end_date)[["Close"]]
        adjCl[ticker] = hist
    returns = adjCl.pct_change()
    returns = returns[1:]
    returns["Date"] = returns.index
    firstCol = returns.pop("Date")
    returns.insert(0, "Date", firstCol)
    returns = returns.set_index("Date")
    return returns

def RFmodel(dailyReturns):
    return_preds = pd.DataFrame()
    rmses = {}
    for tick in dailyReturns.columns:
        X = dailyReturns[:-1]
        y = dailyReturns[tick][1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, shuffle=False)
        rf = RandomForestRegressor(n_estimators=100, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        pred_returns = pd.DataFrame(y_pred, columns=[tick], index=y_test.index)
        rmse = str(round(np.sqrt(mean_squared_error(y_test, y_pred)), 5)*100) + "%"
        return_preds = pd.concat([return_preds, pred_returns], axis=1)
        rmses[tick] = rmse
    return return_preds, rmses

def getCovMats(return_preds):
    covMats = []
    for i in range(len(return_preds)-40):
        fortyDaysReturns = return_preds[i:i+40]
        covMats += [np.array(fortyDaysReturns.cov())]
    return covMats

def getOpts(covMats, expReturns, n, rMin):
    sols = []
    for i in range(len(covMats)):
        while True:
            try:
                sigma = matrix(covMats[i])
                P = sigma
                q = matrix(np.zeros((n,1)))
                G = matrix(np.concatenate((
                    -np.transpose(np.atleast_2d(np.array(expReturns[i])).T),
                    np.identity(n),
                    -np.identity(n))))
                h = matrix(np.concatenate((
                    -np.ones((1,1))*rMin,
                    np.full((n,1), .4), 
                    np.full((n,1), 0))))
                print(h)
                A = matrix(1.0, (1,n))
                b = matrix(1.0)
                sol = cvxopt.solvers.qp(P, q, G, h, A, b)
                sols += [sol['x']]
                break
            except ValueError:
                rMin -= 0.01
    return sols

def plot_weights(sols, tickers):
    ax = plt.gca()
    colors = ['green', 'blue', 'orange', 'red', 'purple', 'yellow', 'pink', 'gray', 'olive', 'cyan']
    for i in  tickers:
        sols.plot(kind = 'line', x = 'Date', y = i, color = colors[tickers.index(i)], ax = ax)
    plt.title("Daily Asset Allocation Optimization")
    plt.show()

def plot_portfolio_value(weights, tickers):
    weightsLG = weights[0]
    weightsMD = weights[1]
    weightsSM = weights[2]
    #LG Portfolio
    value1 = pd.DataFrame(columns = ["Date", "Value"])
    adjCl = pd.DataFrame()
    for ticker in tickers[0]:
        tick = yf.Ticker(ticker)
        hist = tick.history(start=pd.to_datetime(weightsLG["Date"].values[0]), end=pd.to_datetime(weightsLG["Date"].values[-1])+ pd.Timedelta(days=1))[["Close"]]
        adjCl[ticker] = hist
    adjCl.reset_index(inplace=True)
    for i, j in enumerate(weightsLG["Date"]):
        value1.loc[i] = [j, np.dot(weightsLG.iloc[i][1:], adjCl.iloc[i][1:])]
    value1['daily_returns'] = value1['Value'].pct_change()
    value1['LG Portfolio'] = (1 + value1['daily_returns']).cumprod()

    #MD Portfolio
    value2 = pd.DataFrame(columns = ["Date", "Value"])
    adjCl = pd.DataFrame()
    for ticker in tickers[1]:
        tick = yf.Ticker(ticker)
        hist = tick.history(start=pd.to_datetime(weightsMD["Date"].values[0]), end=pd.to_datetime(weightsMD["Date"].values[-1])+ pd.Timedelta(days=1))[["Close"]]
        adjCl[ticker] = hist
    adjCl.reset_index(inplace=True)
    for i, j in enumerate(weightsMD["Date"]):
        value2.loc[i] = [j, np.dot(weightsMD.iloc[i][1:], adjCl.iloc[i][1:])]
    value2['daily_returns'] = value2['Value'].pct_change()
    value2['MD Portfolio'] = (1 + value2['daily_returns']).cumprod()

    #SM Portfolio
    value3 = pd.DataFrame(columns = ["Date", "Value"])
    adjCl = pd.DataFrame()
    for ticker in tickers[2]:
        tick = yf.Ticker(ticker)
        hist = tick.history(start=pd.to_datetime(weightsSM["Date"].values[0]), end=pd.to_datetime(weightsSM["Date"].values[-1])+ pd.Timedelta(days=1))[["Close"]]
        adjCl[ticker] = hist
    adjCl.reset_index(inplace=True)
    for i, j in enumerate(weightsSM["Date"]):
        value3.loc[i] = [j, np.dot(weightsSM.iloc[i][1:], adjCl.iloc[i][1:])]
    value3['daily_returns'] = value3['Value'].pct_change()
    value3['SM Portfolio'] = (1 + value3['daily_returns']).cumprod()

    spy = pd.DataFrame()
    tickspy = yf.Ticker("SPY")
    hist = tickspy.history(start=pd.to_datetime(weightsLG["Date"].values[0]), end=pd.to_datetime(weightsLG["Date"].values[-1])+ pd.Timedelta(days=1))[["Close"]]
    spy["SPY"] = hist
    spy.reset_index(inplace=True)
    value1["SPYpct"] = spy["SPY"].pct_change()
    value1["SPY"] = (1 + value1["SPYpct"]).cumprod()
    value1 = value1.dropna()
    value2 = value2.dropna()
    value3 = value3.dropna()
    val1_mean_ret = np.mean(value1['daily_returns']) *252
    val1_vol = np.std(value1['daily_returns']) * np.sqrt(252)
    val1_sharpe = (val1_mean_ret) / val1_vol
    val2_mean_ret = np.mean(value2['daily_returns']) *252
    val2_vol = np.std(value2['daily_returns']) * np.sqrt(252)
    val2_sharpe = (val2_mean_ret) / val2_vol
    val3_mean_ret = np.mean(value3['daily_returns']) *252
    val3_vol = np.std(value3['daily_returns']) * np.sqrt(252)
    val3_sharpe = (val3_mean_ret) / val3_vol
    spy_mean_ret = np.mean(value1['SPYpct']) *252
    spy_vol = np.std(value1['SPYpct']) * np.sqrt(252)
    spy_sharpe = (spy_mean_ret) / spy_vol
    print(weightsLG)
    print(value1)
    print(weightsMD)
    print(value2)
    print(weightsSM)
    print(value3)
    print("SPY Sharpe: " + str(spy_sharpe))
    print("LG Sharpe: " + str(val1_sharpe))
    print("MD Sharpe: " + str(val2_sharpe))
    print("SM Sharpe: " + str(val3_sharpe))

    ax = plt.gca()
    value1.plot(kind = 'line', x = 'Date', y = 'SPY', color = 'blue', ax = ax)
    value1.plot(kind = 'line', x = 'Date', y = 'LG Portfolio', color = 'green', ax = ax)
    value2.plot(kind = 'line', x = 'Date', y = 'MD Portfolio', color = 'red', ax = ax)
    value3.plot(kind = 'line', x = 'Date', y = 'SM Portfolio', color = 'orange', ax = ax)
    plt.title("Cumulative Returns of Optimized Portfolios vs. SPY")
    plt.show()

if __name__ == '__main__':
    tickers = [['IJH', 'IWM', "SPY", "EEM", "EWJ", "TIP", "DBO", "FXE", "CEW", "USCI"], 
               ['IJH', 'EWJ', 'FXE', 'CEW', 'USCI'],
               ['IJH', 'EWJ', 'CEW']]
    start_date = "2011-01-01" 
    end_date = "2024-01-01"
    optimal_weights_all = []
    for ticker in tickers:
        daily_returns = getDailyReturns(ticker, start_date, end_date)
        return_preds, rmses = RFmodel(daily_returns)
        covMats = getCovMats(return_preds)
        dates = return_preds.index[40:].to_frame()
        dates.reset_index(drop=True, inplace=True)
        n = len(ticker)
        r_min = 0.1/252
        optimal_weights = getOpts(covMats, np.array(return_preds[40:]), n, r_min)
        for i in range(len(optimal_weights)):
            optimal_weights[i] = list(optimal_weights[i])
        optimal_weights = pd.DataFrame(optimal_weights, columns = ticker)
        optimal_weights = pd.concat([dates, optimal_weights], axis =1)
        plot_weights(optimal_weights, ticker)
        print(rmses)
        optimal_weights_all += [optimal_weights]
    plot_portfolio_value(optimal_weights_all, tickers)

## Tickers
    # ['IJH', 'IWM', "SPY", "EEM", "EWJ", "TIP", "DBO", "FXE", "CEW", "USCI"]
    # ['IJH', 'EWJ', 'FXE', 'CEW', 'USCI']
    # ['IJH', 'EWJ', 'CEW']
