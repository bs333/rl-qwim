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
    adjCl = pd.DataFrame()
    spy = pd.DataFrame()
    for ticker in tickers:
        tick = yf.Ticker(ticker)
        hist = tick.history(start=pd.to_datetime(weights["Date"].values[0]), end=pd.to_datetime(weights["Date"].values[-1])+ pd.Timedelta(days=1))[["Close"]]
        adjCl[ticker] = hist
    tick = yf.Ticker("SPY")
    hist = tick.history(start=pd.to_datetime(weights["Date"].values[0]), end=pd.to_datetime(weights["Date"].values[-1])+ pd.Timedelta(days=1))[["Close"]]
    spy["SPY"] = hist
    adjCl.reset_index(inplace=True)
    spy.reset_index(inplace=True)
    value = pd.DataFrame(columns = ["Date", "Value"])
    for i, j in enumerate(weights["Date"]):
        value.loc[i] = [j, np.dot(weights.iloc[i][1:], adjCl.iloc[i][1:])]
    value['daily_returns'] = value['Value'].pct_change()
    value['cumulative_returns'] = (1 + value['daily_returns']).cumprod()
    value["SPYpct"] = spy["SPY"].pct_change()
    value["SPY"] = (1 + value["SPYpct"]).cumprod()
    value = value.dropna()
    val_mean_ret = np.mean(value['daily_returns']) *252
    val_vol = np.std(value['daily_returns']) * np.sqrt(252)
    val_sharpe = (val_mean_ret) / val_vol
    print(val_sharpe)
    spy_mean_ret = np.mean(value['SPYpct']) *252
    spy_vol = np.std(value['SPYpct']) * np.sqrt(252)
    spy_sharpe = (spy_mean_ret) / spy_vol
    print(spy_sharpe)
    print(weights)
    print(value)
    ax = plt.gca()
    value.plot(kind = 'line', x = 'Date', y = 'cumulative_returns', color = 'green', ax = ax)
    value.plot(kind = 'line', x = 'Date', y = 'SPY', color = 'blue', ax = ax)
    plt.title("Portfolio Value Over Time")
    plt.show()

if __name__ == '__main__':
    tickers = ['IWD', 'IWF', 'IWO', 'EWJ']
    start_date = "2011-01-01" 
    end_date = "2024-01-01"
    daily_returns = getDailyReturns(tickers, start_date, end_date)
    return_preds, rmses = RFmodel(daily_returns)
    covMats = getCovMats(return_preds)
    dates = return_preds.index[40:].to_frame()
    dates.reset_index(drop=True, inplace=True)
    n = len(tickers)
    r_min = 0.1/252
    optimal_weights = getOpts(covMats, np.array(return_preds[40:]), n, r_min)
    for i in range(len(optimal_weights)):
        optimal_weights[i] = list(optimal_weights[i])
    optimal_weights = pd.DataFrame(optimal_weights, columns = tickers)
    optimal_weights = pd.concat([dates, optimal_weights], axis =1)
    plot_weights(optimal_weights, tickers)
    plot_portfolio_value(optimal_weights, tickers)
    print(rmses)

## Tickers
    # ['IJH', 'IWM', "SPY", "EEM", "EWJ", "TIP", "DBO", "FXE", "CEW", "USCI"]
    # ['IWD', 'IWF', 'IWO', 'EWJ']
    # ['IWF', 'EEM', 'SHYG', 'MTUM', 'IWD', 'EWJ']