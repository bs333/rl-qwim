import yfinance as yf
import pulp
import numpy as np


tickers = ["IWD", "IWF", "IWO", "EWJ","EEM"] # list of ETFs

ETFdata = yf.download(tickers, start="2004-01-01", end="2022-12-31")['Adj Close']

dailyReturns = stock_data.pct_change().dropna() # daily returns

expectedReturns = returns.mean().values # expected returns
covarianceMatrix = returns.cov().values #covariance matrix

binIntProb = pulp.LpProblem("ETF_Selection", pulp.LpMaximize) # binary programming problem

selected = {}
for i in range(len(tickers)):
    selected = pulp.LpVariable(f"Selected_{i}", cat='Binary')


for p in range(len(tickers)):
    binIntProb += pulp.lpSum(selected[i] * expected_returns[i]) # Our objective is to maximize expected return


# define and add constraints
maxSelectedETFs = len(tickers)  # maximum number of ETFs to select

for j in range(len(tickers)):
    binIntProb += pulp.lpSum(selected[i] <= maxSelectedETFs

binIntProb.solve() # solve the problem

print("Selected ETFs")
for i in range(len(tickers)):
    if selected[i].value() == 1:
        print(f"ETF {tickers[i+1]}: Return = {expectedReturns[i]}")