import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf
import gurobipy as gp

print(cp.installed_solvers())

with gp.Env(empty=True) as env:
    env.setParam("OutputFlag", 0)
    env.start()

def optimize_portfolio_with_risk(stocks, returns, cov_matrix, min_return, max_stocks):
    """
    Select a certain amount of stocks to minimize risk subject to a minimum expected return.

    Parameters:
    - stocks: List of stock symbols.
    - returns: Array of expected returns for each stock.
    - cov_matrix: Covariance matrix of stock returns.
    - min_return: Minimum expected return for the portfolio.
    - max_stocks: Maximum number of stocks to select.

    Returns:
    - A tuple of the optimized portfolio, its expected return, and its risk (standard deviation).
    """

    n = len(stocks) # Number of stocks
    # Decision variables
    x = cp.Variable(n, boolean=True) # Binary variables for stock selection
    w = cp.Variable(n) # Continuous variables for weights

    # Objective: Minimize portfolio risk (standard deviation)
    risk = cp.quad_form(w, cov_matrix)
    objective = cp.Minimize(risk)

    # Constraints
    constraints = [
        cp.sum(w) == 1, # Sum of weights is 1
        w >= 0, # No short selling
        cp.sum(x) == max_stocks,
        w <= x, # Select exactly max_stocks stocks
        cp.sum(cp.multiply(w, returns)) >= min_return # Minimum expected return
    ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)

    # Check if a valid solution exists
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        selected_stocks = [stocks[i] for i in range(n) if round(x.value[i]) == 1]
        portfolio_return = sum(w.value[i] * returns[i] for i in range(n))
        portfolio_risk = np.sqrt(risk.value)
        return selected_stocks, portfolio_return, portfolio_risk, w.value
    else:
        raise Exception("No optimal solution found.")

def getExpRetCovMatr(tickers):
    #Import ETFs
    data = yf.download(tickers, start="2014-01-01", end="2024-01-01")['Adj Close']

    # Calculate daily returns
    daily_returns = data.pct_change() * 100

    # Calculate expected returns (mean of daily returns)
    expected_returns = daily_returns.mean() 

    # Calculate covariance matrix of returns
    covariance_matrix = daily_returns.cov() 

    return expected_returns, covariance_matrix


