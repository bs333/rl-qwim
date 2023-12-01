import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# Function to download data
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data.pct_change().dropna()

# Sharpe Ratio
def sharpe_ratio(weights, returns, risk_free_rate=0.0):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return (portfolio_return - risk_free_rate) / portfolio_std

# Sortino Ratio
def sortino_ratio(weights, returns, risk_free_rate=0.0):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    negative_returns = returns[returns < risk_free_rate]
    downside_std = np.sqrt(np.dot(weights.T, np.dot(negative_returns.cov() * 252, weights)))
    return (portfolio_return - risk_free_rate) / downside_std

# Objective function: negative Sharpe Ratio (for maximization)
def neg_sharpe(weights, returns, risk_free_rate=0.0):
    return -sharpe_ratio(weights, returns, risk_free_rate)

# Constraints
def check_sum(weights):
    return np.sum(weights) - 1

# Portfolio optimization
def optimize_portfolio(returns, optimization_func, risk_free_rate=0.0):
    num_assets = len(returns.columns)
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': check_sum})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(optimization_func, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Main function
if __name__ == "__main__":
    # Define assets and time period
    tickers = ["IWF", "EEM", "SHYG", "MTUM"]
    start_date = "2001-01-01"
    end_date = "2022-12-31"

    # Download and process data
    data = download_data(tickers, start_date, end_date)

    # Optimize portfolio
    optimal_weights = optimize_portfolio(data, neg_sharpe)

    # Print optimized weights
    print("Optimized Weights:", optimal_weights.x)

    