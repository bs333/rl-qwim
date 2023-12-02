import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to download data
def download_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end)['Adj Close']

# Sharpe and Sortino Ratio functions
def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

def sortino_ratio(returns, risk_free_rate=0.0):
    negative_returns = returns[returns < risk_free_rate]
    downside_std = negative_returns.std()
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / downside_std

# Objective function for optimization
def neg_sharpe(weights, returns, risk_free_rate=0.0):
    portfolio_returns = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return -(portfolio_returns - risk_free_rate) / portfolio_std

# Constraint: sum of weights is 1
def check_sum(weights):
    return np.sum(weights) - 1

# Portfolio optimization
def optimize_portfolio(returns, optimization_func, risk_free_rate=0.0):
    num_assets = len(returns.columns)
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': check_sum})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = [1. / num_assets] * num_assets

    result = minimize(optimization_func, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Y-axis formatter to show returns as '1x', '2x', etc.
def y_formatter(y, pos):
    return f'{y:.0f}x'

# Main execution
if __name__ == "__main__":
    tickers = ["IWF", "EEM", "SHYG", "MTUM"]
    start_date = "2001-01-01"
    end_date = "2022-12-31"

    data = download_data(tickers, start_date, end_date)
    daily_returns = data.pct_change().dropna()

    optimal_weights = optimize_portfolio(daily_returns, neg_sharpe)
    print("Optimized Weights:", optimal_weights)

    # Calculate portfolio daily returns
    portfolio_returns = (daily_returns * optimal_weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Calculate Sharpe and Sortino Ratios
    annual_sharpe = sharpe_ratio(portfolio_returns) * np.sqrt(252)
    annual_sortino = sortino_ratio(portfolio_returns) * np.sqrt(252)

    # Plotting
    plt.figure(figsize=(10, 6))
    cumulative_returns.plot()
    plt.title('Cumulative Returns of Optimized Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()

    print(f"Annualized Sharpe Ratio: {annual_sharpe}")
    print(f"Annualized Sortino Ratio: {annual_sortino}")