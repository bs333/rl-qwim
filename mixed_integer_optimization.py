from portfolio_optimization import *
import pulp

portfolio_opt = PortfolioOptimization(tickers=["IWD", "IWF", "IWO", "EWJ","EEM"], 
                                        start_date="2004-01-01", 
                                        end_date="2022-12-31")

# Load and preprocess data.
portfolio_opt.load_data()
portfolio_opt.clean_data()
portfolio_opt.normalize_data()

# Split data into training and testing.
train_data, test_data = portfolio_opt.split_data("2020-01-01")

# make a new table for just daily returns, then use pulp to conduct the optimization

if __name__ == "__main__":
    print(train_data["Normalized_Close_EEM"])