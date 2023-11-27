import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

class PortfolioOptimization:
    """
    A class for portfolio optimization using financial data.

    Attributes:
        tickers (List[str]): List of ticker symbols for ETFs.
        start_date (str): Start date for data retrieval.
        end_date (str): End date for data retrieval.
        data (pd.DataFrame): Dataframe holding the financial data.
    """
