import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class LogisticRegressionPortfolioOptimizer:
    """
    A portfolio optimization class using logistic regression to predict the direction of stock price movement.
    
    Attributes:
        tickers (List[str]): List of ticker symbols to be included in the portfolio.
        start_date (str): The start date for historical data retrieval.
        end_date (str): The end date for historical data retrieval.
        data (pd.DataFrame): DataFrame containing the historical data.
        models (Dict[str, Tuple[LogisticRegression, StandardScaler]]): Dictionary mapping ticker symbols to tuples of trained logistic regression models and their associated scalers.
    """
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.models = {}