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

    def __init__(self, tickers: list[str], start_date: str, end_date: str) -> None:
        """
        Initializes the LogisticRegressionPortfolioOptimizer with given tickers and date range.

        Args:
            tickers (List[str]): List of ticker symbols.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.models: Dict[str, Tuple[LogisticRegression, StandardScaler]] = {}

    def load_data(self) -> None:
        """Loads financial data from Yahoo Finance for the specified tickers and date range."""
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares the data for logistic regression by calculating daily returns and creating binary outcomes.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the features (previous day's returns) and outcomes (binary indicators of price movement direction).
        """
        # Calculate daily returns
        daily_returns = self.data['Close'].pct_change()

        # Calculate binary outcomes: 1 for positive return, 0 for negative
        outcomes = (daily_returns > 0).astype(int)

        # Use previous day's returns as features
        features = daily_returns.shift(1).fillna(0)

        return features, outcomes


