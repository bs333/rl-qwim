import numpy as np
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

    def __init__(self, tickers: list, start_date: str, end_date: str):
        """
        Initializes the PortfolioOptimization class with given tickers and date range.

        Args:
            tickers (List[str]): List of ticker symbols.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
    
    def load_data(self):
        """
        Loads financial data from Yahoo Finance for the specified tickers and date range.
        """
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)

    def clean_data(self):
        """
        Cleans the loaded data by filling missing values.
        
        Uses forward fill to handle missing values, which is common in financial time series data.
        """
        self.data.fillna(method='ffill', inplace=True)

    def plot_closing_prices(self):
        """
        Plots the closing prices of the ETFs.
        """
        self.data['Close'].plot(figsize=(15, 7))
        plt.title('Closing Prices of ETFs')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

    def show_statistics(self):
        """
        Displays summary statistics of the data.
        """
        print(self.data.describe())

    def test_stationarity(self):
        """
        Performs stationarity tests on the daily log returns of each ETF.

        Uses the Augmented Dickey-Fuller test to check for stationarity of log returns.
        """
        for ticker in self.tickers:
            print(f'Stationarity test for daily log returns of {ticker}:')
            closing_prices = self.data['Close'][ticker]
            log_returns = np.log(closing_prices / closing_prices.shift(1))
            self._adf_test(log_returns)
    
    def _adf_test(self, timeseries: pd.Series):
        """
        Private method to perform the Augmented Dickey-Fuller test.

        Args:
            timeseries (pd.Series): Time series data for a single ETF.
        """
        result = adfuller(timeseries.dropna())
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value:.3f}')
        print()
    
    def split_data(self, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing sets based on a specified date.

        Args:
            split_date (str): Date to split the data on in 'YYYY-MM-DD' format.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing data sets.
        """
        train_data = self.data[:split_date]
        test_data = self.data[split_date:]
        return train_data, test_data








