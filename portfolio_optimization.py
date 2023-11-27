import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import random

from . import PPO

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
    
    def split_data(self, split_date: str):
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

    def setup_environment(self):
        """
        Setup the environment for the PPO algorithm.
        This involves defining the state space, action space, and reward function.
        """
        # Example state space: window of past N days of normalized log returns
        self.state_window = 60  # Last 60 days of data
        self.normalize_data()   # Method to normalize data (to be implemented)

        # Example action space: allocation percentages for each ETF
        self.num_assets = len(self.tickers)
        self.action_space = np.linspace(0, 1, num=self.num_assets)  # Simple discrete allocation for each asset

        # Define the reward structure (to be implemented in a separate method)
        # This could be daily returns, Sharpe ratio, etc.
        self.reward_function = self.calculate_reward  # Placeholder for reward function method

    def normalize_data(self):
        """
        Normalizes the financial data using Z-score normalization.

        This method scales each feature (e.g., closing prices of each ETF) based on its mean and standard deviation.
        """
        closing_prices = self.data['Close']

        # Apply Z-score normalization
        mean_vals = closing_prices.mean()
        std_vals = closing_prices.std()
        normalized_data = (closing_prices - mean_vals) / std_vals

        self.data['Normalized_Close'] = normalized_data

    def calculate_reward(self, action: np.ndarray, current_prices: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the reward based on the Sortino Ratio for the given action and current market prices.

        Args:
            action (np.ndarray): The action taken by the agent, representing the portfolio allocation.
            current_prices (np.ndarray): Current market prices of the assets.
            risk_free_rate (float): The risk-free rate for the period, default is 0.0.

        Returns:
            float: The calculated reward based on the Sortino Ratio.
        """
        # Ensure the action sums up to 1 (100% of the portfolio).
        normalized_action = action / np.sum(action)

        # Calculate portfolio return.
        # Assuming current_prices are relative changes (e.g., today's price / yesterday's price)
        portfolio_return = np.sum(normalized_action * current_prices) - 1

        # Calculate the downside deviation (only consider negative returns).
        negative_returns = [min(0, r - risk_free_rate) for r in current_prices]
        downside_deviation = np.sqrt(np.mean(np.square(negative_returns)))

        # Avoid division by zero in case of no downside risk.
        if downside_deviation == 0:
            downside_deviation = 1e-6

        # Calculate the Sortino Ratio.
        sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation

        return sortino_ratio

    def train_agent(self, episodes: int, actor_lr: float, critic_lr: float, clip_ratio: float, training_interval: int):
        """
        Trains the PPO agent over a specified number of episodes.

        Args:
            episodes (int): The number of episodes to train the agent.
            actor_lr (float): Learning rate for the actor.
            critic_lr (float): Learning rate for the critic.
            clip_ratio (float): Clipping ratio for the PPO algorithm.
            training_interval (int): Number of steps to run before updating the PPO agent.
        """
        # Initialize the PPO agent.
        ppo_agent = PPO(self.state_window * len(self.tickers), self.num_assets, actor_lr, critic_lr, clip_ratio)

        for episode in range(episodes):
            state = self.reset_environment()  # Reset the environment at the start of each episode.
            total_reward = 0

            states, actions, rewards, next_states, dones = [], [], [], [], []

            for step in range(training_interval):
                # Agent decides on the action based on the current state.
                action = ppo_agent.select_action(state)

                # Execute the action and get the next state and reward.
                next_state, reward, done = self.execute_action(action)

                # Store this transition.
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                total_reward += reward
                state = next_state

                if done:
                    break

            # Prepare data for PPO training.
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)

            # Train the PPO agent.
            ppo_agent.train(states, actions, rewards, next_states, dones)

            print(f'Episode {episode + 1}, Total Reward: {total_reward}')

    def reset_environment(self):
        """
        Initializes the environment to an initial state at the start of an episode.
        Also returns the starting index in the dataset for this episode.

        Returns:
            initial_state (np.ndarray): The initial state of the environment.
            start_index (int): The starting index in the dataset for this episode.
        """
        max_start_index = len(self.data) - self.state_window
        start_index = random.randint(0, max_start_index)

        initial_state_data = self.data.iloc[start_index:start_index + self.state_window]
        initial_state = initial_state_data['Normalized_Close'].values.flatten()

        return initial_state, start_index

    def execute_action(self, action: np.ndarray, current_index: int) -> (np.ndarray, float, bool):
        """
        Executes the given action in the environment and returns the next state, reward, and done flag.

        Args:
            action (np.ndarray): The action to be executed, representing portfolio allocations.
            current_index (int): The current index in the dataset.

        Returns:
            next_state (np.ndarray): The next state of the environment after taking the action.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended.
        """
        # Ensure action is normalized (sums to 1)
        normalized_action = action / np.sum(action)

        # Calculate next state
        next_index = current_index + 1
        if next_index + self.state_window >= len(self.data):
            done = True  # End of data
            next_state = None  # There's no next state if data ends
        else:
            done = False
            next_state_data = self.data.iloc[next_index:next_index + self.state_window]
            next_state = next_state_data['Normalized_Close'].values.flatten()

        # Calculate reward
        # Assuming 'Close' column exists and contains closing prices
        current_prices = self.data['Close'].iloc[current_index]
        next_prices = self.data['Close'].iloc[next_index]
        price_change = next_prices / current_prices
        reward = self.calculate_reward(normalized_action, price_change)

        return next_state, reward, done

if __name__ == '__main__':
    # List of ETF tickers.
    tickers = ["IWF", "EEM", "SHYG", "MTUM"]

    # Create an instance of the PortfolioOptimization class.
    portfolio_opt = PortfolioOptimization(tickers, "2001-01-01", "2022-12-31")

    # Load and clean the data.
    portfolio_opt.load_data()
    portfolio_opt.clean_data()

    # Perform exploratory data analysis.
    portfolio_opt.plot_closing_prices()
    portfolio_opt.show_statistics()

    # Perform stationarity test on the daily log returns.
    portfolio_opt.test_stationarity()

    # Split the data into training and testing sets.
    train_data, test_data = portfolio_opt.split_data("2020-01-01")


