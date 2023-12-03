import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import random
from datetime import datetime, timedelta

from PPO import PPO

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
        # Fill NaNs with the mean of each column.
        self.data.fillna(self.data.mean())

        if self.data.isnull().any().any():
            print("NaNs found after filling missing values")

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
        for ticker in self.tickers:
            closing_prices = self.data['Close'][ticker]
            mean_val = closing_prices.mean()
            std_val = closing_prices.std()
            normalized_prices = (closing_prices - mean_val) / std_val

            if normalized_prices.isnull().any().any():
                print("NaNs found after normalization")

            # Create a new column for each ticker's normalized prices.
            self.data[f'Normalized_Close_{ticker}'] = normalized_prices

    def get_current_risk_free_rate(self, date: str):
        """
        Fetches the nearest available 10-Year US Treasury Yield (10Y UST) for the given date as the risk-free rate.

        Args:
            date (str): The date for which to retrieve the 10Y UST, in 'YYYY-MM-DD' format.

        Returns:
            float: The 10-Year US Treasury Yield for the given date, converted to a percentage.
        """
        # # Ticker symbol for the 10-Year Treasury Yield.
        # treasury_yield_ticker = yf.Ticker("^TNX")

        # # Convert the string date to a datetime object
        # target_date = datetime.strptime(date, '%Y-%m-%d')
        # delta = timedelta(days=1)

        # for _ in range(7):  # Check for a week in both directions.
        #     try:
        #         # Try fetching data for the target date.
        #         hist = treasury_yield_ticker.history(start=target_date, end=target_date)

        #         if not hist.empty:
        #             return hist['Close'].iloc[-1] / 100  # Return the yield if available.

        #         # If no data, check one day earlier and one day later.
        #         target_date -= delta  # Check one day earlier.

        #     except Exception as e:
        #         # Handle exceptions (e.g., connection issues, API limitations)..
        #         print(f"Error fetching data for date {target_date}: {e}")
        #         break  # Exit the loop if there's an error.

        # If no data is found after checking, assume 0.00 to be the risk-free-rate.
        return 0.00

    def calculate_reward(self, action: np.ndarray, current_prices: np.ndarray, date: str) -> float:
        """
        Calculates the reward based on the Sortino Ratio for the given action and current market prices.

        Args:
            action (np.ndarray): The action taken by the agent, representing the portfolio allocation.
            current_prices (np.ndarray): Current market prices of the assets.
            risk_free_rate (float): The risk-free rate for the period, default is 0.0.

        Returns:
            float: The calculated reward based on the Sortino Ratio.
        """

        if np.sum(action) == 0:
            print("Sum of actions is zero. Cannot normalize.")
        else:
            # Ensure the action sums up to 1 (100% of the portfolio).
            normalized_action = action / np.sum(action)

        # Calculate portfolio return.
        # Assuming current_prices are relative changes (e.g., today's price / yesterday's price)
        portfolio_return = np.sum(normalized_action * current_prices) - 1

        risk_free_rate = self.get_current_risk_free_rate(date)

        # Calculate the downside deviation (only consider negative returns).
        negative_returns = [min(0, r - risk_free_rate) for r in current_prices]
        downside_deviation = np.sqrt(np.mean(np.square(negative_returns)))

        # Avoid division by zero in case of no downside risk.
        if downside_deviation == 0:
            downside_deviation = 1e-6

        # Calculate the Sortino Ratio.
        sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation

        return sortino_ratio

    def train_agent(self, ppo_agent, episodes: int, training_interval: int):
        """
        Trains the PPO agent over a specified number of episodes.

        Args:
            ppo_agent (PPO): Instance of the PPO agent to be trained.
            episodes (int): The number of episodes to train the agent.
            training_interval (int): Number of steps to run before updating the PPO agent.
        """
        for episode in range(episodes):
            state, current_index = self.reset_environment()  # Reset the environment at the start of each episode.
            total_reward = 0

            states, actions, rewards, next_states, dones = [], [], [], [], []

            for step in range(training_interval):
                # Agent decides on the action based on the current state.
                action = ppo_agent.select_action(state)

                # Execute the action and get the next state and reward.
                next_state, reward, done = self.execute_action(action, current_index)
                next_index = current_index + 1  # Update the index for the next iteration.

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

        # Extract the window of data for each normalized column and concatenate them.
        initial_state_data = self.data.iloc[start_index:start_index + self.state_window]
        initial_state = np.concatenate([initial_state_data[f'Normalized_Close_{ticker}'].values for ticker in self.tickers], axis=0)

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
        if np.sum(action) == 0:
            print("Sum of actions is zero. Assigning default action.")
            # Establish a default non-zero action, for example, equal allocation to all assets.
            normalized_action = np.ones_like(action) / action
        else:
            # Normalize the action if the sum is not zero.
            normalized_action = action / np.sum(action)

        # Calculate next state
        next_index = current_index + 1
        if next_index + self.state_window >= len(self.data):
            done = True  # End of data
            next_state = None  # There's no next state if data ends
        else:
            done = False
            next_state_data = self.data.iloc[next_index:next_index + self.state_window]
            next_state_parts = [next_state_data[f'Normalized_Close_{ticker}'].values for ticker in self.tickers]
            next_state = np.concatenate(next_state_parts, axis=0)

        # Retrieve the date for the current index
        current_date = self.data.index[current_index].strftime('%Y-%m-%d')

        # Calculate reward
        # Assuming 'Close' column exists and contains closing prices
        current_prices = self.data['Close'].iloc[current_index]
        next_prices = self.data['Close'].iloc[next_index]
        price_change = next_prices / current_prices
        reward = self.calculate_reward(normalized_action, price_change, current_date)

        return next_state, reward, done

    def evaluate_agent(self, test_data: pd.DataFrame, ppo_agent: PPO, risk_free_rate: float = 0.0):
        """
        Evaluates the PPO agent using the testing dataset.

        Args:
            test_data (pd.DataFrame): The testing dataset.
            ppo_agent (PPO): The trained PPO agent.
            risk_free_rate (float): The risk-free rate for calculating risk-adjusted returns.
        """

        current_index = 0
        total_reward = 0
        portfolio_values = []
        negative_returns = []

        while current_index + self.state_window < len(test_data):
            # Extract the date for the current step.
            current_step_date = test_data.iloc[current_index].name.strftime('%Y-%m-%d')

            # Retrieve the risk-free rate for the current date.
            risk_free_rate = self.get_current_risk_free_rate(current_step_date)

            state_data = test_data.iloc[current_index:current_index + self.state_window]
            state = state_data['Normalized_Close'].values.flatten()

            action = ppo_agent.select_action(state)
            next_index = current_index + 1
            next_state, reward, done = self.execute_action(action, current_index, test_data)

            total_reward += reward
            current_index = next_index

            portfolio_value = 1 + total_reward
            portfolio_values.append(portfolio_value)

            # Track negative returns for Sortino Ratio.
            if reward < risk_free_rate:
                negative_returns.append(reward - risk_free_rate)

            if done:
                break

        final_portfolio_value = portfolio_values[-1]
        average_return = np.mean(portfolio_values)
        std_dev = np.std(portfolio_values)

        sharpe_ratio = (average_return - risk_free_rate) / std_dev if std_dev != 0 else 0

        # Calculate the Sortino Ratio.
        downside_deviation = np.sqrt(np.mean(np.square(negative_returns))) if negative_returns else 0
        sortino_ratio = (average_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0

        print(f'Final Portfolio Value: {final_portfolio_value}')
        print(f'Average Return: {average_return}')
        print(f'Sharpe Ratio: {sharpe_ratio}')
        print(f'Sortino Ratio: {sortino_ratio}')

if __name__ == '__main__':

    # Initialize PortfolioOptimization.
    portfolio_opt = PortfolioOptimization(tickers=["IWD", "IWF", "IWO", "EWJ"], 
                                          start_date="2001-01-01", 
                                          end_date="2022-12-31")

    # Load and preprocess data.
    portfolio_opt.load_data()
    portfolio_opt.clean_data()
    portfolio_opt.normalize_data()
    
    # Split data into training and testing.
    train_data, test_data = portfolio_opt.split_data("2020-01-01")

    # Setup environment.
    portfolio_opt.setup_environment()

    # Initialize the PPO agent.
    ppo_agent = PPO(portfolio_opt.state_window * len(portfolio_opt.tickers), 
                    portfolio_opt.num_assets, 
                    actor_lr=0.001, 
                    critic_lr=0.001, 
                    clip_ratio=0.2)

    # Train the PPO agent.
    portfolio_opt.train_agent(ppo_agent, 
                              episodes=10, 
                              training_interval=10)

    # Evaluate the trained agent.
    portfolio_opt.evaluate_agent(test_data, ppo_agent)




