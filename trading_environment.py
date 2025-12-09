
# trading_environment.py
# Custom Gymnasium environment for stock trading

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class TradingEnvironment(gym.Env):
    """
    Custom Trading Environment that follows Gymnasium interface

    State Space:
        - OHLCV features
        - Technical indicators (RSI, MACD, etc.)
        - Portfolio state (position, cash, unrealized P&L)

    Action Space:
        - Discrete: {0: Hold, 1: Buy, 2: Sell}
        - Or Continuous: [-1, 1] representing position sizing

    Reward:
        - Risk-adjusted portfolio returns with transaction costs
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        reward_scaling: float = 1e-4,
        window_size: int = 20,
        action_space_type: str = 'discrete',  # 'discrete' or 'continuous'
        features: Optional[list] = None
    ):
        """
        Initialize trading environment

        Args:
            df: DataFrame with OHLCV and technical indicators
            initial_balance: Starting cash balance
            transaction_cost: Trading fee as fraction (0.001 = 0.1%)
            reward_scaling: Scale factor for rewards
            window_size: Number of time steps to include in state
            action_space_type: 'discrete' or 'continuous'
            features: List of feature column names to use
        """
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.action_space_type = action_space_type

        # Select features for state space
        if features is None:
            # Default features: returns, technical indicators
            self.features = [
                'Returns', 'Log_Returns', 'RSI', 'MACD', 'MACD_Hist',
                'BB_Position', 'Volume_Ratio', 'Volatility', 'Momentum'
            ]
        else:
            self.features = features

        # Verify features exist
        missing_features = [f for f in self.features if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataframe: {missing_features}")

        # Define action space
        if action_space_type == 'discrete':
            # 0: Hold, 1: Buy, 2: Sell
            self.action_space = spaces.Discrete(3)
        else:
            # Continuous: [-1, 1] where -1=sell all, 0=hold, 1=buy all
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        # Define observation space
        # State includes: windowed features + portfolio state (3 values)
        n_features = len(self.features) * window_size + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Performance tracking
        self.portfolio_values = []
        self.trades = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment to initial state

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset to random starting point (but ensure enough history)
        max_start = len(self.df) - 200  # Keep at least 200 steps for episode
        self.current_step = np.random.randint(self.window_size, max(self.window_size + 1, max_start))

        # Reset portfolio
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Reset tracking
        self.portfolio_values = [self.initial_balance]
        self.trades = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment

        Args:
            action: Action to take

        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Get current price
        current_price = self.df.loc[self.current_step, 'Close']

        # Execute action
        if self.action_space_type == 'discrete':
            action_type = action
        else:
            # Convert continuous action to discrete
            action = action[0] if isinstance(action, np.ndarray) else action
            if action < -0.33:
                action_type = 2  # Sell
            elif action > 0.33:
                action_type = 1  # Buy
            else:
                action_type = 0  # Hold

        # Calculate shares to trade
        shares_traded = 0
        cost = 0

        if action_type == 1:  # Buy
            # Buy as many shares as possible with available balance
            max_shares = int(self.balance / (current_price * (1 + self.transaction_cost)))
            if max_shares > 0:
                shares_traded = max_shares
                cost = shares_traded * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += shares_traded
                self.trades.append({
                    'step': self.current_step,
                    'type': 'buy',
                    'shares': shares_traded,
                    'price': current_price,
                    'cost': cost
                })

        elif action_type == 2:  # Sell
            # Sell all held shares
            if self.shares_held > 0:
                shares_traded = self.shares_held
                revenue = shares_traded * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.total_shares_sold += shares_traded
                self.total_sales_value += revenue
                self.shares_held = 0
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'shares': shares_traded,
                    'price': current_price,
                    'revenue': revenue
                })

        # Move to next step
        self.current_step += 1

        # Calculate portfolio value
        next_price = self.df.loc[self.current_step, 'Close']
        portfolio_value = self.balance + self.shares_held * next_price
        self.portfolio_values.append(portfolio_value)

        # Calculate reward (risk-adjusted returns)
        reward = self._calculate_reward()

        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Get next observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Get current observation (state)

        Returns:
            np.array: State vector
        """
        # Get windowed features
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step

        window_data = self.df.loc[start_idx:end_idx-1, self.features].values
        window_data = window_data.flatten()

        # Replace NaN/inf with 0
        window_data = np.nan_to_num(window_data, nan=0.0, posinf=1.0, neginf=-1.0)

        # Portfolio state
        current_price = self.df.loc[self.current_step, 'Close']
        portfolio_value = self.balance + self.shares_held * current_price

        portfolio_state = np.array([
            self.shares_held / 100.0,  # Normalized shares held
            self.balance / self.initial_balance,  # Normalized cash
            portfolio_value / self.initial_balance  # Normalized portfolio value
        ])

        # Combine
        observation = np.concatenate([window_data, portfolio_state])

        return observation.astype(np.float32)

    def _calculate_reward(self):
        """
        Calculate reward based on portfolio performance
        Returns:
            float: Reward value
        """
        # Simple return-based reward
        if len(self.portfolio_values) < 2:
            return 0.0

        # Portfolio return for current step
        current_value = self.portfolio_values[-1]
        previous_value = self.portfolio_values[-2]
        returns = (current_value - previous_value) / (previous_value + 1e-8)

        # Risk adjustment using recent volatility
        if len(self.portfolio_values) >= 21:
            # Use last 21 values to get 20 returns
            recent_vals = np.array(self.portfolio_values[-21:], dtype=np.float32)
            recent_returns = np.diff(recent_vals) / (recent_vals[:-1] + 1e-8)
            volatility = np.std(recent_returns)
            risk_adjusted_return = returns / (volatility + 1e-6)
        else:
            risk_adjusted_return = returns

        # Scale reward
        reward = risk_adjusted_return * self.reward_scaling
        return float(reward)

    def _get_info(self):
        """
        Get additional information about current state

        Returns:
            dict: Information dictionary
        """
        current_price = self.df.loc[self.current_step, 'Close']
        portfolio_value = self.balance + self.shares_held * current_price

        info = {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'total_trades': len(self.trades)
        }

        return info

    def render(self, mode='human'):
        """
        Render environment state
        """
        current_price = self.df.loc[self.current_step, 'Close']
        portfolio_value = self.balance + self.shares_held * current_price
        profit = portfolio_value - self.initial_balance

        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Shares: {self.shares_held}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Profit: ${profit:.2f} ({(profit/self.initial_balance)*100:.2f}%)")
        print("-" * 50)


# Example usage
if __name__ == "__main__":
    # Load data
    import pandas as pd
    df = pd.read_csv('aapl_processed.csv', index_col=0)

    # Create environment
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        transaction_cost=0.001,
        action_space_type='discrete'
    )

    # Test environment
    from gymnasium.utils.env_checker import check_env
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")

    # Run random episode
    obs, info = env.reset()
    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            break
