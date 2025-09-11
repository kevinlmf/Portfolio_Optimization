"""
FinRL-style Portfolio Optimization Environment
A Gym-compatible environment for portfolio optimization using reinforcement learning.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizationEnv(gym.Env):
    """
    A portfolio optimization environment following FinRL conventions.
    
    The environment simulates a portfolio management task where:
    - State: Market features, portfolio weights, account balance
    - Action: New portfolio weight allocation
    - Reward: Risk-adjusted returns (Sharpe ratio, Calmar ratio, etc.)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 1}
    
    def __init__(self,
                 df: pd.DataFrame,
                 initial_amount: float = 1000000,
                 lookback_window: int = 20,
                 transaction_cost_pct: float = 0.001,
                 reward_scaling: float = 1.0,
                 mode: str = 'train',
                 features: List[str] = None,
                 print_verbosity: int = 10,
                 day: int = 0,
                 initial: bool = True,
                 previous_state: List = [],
                 model_name: str = '',
                 iteration: str = ''):
        """
        Initialize the Portfolio Optimization Environment.
        
        Args:
            df: Market data with columns ['date', 'tic', 'close', 'volume', ...]
            initial_amount: Initial capital
            lookback_window: Number of days to look back for state features
            transaction_cost_pct: Transaction cost as percentage
            reward_scaling: Scaling factor for rewards
            mode: 'train' or 'test' mode
            features: List of feature columns to use
            print_verbosity: How often to print progress
            day: Starting day index
            initial: Whether this is initial state
            previous_state: Previous state for continuation
            model_name: Name of the model
            iteration: Iteration identifier
        """
        self.df = df.copy()
        self.initial_amount = initial_amount
        self.lookback_window = lookback_window
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.mode = mode
        self.print_verbosity = print_verbosity
        self.model_name = model_name
        self.iteration = iteration
        
        # Data preprocessing
        self.tic_list = sorted(self.df['tic'].unique())
        self.stock_dim = len(self.tic_list)
        self.df = self.df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # Default features if not provided
        if features is None:
            self.tech_indicator_list = ['close', 'volume', 'high', 'low', 'open']
        else:
            self.tech_indicator_list = features
            
        self.feature_dim = len(self.tech_indicator_list)
        
        # Environment state variables
        self.day = day
        self.data = self.df.loc[self.day:(self.day + self.lookback_window - 1), :]
        self.terminal = False
        self.initial = initial
        self.previous_state = previous_state
        
        # Portfolio state
        self.current_portfolio_value = self.initial_amount
        self.current_weights = np.array([1.0] + [0.0] * self.stock_dim)  # [cash, stocks...]
        self.cash_balance = self.initial_amount
        self.stock_shares = np.zeros(self.stock_dim)
        
        # Performance tracking
        self.portfolio_value_history = [self.initial_amount]
        self.portfolio_return_history = [0]
        self.actions_history = []
        self.reward_history = []
        self.date_history = [self._get_current_date()]
        
        # Action space: portfolio weights (including cash)
        # Sum of weights should be 1, each weight between 0 and 1
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.stock_dim + 1,),  # +1 for cash
            dtype=np.float32
        )
        
        # Observation space
        # [portfolio_weights, current_prices, technical_indicators, portfolio_stats]
        obs_dim = (
            (self.stock_dim + 1) +  # portfolio weights
            self.stock_dim +        # current prices
            (self.feature_dim * self.stock_dim) +  # technical indicators
            5  # portfolio statistics (value, return, volatility, sharpe, max_drawdown)
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.state_dim = obs_dim
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset to initial conditions
        self.day = 0
        self.data = self.df.loc[self.day:(self.day + self.lookbook_window - 1), :]
        self.terminal = False
        
        self.current_portfolio_value = self.initial_amount
        self.current_weights = np.array([1.0] + [0.0] * self.stock_dim)
        self.cash_balance = self.initial_amount
        self.stock_shares = np.zeros(self.stock_dim)
        
        # Reset tracking
        self.portfolio_value_history = [self.initial_amount]
        self.portfolio_return_history = [0]
        self.actions_history = []
        self.reward_history = []
        self.date_history = [self._get_current_date()]
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action):
        """Execute one step in the environment."""
        self.terminal = self.day >= len(self.df.index.unique()) - self.lookback_window - 1
        
        if self.terminal:
            # Terminal state
            final_reward = self._calculate_terminal_reward()
            state = self._get_state()
            info = self._get_info()
            info['final_portfolio_value'] = self.current_portfolio_value
            info['total_return'] = (self.current_portfolio_value / self.initial_amount - 1) * 100
            
            if self.mode == 'train':
                print(f"=== Episode Complete ===")
                print(f"Final Portfolio Value: ${self.current_portfolio_value:,.2f}")
                print(f"Total Return: {info['total_return']:.2f}%")
                print(f"Sharpe Ratio: {self._calculate_sharpe_ratio():.4f}")
                print(f"Max Drawdown: {self._calculate_max_drawdown():.2f}%")
                
            return state, final_reward, True, False, info
        
        # Normalize action to ensure weights sum to 1
        action = np.array(action)
        action = np.abs(action)  # Ensure non-negative
        action = action / np.sum(action)  # Normalize to sum to 1
        
        # Store current state for comparison
        previous_portfolio_value = self.current_portfolio_value
        previous_weights = self.current_weights.copy()
        
        # Execute trading action
        self._execute_trades(action)
        
        # Move to next day
        self.day += 1
        self.data = self.df.loc[self.day:(self.day + self.lookback_window - 1), :]
        
        # Update portfolio value based on market movement
        self._update_portfolio_value()
        
        # Calculate reward
        reward = self._calculate_reward(previous_portfolio_value)
        
        # Update tracking
        portfolio_return = (self.current_portfolio_value / previous_portfolio_value - 1)
        self.portfolio_value_history.append(self.current_portfolio_value)
        self.portfolio_return_history.append(portfolio_return)
        self.actions_history.append(action)
        self.reward_history.append(reward)
        self.date_history.append(self._get_current_date())
        
        # Get new state
        state = self._get_state()
        info = self._get_info()
        
        # Print progress
        if self.day % self.print_verbosity == 0 and self.mode == 'train':
            print(f"Day {self.day}: Portfolio Value = ${self.current_portfolio_value:,.2f}, "
                  f"Return = {portfolio_return:.4f}, Reward = {reward:.4f}")
        
        return state, reward, False, False, info
    
    def _execute_trades(self, target_weights):
        """Execute trades to achieve target portfolio weights."""
        target_weights = np.array(target_weights)
        current_prices = self._get_current_prices()
        
        # Calculate target values for each position
        target_cash = target_weights[0] * self.current_portfolio_value
        target_stock_values = target_weights[1:] * self.current_portfolio_value
        target_stock_shares = target_stock_values / current_prices
        
        # Calculate trades needed
        cash_trade = target_cash - self.cash_balance
        share_trades = target_stock_shares - self.stock_shares
        
        # Calculate transaction costs
        total_trade_value = np.sum(np.abs(share_trades) * current_prices) + np.abs(cash_trade)
        transaction_costs = total_trade_value * self.transaction_cost_pct
        
        # Execute trades
        self.cash_balance = target_cash - transaction_costs
        self.stock_shares = target_stock_shares
        self.current_weights = target_weights
        
        # Update portfolio value after transaction costs
        self.current_portfolio_value -= transaction_costs
        
    def _update_portfolio_value(self):
        """Update portfolio value based on current market prices."""
        current_prices = self._get_current_prices()
        stock_values = self.stock_shares * current_prices
        self.current_portfolio_value = self.cash_balance + np.sum(stock_values)
        
        # Update current weights
        if self.current_portfolio_value > 0:
            self.current_weights[0] = self.cash_balance / self.current_portfolio_value
            self.current_weights[1:] = stock_values / self.current_portfolio_value
    
    def _get_current_prices(self):
        """Get current stock prices."""
        if len(self.data) == 0:
            return np.ones(self.stock_dim)  # Fallback
        
        current_data = self.data.groupby('tic')['close'].last()
        prices = np.array([current_data.get(tic, 1.0) for tic in self.tic_list])
        return prices
    
    def _get_current_date(self):
        """Get current date."""
        if len(self.data) == 0:
            return datetime.now().strftime('%Y-%m-%d')
        return self.data['date'].iloc[-1]
    
    def _get_state(self):
        """Get current state observation."""
        # Portfolio weights
        weights = self.current_weights
        
        # Current prices (normalized)
        prices = self._get_current_prices()
        if len(self.portfolio_value_history) > 1:
            price_changes = prices / np.array([1.0] * len(prices))  # Simplified
        else:
            price_changes = np.ones_like(prices)
        
        # Technical indicators
        tech_features = []
        if len(self.data) > 0:
            for feature in self.tech_indicator_list:
                feature_data = self.data.groupby('tic')[feature].last()
                feature_values = np.array([feature_data.get(tic, 0.0) for tic in self.tic_list])
                tech_features.extend(feature_values)
        else:
            tech_features = [0.0] * (self.feature_dim * self.stock_dim)
        
        # Portfolio statistics
        portfolio_return = 0 if len(self.portfolio_return_history) < 2 else self.portfolio_return_history[-1]
        portfolio_volatility = np.std(self.portfolio_return_history) if len(self.portfolio_return_history) > 1 else 0
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        
        portfolio_stats = [
            self.current_portfolio_value / self.initial_amount,  # Normalized portfolio value
            portfolio_return,
            portfolio_volatility,
            sharpe_ratio,
            max_drawdown / 100.0  # Normalized max drawdown
        ]
        
        # Combine all features
        state = np.concatenate([
            weights,
            price_changes,
            tech_features,
            portfolio_stats
        ])
        
        return state.astype(np.float32)
    
    def _calculate_reward(self, previous_portfolio_value):
        """Calculate step reward."""
        # Portfolio return
        portfolio_return = (self.current_portfolio_value / previous_portfolio_value - 1)
        
        # Risk adjustment
        if len(self.portfolio_return_history) > 20:  # Need enough history
            volatility = np.std(self.portfolio_return_history[-20:])
            if volatility > 0:
                risk_adjusted_return = portfolio_return / volatility
            else:
                risk_adjusted_return = portfolio_return
        else:
            risk_adjusted_return = portfolio_return
        
        # Diversification bonus
        diversification_bonus = 0
        if np.sum(self.current_weights[1:]) > 0:  # If invested in stocks
            # Reward diversification (penalize concentration)
            concentration = np.sum(self.current_weights[1:]**2)
            diversification_bonus = -concentration * 0.1
        
        # Combine components
        reward = risk_adjusted_return + diversification_bonus
        
        return reward * self.reward_scaling
    
    def _calculate_terminal_reward(self):
        """Calculate reward at episode termination."""
        total_return = (self.current_portfolio_value / self.initial_amount - 1)
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown() / 100.0
        
        # Terminal reward combines multiple metrics
        terminal_reward = (
            total_return * 2.0 +          # Total return
            sharpe_ratio * 1.0 +          # Risk-adjusted performance
            max_drawdown * (-0.5)         # Penalty for large drawdowns
        )
        
        return terminal_reward * self.reward_scaling
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio of the portfolio."""
        if len(self.portfolio_return_history) < 2:
            return 0.0
        
        returns = np.array(self.portfolio_return_history[1:])  # Exclude initial 0
        if np.std(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = excess_returns / np.std(returns) * np.sqrt(252)
        
        return sharpe_ratio
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown percentage."""
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        portfolio_values = np.array(self.portfolio_value_history)
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max * 100
        
        return np.min(drawdowns)
    
    def _get_info(self):
        """Get additional information."""
        return {
            'day': self.day,
            'portfolio_value': self.current_portfolio_value,
            'portfolio_weights': self.current_weights.copy(),
            'cash_balance': self.cash_balance,
            'stock_shares': self.stock_shares.copy(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'total_return': (self.current_portfolio_value / self.initial_amount - 1) * 100
        }
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"\n=== Day {self.day} ===")
            print(f"Date: {self._get_current_date()}")
            print(f"Portfolio Value: ${self.current_portfolio_value:,.2f}")
            print(f"Portfolio Weights: {self.current_weights}")
            print(f"Sharpe Ratio: {self._calculate_sharpe_ratio():.4f}")
            print(f"Max Drawdown: {self._calculate_max_drawdown():.2f}%")
            
        elif mode == 'rgb_array':
            return self._render_plot()
    
    def _render_plot(self):
        """Create a plot of portfolio performance."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Portfolio value over time
        ax1.plot(self.portfolio_value_history)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # Portfolio weights over time
        if len(self.actions_history) > 0:
            weights_history = np.array(self.actions_history)
            for i, tic in enumerate(['Cash'] + self.tic_list):
                if i < weights_history.shape[1]:
                    ax2.plot(weights_history[:, i], label=tic)
        
        ax2.set_title('Portfolio Weights Over Time')
        ax2.set_ylabel('Weight')
        ax2.set_xlabel('Time Steps')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Convert plot to RGB array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return buf
    
    def get_sb_env(self):
        """Get environment compatible with stable-baselines3."""
        env = DummyVecEnv([lambda: self])
        return env
    
    def save_asset_memory(self):
        """Save portfolio performance data."""
        if len(self.portfolio_value_history) <= 1:
            return pd.DataFrame()
        
        df_account_value = pd.DataFrame({
            'date': self.date_history,
            'account_value': self.portfolio_value_history
        })
        
        return df_account_value
    
    def save_action_memory(self):
        """Save action history."""
        if len(self.actions_history) == 0:
            return pd.DataFrame()
        
        df_actions = pd.DataFrame(self.actions_history)
        df_actions.columns = ['cash'] + self.tic_list
        df_actions['date'] = self.date_history[1:]  # Actions start from day 1
        
        return df_actions


# Utility function for creating vectorized environment
try:
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    # Fallback if stable-baselines3 not available
    class DummyVecEnv:
        def __init__(self, env_list):
            self.env = env_list[0]()
            
        def reset(self):
            return self.env.reset()
            
        def step(self, action):
            return self.env.step(action)
            
        def render(self, mode='human'):
            return self.env.render(mode)


def create_portfolio_env(df: pd.DataFrame, **kwargs) -> PortfolioOptimizationEnv:
    """
    Factory function to create portfolio optimization environment.
    
    Args:
        df: Market data DataFrame
        **kwargs: Additional environment parameters
        
    Returns:
        Portfolio optimization environment
    """
    return PortfolioOptimizationEnv(df, **kwargs)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('/Users/mengfanlong/Downloads/Portfolio_Optimization_system')
    
    from data.real_data import RealDataFetcher
    
    # Create sample data
    fetcher = RealDataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    # Get price data
    prices = fetcher.get_stock_data(tickers)
    
    # Create DataFrame in required format
    df_list = []
    for date in prices.index:
        for tic in tickers:
            row = {
                'date': date,
                'tic': tic,
                'close': prices.loc[date, tic],
                'open': prices.loc[date, tic] * 0.99,  # Simplified
                'high': prices.loc[date, tic] * 1.02,
                'low': prices.loc[date, tic] * 0.98,
                'volume': 1000000  # Simplified
            }
            df_list.append(row)
    
    df = pd.DataFrame(df_list)
    
    # Create environment
    env = PortfolioOptimizationEnv(
        df=df,
        initial_amount=100000,
        lookback_window=10,
        transaction_cost_pct=0.001
    )
    
    print("Portfolio Optimization Environment Created!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of stocks: {env.stock_dim}")
    
    # Test environment
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Random action test
    for i in range(5):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward = {reward:.4f}, Portfolio Value = ${info['portfolio_value']:,.2f}")
        
        if terminated:
            break
    
    env.render()