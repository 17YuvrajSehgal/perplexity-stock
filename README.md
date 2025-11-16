# Deep Reinforcement Learning Trading System

A complete implementation of a deep reinforcement learning system for stock trading using PPO (Proximal Policy Optimization).

## 🎯 Overview

This system implements the recommended approach from state-of-the-art research:
- **Continuous state space** with neural network function approximation
- **Risk-adjusted rewards** (Sharpe-based with transaction costs)
- **Technical indicators** as state features
- **PPO algorithm** for stable training
- **Walk-forward validation** for realistic backtesting

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   yfinance   │→ │  Technical   │→ │ Normalization│     │
│  │  OHLCV Data  │  │  Indicators  │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              TRADING ENVIRONMENT (Gymnasium)                 │
│  State: OHLCV + Indicators + Portfolio State                │
│  Action: Buy/Sell/Hold (Discrete) or Position Size (Cont.)  │
│  Reward: Risk-adjusted Returns - Transaction Costs          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PPO AGENT                                 │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │    Actor     │  │    Critic    │                        │
│  │   Network    │  │   Network    │                        │
│  │ (Policy π)   │  │  (Value V)   │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              EVALUATION & BACKTESTING                        │
│  Metrics: Sharpe, Sortino, Max Drawdown, Win Rate          │
│  Baseline: Buy-and-Hold Comparison                          │
│  Visualization: Portfolio value, trades, drawdown           │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Step 1: Clone/Download the Repository
```bash
# If using git
git clone <repository_url>
cd deep-rl-trading

# Or simply navigate to the directory with the files
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate

# Or using conda
conda create -n trading_env python=3.9
conda activate trading_env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Optional: Install TA-Lib (for advanced indicators)
The system uses pandas for technical indicators by default. For TA-Lib:

**Linux/Mac:**
```bash
# Install C library first
# Ubuntu/Debian
sudo apt-get install ta-lib

# macOS
brew install ta-lib

# Then install Python wrapper
pip install TA-Lib
```

**Windows:**
Download wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and:
```bash
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl
```

## 🚀 Quick Start

### Basic Usage (Default Settings)
```bash
python main.py --ticker AAPL --total_timesteps 100000
```

This will:
1. Download AAPL data from 2020-2024
2. Add technical indicators
3. Train PPO agent for 100k steps
4. Backtest on test set
5. Generate performance plots and metrics

### Custom Configuration
```bash
python main.py \
    --ticker TSLA \
    --start_date 2019-01-01 \
    --end_date 2024-01-01 \
    --initial_balance 50000 \
    --transaction_cost 0.002 \
    --total_timesteps 200000 \
    --learning_rate 0.0003 \
    --action_space continuous
```

### Available Arguments
```
Data Arguments:
  --ticker              Stock ticker symbol (default: AAPL)
  --start_date          Start date YYYY-MM-DD (default: 2020-01-01)
  --end_date            End date YYYY-MM-DD (default: 2024-01-01)
  --interval            Data interval: 1d, 1h, etc. (default: 1d)
  --data_file           Path to existing processed CSV

Environment Arguments:
  --initial_balance     Starting cash (default: 10000)
  --transaction_cost    Trading fee fraction (default: 0.001 = 0.1%)
  --window_size         Observation window size (default: 20)
  --action_space        discrete or continuous (default: discrete)

Training Arguments:
  --train_split         Train/test split (default: 0.8)
  --total_timesteps     Training steps (default: 100000)
  --learning_rate       Learning rate (default: 3e-4)
  --n_steps             Steps per update (default: 2048)
  --batch_size          Mini-batch size (default: 64)
  --n_epochs            Epochs per update (default: 10)
  --gamma               Discount factor (default: 0.99)
```

## 📊 Module Details

### 1. data_collection.py
Handles data download and preprocessing:
- Downloads OHLCV data from Yahoo Finance
- Calculates technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Normalizes features to prevent look-ahead bias
- Saves processed data to CSV

**Usage:**
```python
from data_collection import DataCollector

collector = DataCollector('AAPL', start_date='2020-01-01')
collector.download_data()
collector.add_technical_indicators()
data = collector.prepare_data()
```

### 2. trading_environment.py
Custom Gymnasium environment:
- Implements standard gym.Env interface
- Configurable discrete/continuous action spaces
- Risk-adjusted reward function
- Transaction cost modeling
- Portfolio state tracking

**Usage:**
```python
from trading_environment import TradingEnvironment

env = TradingEnvironment(
    df=data,
    initial_balance=10000,
    transaction_cost=0.001
)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### 3. ppo_agent.py
PPO agent implementation:
- Wraps Stable-Baselines3 PPO
- Custom callback for trading metrics
- VecNormalize for observation/reward normalization
- Backtest functionality
- Model save/load

**Usage:**
```python
from ppo_agent import PPOTradingAgent

agent = PPOTradingAgent(env=env, learning_rate=3e-4)
agent.train(total_timesteps=100000)
results = agent.backtest(test_env, n_episodes=1)
```

### 4. evaluation.py
Performance evaluation:
- Comprehensive metrics (Sharpe, Sortino, Calmar, Max Drawdown)
- Buy-and-hold baseline comparison
- Win rate and profit factor
- Multi-panel visualization
- CSV export

**Usage:**
```python
from evaluation import TradingEvaluator

evaluator = TradingEvaluator(initial_balance=10000)
metrics = evaluator.calculate_metrics(portfolio_values, trades)
evaluator.plot_performance(portfolio_values, baseline_values, prices, trades)
```

### 5. main.py
End-to-end pipeline:
- Orchestrates all modules
- Command-line interface
- Automatic train/test splitting
- Results saving

## 📈 Performance Metrics

The system calculates:

- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return vs maximum drawdown
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Volatility**: Annualized standard deviation of returns

## 🎨 Visualization

The system generates a comprehensive 6-panel plot:
1. Portfolio value over time (vs buy-and-hold)
2. Returns distribution
3. Cumulative returns
4. Drawdown chart
5. Stock price with trade markers
6. Rolling Sharpe ratio

## 🔬 Advanced Usage

### Using Pre-collected Data
```python
# If you already have processed data
python main.py --data_file my_stock_data.csv --total_timesteps 200000
```

### Training Multiple Stocks
```bash
for ticker in AAPL MSFT GOOGL TSLA; do
    python main.py --ticker $ticker --total_timesteps 150000
done
```

### Continuous Action Space
```bash
python main.py --action_space continuous --ticker SPY
```

### Hyperparameter Tuning
```bash
# Experiment with different learning rates
for lr in 0.0001 0.0003 0.001; do
    python main.py --learning_rate $lr --ticker AAPL
done
```

## 🧪 Testing Your Setup

After installation, run a quick test:

```python
# test_setup.py
from trading_environment import TradingEnvironment
from gymnasium.utils.env_checker import check_env
import pandas as pd
import numpy as np

# Create dummy data
dates = pd.date_range('2020-01-01', periods=500)
df = pd.DataFrame({
    'Close': np.random.randn(500).cumsum() + 100,
    'Returns': np.random.randn(500) * 0.01,
    'Log_Returns': np.random.randn(500) * 0.01,
    'RSI': np.random.rand(500) * 100,
    'MACD': np.random.randn(500),
    'MACD_Hist': np.random.randn(500),
    'BB_Position': np.random.rand(500),
    'Volume_Ratio': np.random.rand(500) * 2,
    'Volatility': np.random.rand(500) * 0.02,
    'Momentum': np.random.randn(500)
}, index=dates)

env = TradingEnvironment(df=df)
check_env(env)
print("✓ Environment setup successful!")
```

## 📚 Key Concepts

### State Space Design
The agent observes:
- **Historical prices**: 20-day window of returns and log-returns
- **Technical indicators**: RSI, MACD, Bollinger Bands position, volume ratio
- **Portfolio state**: Current position, cash balance, portfolio value (normalized)

### Reward Function
```
reward = (portfolio_return / recent_volatility) * scaling_factor
```
This encourages:
- Positive returns
- Low volatility (risk management)
- Consistent performance

### Why PPO?
- **Stable**: Clipped objective prevents destructive updates
- **Sample efficient**: Reuses collected experience
- **Proven**: Best empirical results for trading (vs DQN, A2C, SAC)
- **Robust**: Works well across different market regimes

## ⚠️ Important Considerations

### Overfitting Risk
- Markets are non-stationary—past patterns don't guarantee future performance
- Use walk-forward validation, not just train/test split
- Implement robust risk management (stop-losses, position limits)
- Regularly retrain with recent data

### Transaction Costs Matter
- Always include realistic transaction costs (0.1-0.3% for stocks)
- Model slippage for large orders
- Consider market impact

### Computational Requirements
- Training 100k steps: ~10-30 minutes on CPU
- Use GPU for faster training with large networks
- Start with shorter training runs for testing

### Deployment Caution
- **This is for educational/research purposes**
- Real trading requires additional considerations:
  - Real-time data feeds
  - Order execution infrastructure
  - Risk management systems
  - Regulatory compliance
- Paper trade extensively before live deployment

## 🛠️ Troubleshooting

### Common Issues

**1. Installation errors**
```bash
# If PyTorch fails
pip install torch --index-url https://download.pytorch.org/whl/cpu

# If gymnasium conflicts
pip install gymnasium==0.29.0 --force-reinstall
```

**2. Data download fails**
- Check internet connection
- Verify ticker symbol is valid
- Try different date ranges
- yfinance occasionally has rate limits—wait and retry

**3. Training is slow**
- Reduce `total_timesteps`
- Reduce `n_steps` (e.g., 1024)
- Use smaller batch size
- Consider using fewer features

**4. Agent performs poorly**
- Increase training timesteps (try 200k-500k)
- Adjust learning rate (try 1e-4 or 5e-4)
- Check for data quality issues
- Try different reward scaling factors

## 📖 Further Reading

**Research Papers:**
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- "FinRL: Deep Reinforcement Learning Framework to Automate Trading" (Liu et al., 2020)
- "Algorithmic Trading using Continuous Action Space Deep RL" (Ponomarev et al., 2024)

**Documentation:**
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [yfinance Docs](https://pypi.org/project/yfinance/)

## 🤝 Contributing

To extend this system:
1. Add new technical indicators in `data_collection.py`
2. Modify reward function in `trading_environment.py`
3. Experiment with different RL algorithms (SAC, TD3, A2C)
4. Implement hierarchical RL for multi-timeframe trading
5. Add portfolio optimization for multiple stocks

## 📄 License

This project is for educational purposes. Use at your own risk. Not financial advice.

## 🙏 Acknowledgments

Built using:
- Stable-Baselines3 (RL algorithms)
- Gymnasium (Environment interface)
- yfinance (Data collection)
- PyTorch (Neural networks)

---

**Happy Trading! 📈🤖**
