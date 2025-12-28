# Stock Trading Environment

A production-ready Python package for stock trading reinforcement learning environments.

## Installation

This package requires:
- Python 3.8+
- gymnasium
- numpy
- pandas

Install dependencies:
```bash
pip install gymnasium numpy pandas
```

## Quick Start

### Loading Data

```python
from stock_trading_env import load_many_from_dir, split_many_by_ratio

# Load all CSV files from a directory
prices = load_many_from_dir("yf_data")

# Split into train/validation sets (chronological, no leakage)
train_prices, val_prices = split_many_by_ratio(
    prices,
    train_ratio=0.8,
    min_train=200,
    min_val=200
)
```

### Creating an Environment

```python
from stock_trading_env import StocksEnv

# Create environment from price data
env = StocksEnv(
    prices=train_prices,
    bars_count=10,
    volumes=True,
    extra_features=True,
    reward_mode="close_pnl",
    commission=0.001,
    hold_penalty_per_step=0.00002,
    max_hold_steps=250
)

# Or load directly from directory
env = StocksEnv.from_dir("yf_data", bars_count=10)
```

### Using the Environment

```python
from stock_trading_env import Actions

# Reset environment
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# Step through environment
action = Actions.Buy.value  # or 1
obs, reward, terminated, truncated, info = env.step(action)

# Actions:
# - Actions.Skip (0): Hold current position
# - Actions.Buy (1): Open a long position
# - Actions.Close (2): Close current position
```

## Package Structure

```
stock_trading_env/
├── __init__.py          # Main package exports
├── data/                # Data loading and preprocessing
│   ├── __init__.py
│   └── loader.py        # CSV loading, splitting functions
└── environment/         # Trading environment
    ├── __init__.py
    └── env.py           # StocksEnv, State, Actions classes
```

## API Reference

### Data Loading

- `load_yfinance_csv(csv_path, use_adj_close=False, fill_volume=0.0)` - Load a single CSV file
- `load_many_from_dir(data_dir, pattern="*.csv", use_adj_close=False)` - Load all CSVs from directory
- `split_prices_by_ratio(prices, train_ratio=0.8, min_train=200, min_val=200)` - Split single Prices object
- `split_many_by_ratio(prices_dict, train_ratio=0.8, min_train=200, min_val=200)` - Split dictionary of Prices

### Environment

- `StocksEnv` - Main Gymnasium-compatible environment
- `Actions` - Action enum (Skip, Buy, Close)
- `State` - Flat state encoder (for MLP models)
- `State1D` - 1D state encoder (for CNN models)

### Constants

- `DEFAULT_BARS_COUNT = 10`
- `DEFAULT_COMMISSION_PERC = 0.001`
- `DEFAULT_HOLD_PENALTY_PERC = 0.00002`
- `DEFAULT_MAX_HOLD_STEPS = 250`

## Features

- **Gymnasium-compatible**: Works with any RL library that supports Gymnasium
- **No data leakage**: Proper chronological train/validation splits
- **Flexible state encoding**: Supports both flat (MLP) and 1D (CNN) observations
- **Production-ready**: Clean package structure, type hints, documentation
- **Notebook-friendly**: Easy to import and use in Jupyter notebooks

## Example: Training in Jupyter Notebook

```python
import gymnasium as gym
from stock_trading_env import StocksEnv, load_many_from_dir, split_many_by_ratio

# Load and split data
prices = load_many_from_dir("yf_data")
train_prices, val_prices = split_many_by_ratio(prices, train_ratio=0.8)

# Create environments
train_env = StocksEnv(train_prices, bars_count=10, volumes=True)
val_env = StocksEnv(val_prices, bars_count=10, volumes=True)

# Wrap with TimeLimit if needed
train_env = gym.wrappers.TimeLimit(train_env, max_episode_steps=1000)

# Now ready for RL training!
```

