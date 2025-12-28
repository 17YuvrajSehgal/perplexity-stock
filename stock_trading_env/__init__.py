"""
Stock Trading Environment Package

A production-ready package for stock trading reinforcement learning environments.
"""

from stock_trading_env.data import (
    Prices,
    load_yfinance_csv,
    load_many_from_dir,
    split_prices_by_ratio,
    split_many_by_ratio,
)

from stock_trading_env.environment import (
    Actions,
    StocksEnv,
    State,
    State1D,
    DEFAULT_BARS_COUNT,
    DEFAULT_COMMISSION_PERC,
    DEFAULT_HOLD_PENALTY_PERC,
    DEFAULT_MAX_HOLD_STEPS,
)

__version__ = "0.1.0"
__all__ = [
    # Data classes and functions
    "Prices",
    "load_yfinance_csv",
    "load_many_from_dir",
    "split_prices_by_ratio",
    "split_many_by_ratio",
    # Environment classes
    "Actions",
    "StocksEnv",
    "State",
    "State1D",
    # Constants
    "DEFAULT_BARS_COUNT",
    "DEFAULT_COMMISSION_PERC",
    "DEFAULT_HOLD_PENALTY_PERC",
    "DEFAULT_MAX_HOLD_STEPS",
]

