"""
Data loading and preprocessing for stock trading environments.

This module provides utilities to load and process stock price data
from CSV files (e.g., from Yahoo Finance).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Prices:
    """
    Simple container for price history.

    Each field is a NumPy array where index i means "time step i" (e.g., day i).
    Example:
      close[i] = closing price on day i
    """
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @property
    def length(self) -> int:
        """
        Number of rows (time steps) in this price history.
        We assume all arrays have the same length.
        """
        return int(self.close.shape[0])


def _pick_col(df: pd.DataFrame, key: str) -> str:
    """
    Find the "best matching" column name in a CSV DataFrame.

    Yahoo Finance (and some export scripts) can produce weird column names like:
      "('Close',_'AAPL')" or "('Adj_Close',_'AAPL')"

    Instead of hard-coding exact column names, we search for the column that CONTAINS
    the keyword (like 'close', 'open', 'volume').
    """
    key_l = key.lower()
    cols = list(df.columns)

    # 1) Exact match (case-insensitive)
    for c in cols:
        if str(c).strip().lower() == key_l:
            return c

    # 2) Contains match
    hits = [c for c in cols if key_l in str(c).lower()]
    if not hits:
        raise KeyError(f"Could not find a column containing '{key}'. Columns: {cols[:20]}...")

    # 3) Special case: for "close", prefer non-adjusted close when possible.
    if key_l == "close":
        for c in hits:
            if "adj" not in str(c).lower():
                return c

    # 4) Otherwise return first match
    return hits[0]


def load_yfinance_csv(
    csv_path: str | Path,
    use_adj_close: bool = False,
    fill_volume: float = 0.0,
) -> Prices:
    """
    Load ONE yfinance CSV file and return a Prices object.

    Steps:
      1) Read the CSV using pandas
      2) Sort rows by Date (so data is in correct time order)
      3) Extract Open/High/Low/Close/Volume columns (even if names are weird)
      4) Convert to clean float NumPy arrays
      5) Remove rows with missing OHLC values

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file
    use_adj_close : bool, default=False
        Whether to use adjusted close prices
    fill_volume : float, default=0.0
        Value to fill missing volume data with

    Returns
    -------
    Prices
        A Prices object containing OHLCV data as NumPy arrays

    Raises
    ------
    ValueError
        If the CSV doesn't have enough valid rows
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # --- Find a Date column and sort by it (important for time-series consistency) ---
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    # --- Identify columns (even if messy names) ---
    open_c = _pick_col(df, "open")
    high_c = _pick_col(df, "high")
    low_c = _pick_col(df, "low")

    if use_adj_close:
        close_c = _pick_col(df, "adj_close")
    else:
        close_c = _pick_col(df, "close")

    # Volume might be missing
    vol_c = None
    try:
        vol_c = _pick_col(df, "volume")
    except KeyError:
        vol_c = None

    # Helper: pandas column -> float32 array
    def as_float(col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)

    o = as_float(open_c)
    h = as_float(high_c)
    l = as_float(low_c)
    c = as_float(close_c)

    if vol_c is None:
        v = np.full_like(c, fill_value=float(fill_volume), dtype=np.float32)
    else:
        v = pd.to_numeric(df[vol_c], errors="coerce").fillna(fill_volume).to_numpy(dtype=np.float32)

    # --- Clean up: remove rows where OHLC is NaN/inf ---
    mask = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    o, h, l, c, v = o[mask], h[mask], l[mask], c[mask], v[mask]

    if c.shape[0] < 2:
        raise ValueError(f"Not enough valid rows in {csv_path}")

    return Prices(open=o, high=h, low=l, close=c, volume=v)


def load_many_from_dir(
    data_dir: str | Path,
    pattern: str = "*.csv",
    use_adj_close: bool = False,
) -> Dict[str, Prices]:
    """
    Load ALL CSV files from a folder and return them as a dict.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing CSV files
    pattern : str, default="*.csv"
        Glob pattern to match files
    use_adj_close : bool, default=False
        Whether to use adjusted close prices

    Returns
    -------
    Dict[str, Prices]
        Dictionary mapping instrument names to Prices objects
        Example: { "AAPL_1d": Prices(...), "MSFT_1d": Prices(...), ... }

    Raises
    ------
    FileNotFoundError
        If no CSV files match the pattern
    """
    data_dir = Path(data_dir)
    out: Dict[str, Prices] = {}

    for p in sorted(data_dir.glob(pattern)):
        key = p.stem
        out[key] = load_yfinance_csv(p, use_adj_close=use_adj_close)

    if not out:
        raise FileNotFoundError(f"No CSV files matched {pattern} in {data_dir}")

    return out


# -----------------------------------------------------------------------------
# Chronological train/validation splitting (to avoid leakage)
# -----------------------------------------------------------------------------

def split_prices_by_ratio(
    p: Prices,
    train_ratio: float = 0.8,
    min_train: int = 200,
    min_val: int = 200,
) -> tuple[Prices, Prices]:
    """
    Chronological split (no shuffling):
      train = [0 : split_idx)
      val   = [split_idx : end)

    Ensures both sides have enough samples.

    Parameters
    ----------
    p : Prices
        Price data to split
    train_ratio : float, default=0.8
        Fraction of data to use for training
    min_train : int, default=200
        Minimum number of samples required for training set
    min_val : int, default=200
        Minimum number of samples required for validation set

    Returns
    -------
    tuple[Prices, Prices]
        (train_prices, val_prices)

    Raises
    ------
    ValueError
        If the split would result in invalid sizes
    """
    assert 0.1 < train_ratio < 0.95, "train_ratio should be reasonable"
    n = p.length
    split_idx = int(n * train_ratio)

    # enforce minimum sizes
    split_idx = max(split_idx, min_train)
    split_idx = min(split_idx, n - min_val)

    if split_idx <= 1 or (n - split_idx) <= 1:
        raise ValueError(f"Split invalid: n={n}, split_idx={split_idx}")

    train = Prices(
        open=p.open[:split_idx],
        high=p.high[:split_idx],
        low=p.low[:split_idx],
        close=p.close[:split_idx],
        volume=p.volume[:split_idx],
    )
    val = Prices(
        open=p.open[split_idx:],
        high=p.high[split_idx:],
        low=p.low[split_idx:],
        close=p.close[split_idx:],
        volume=p.volume[split_idx:],
    )
    return train, val


def split_many_by_ratio(
    prices: Dict[str, Prices],
    train_ratio: float = 0.8,
    min_train: int = 200,
    min_val: int = 200,
) -> tuple[Dict[str, Prices], Dict[str, Prices]]:
    """
    Split every instrument chronologically.
    Instruments that are too short (cannot satisfy min_train/min_val) are skipped.

    Parameters
    ----------
    prices : Dict[str, Prices]
        Dictionary of price data to split
    train_ratio : float, default=0.8
        Fraction of data to use for training
    min_train : int, default=200
        Minimum number of samples required for training set
    min_val : int, default=200
        Minimum number of samples required for validation set

    Returns
    -------
    tuple[Dict[str, Prices], Dict[str, Prices]]
        (train_prices_dict, val_prices_dict)

    Raises
    ------
    ValueError
        If no instruments remain after splitting
    """
    train_out: Dict[str, Prices] = {}
    val_out: Dict[str, Prices] = {}

    for k, p in prices.items():
        try:
            tr, va = split_prices_by_ratio(
                p,
                train_ratio=train_ratio,
                min_train=min_train,
                min_val=min_val,
            )
            train_out[k] = tr
            val_out[k] = va
        except Exception:
            # too short / bad split -> skip instrument
            continue

    if not train_out or not val_out:
        raise ValueError(
            "After splitting, no instruments were usable. "
            "Check min_train/min_val or your CSV lengths."
        )

    return train_out, val_out

