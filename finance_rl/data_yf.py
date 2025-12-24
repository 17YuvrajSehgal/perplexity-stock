# data_yf.py
# -----------
# This file is responsible for LOADING Yahoo Finance (yfinance) CSV files
# and converting them into clean NumPy arrays that are easy to use inside a Gym RL environment.

from __future__ import annotations  # allows using type hints like Prices inside the file (forward references)

from dataclasses import dataclass   # convenient way to define simple "data container" classes
from pathlib import Path            # safer path handling across Windows/Mac/Linux
from typing import Dict, Optional   # type hints

import numpy as np                 # fast numeric arrays (what Gym/RL models typically use)
import pandas as pd                # convenient CSV reading and cleaning


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

    Why do we need this?
    --------------------
    Yahoo Finance (and some export scripts) can produce weird column names like:
      "('Close',_'AAPL')" or "('Adj_Close',_'AAPL')"
    Instead of hard-coding exact column names, we search for the column that CONTAINS
    the keyword (like 'close', 'open', 'volume').

    Parameters
    ----------
    df : pandas.DataFrame
        Loaded CSV content.
    key : str
        Keyword we want to locate, e.g. "close", "open", "adj_close".

    Returns
    -------
    str
        The matching column name from df.
    """
    key_l = key.lower()

    cols = list(df.columns)

    # 1) Fast path: if there is an exact match (case-insensitive), use it.
    for c in cols:
        if str(c).strip().lower() == key_l:
            return c

    # 2) Otherwise, find all columns whose text contains the keyword.
    hits = [c for c in cols if key_l in str(c).lower()]
    if not hits:
        # If nothing matched, we cannot proceed safely.
        raise KeyError(f"Could not find a column containing '{key}'. Columns: {cols[:20]}...")

    # 3) Special case: for "close", prefer the non-adjusted close when possible.
    #    If the user wants adjusted close they should request "adj_close".
    if key_l == "close":
        for c in hits:
            if "adj" not in str(c).lower():
                return c

    # 4) If multiple possible matches, just return the first match.
    return hits[0]


def load_yfinance_csv(
    csv_path: str | Path,
    use_adj_close: bool = False,
    fill_volume: float = 0.0,
) -> Prices:
    """
    Load ONE yfinance CSV file and return a Prices object.

    What this function does (in plain terms):
    -----------------------------------------
    1) Read the CSV using pandas
    2) Sort the rows by Date (so data is in correct time order)
    3) Extract Open/High/Low/Close/Volume columns (even if names are weird)
    4) Convert them into clean float NumPy arrays
    5) Remove rows with missing OHLC values
    6) Return the result as a Prices dataclass

    Parameters
    ----------
    csv_path : str | Path
        Path to a CSV file.
    use_adj_close : bool
        If True, use Adjusted Close (accounts for splits/dividends).
        If False, use normal Close.
    fill_volume : float
        If volume is missing or NaN, replace it with this value.

    Returns
    -------
    Prices
        A clean OHLCV dataset stored as NumPy arrays.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # --- Try to find a Date column and sort by it (important for time-series consistency) ---
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break

    if date_col is not None:
        # Convert to datetime; invalid values become NaT (missing)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        # Sort by date so our arrays represent a proper timeline
        df = df.sort_values(date_col)

    # --- Identify the columns we need (even if their names are messy) ---
    open_c = _pick_col(df, "open")
    high_c = _pick_col(df, "high")
    low_c = _pick_col(df, "low")

    # Choose between Close vs Adjusted Close
    if use_adj_close:
        close_c = _pick_col(df, "adj_close")
    else:
        close_c = _pick_col(df, "close")

    # --- Volume might be missing for some assets/files ---
    vol_c = None
    try:
        vol_c = _pick_col(df, "volume")
    except KeyError:
        vol_c = None

    # Helper: convert a pandas column into a float32 NumPy array, turning bad values into NaN
    def as_float(col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)

    # Convert OHLC into numeric arrays
    o = as_float(open_c)
    h = as_float(high_c)
    l = as_float(low_c)
    c = as_float(close_c)

    # Convert volume; if it's missing, create an array filled with fill_volume
    if vol_c is None:
        v = np.full_like(c, fill_value=float(fill_volume), dtype=np.float32)
    else:
        v = pd.to_numeric(df[vol_c], errors="coerce").fillna(fill_volume).to_numpy(dtype=np.float32)

    # --- Clean up: remove any rows where Open/High/Low/Close is NaN or infinite ---
    mask = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    o, h, l, c, v = o[mask], h[mask], l[mask], c[mask], v[mask]

    # Need at least 2 rows for returns / previous close computations
    if c.shape[0] < 2:
        raise ValueError(f"Not enough valid rows in {csv_path}")

    # Package into Prices dataclass for easy use in the RL environment
    return Prices(open=o, high=h, low=l, close=c, volume=v)


def load_many_from_dir(
    data_dir: str | Path,
    pattern: str = "*.csv",
    use_adj_close: bool = False,
) -> Dict[str, Prices]:
    """
    Load ALL CSV files from a folder and return them as a dictionary.

    Example:
      yf_data/
        AAPL_1d.csv
        MSFT_1d.csv

    Returns:
      {
        "AAPL_1d": Prices(...),
        "MSFT_1d": Prices(...)
      }

    Parameters
    ----------
    data_dir : str | Path
        Folder containing CSV files.
    pattern : str
        File glob pattern (default "*.csv").
    use_adj_close : bool
        Whether to use Adjusted Close for every file.

    Returns
    -------
    Dict[str, Prices]
        Mapping from file-stem (filename without extension) to Prices object.
    """
    data_dir = Path(data_dir)

    out: Dict[str, Prices] = {}

    # Loop over every CSV file in the folder that matches the pattern
    for p in sorted(data_dir.glob(pattern)):
        key = p.stem  # filename without ".csv"
        out[key] = load_yfinance_csv(p, use_adj_close=use_adj_close)

    # If no files were loaded, report a clear error
    if not out:
        raise FileNotFoundError(f"No CSV files matched {pattern} in {data_dir}")

    return out
