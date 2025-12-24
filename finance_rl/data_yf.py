# data_yf.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class Prices:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @property
    def length(self) -> int:
        return int(self.close.shape[0])


def _pick_col(df: pd.DataFrame, key: str) -> str:
    """
    Your CSV columns look like:
      "('Close',_'AAPL')" or "('Adj_Close',_'AAPL')"
    This finds a column containing the key (case-insensitive), preferring exact-ish matches.
    """
    key_l = key.lower()

    cols = list(df.columns)
    # fast path: exact match
    for c in cols:
        if str(c).strip().lower() == key_l:
            return c

    # fuzzy contains
    hits = [c for c in cols if key_l in str(c).lower()]
    if not hits:
        raise KeyError(f"Could not find a column containing '{key}'. Columns: {cols[:20]}...")

    # prefer non-Adj close when asking for Close, unless explicitly asking Adj_Close
    if key_l == "close":
        for c in hits:
            if "adj" not in str(c).lower():
                return c

    return hits[0]


def load_yfinance_csv(
    csv_path: str | Path,
    use_adj_close: bool = False,
    fill_volume: float = 0.0,
) -> Prices:
    """
    Loads ONE yfinance CSV produced by your downloader.
    Expected columns include Date + Open/High/Low/Close/Volume (names may be weird strings).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Handle common cases for date column
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    # Pick columns robustly
    open_c = _pick_col(df, "open")
    high_c = _pick_col(df, "high")
    low_c = _pick_col(df, "low")

    if use_adj_close:
        # yfinance often uses "Adj Close" (or your "Adj_Close")
        close_c = _pick_col(df, "adj_close")
    else:
        close_c = _pick_col(df, "close")

    # Volume might be missing for some assets
    vol_c = None
    try:
        vol_c = _pick_col(df, "volume")
    except KeyError:
        vol_c = None

    # Convert to numeric arrays
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

    # Drop rows where OHLC is NaN
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
    Loads all CSVs in a folder: returns dict {instrument_name: Prices}
    """
    data_dir = Path(data_dir)
    out: Dict[str, Prices] = {}
    for p in sorted(data_dir.glob(pattern)):
        key = p.stem  # filename without extension
        out[key] = load_yfinance_csv(p, use_adj_close=use_adj_close)
    if not out:
        raise FileNotFoundError(f"No CSV files matched {pattern} in {data_dir}")
    return out
