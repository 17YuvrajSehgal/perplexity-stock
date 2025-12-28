#!/usr/bin/env python3
"""
Download historical OHLCV data from Yahoo Finance using yfinance and save to files.

Examples:
  python fetch_yf.py --tickers AAPL MSFT TSLA --start 2023-01-01 --end 2023-12-31 --interval 1d --out data
  python fetch_yf.py --tickers SPY QQQ --period 5y --interval 1wk --out data --format parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit(
        "Missing dependency: yfinance\n"
        "Install with: pip install yfinance pandas\n"
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and download data from yfinance (Yahoo Finance).")
    p.add_argument("--tickers", nargs="+", required=True, help="One or more tickers, e.g., AAPL MSFT TSLA")
    p.add_argument("--out", default="yf_data", help="Output directory (default: yf_data)")

    # Either provide period OR start/end
    p.add_argument("--period", default=None, help="Period, e.g., 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,max")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (used if period not set)")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (used if period not set)")

    p.add_argument("--interval", default="1d", help="Interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo")
    p.add_argument("--adjust", action="store_true", help="Auto-adjust prices (dividends/splits)")
    p.add_argument("--actions", action="store_true", help="Include dividends/splits columns (if available)")
    p.add_argument("--prepost", action="store_true", help="Include pre/post market data where available")

    p.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Output file format (default: csv)")
    p.add_argument("--group", choices=["separate", "combined"], default="separate",
                   help="Save each ticker separately or combined file (default: separate)")

    return p.parse_args()


def validate_dates(start: Optional[str], end: Optional[str]) -> None:
    for name, val in [("start", start), ("end", end)]:
        if val is None:
            continue
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid {name} date '{val}'. Expected YYYY-MM-DD.")


def download_one(
    ticker: str,
    period: Optional[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    auto_adjust: bool,
    actions: bool,
    prepost: bool,
) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=actions,
        prepost=prepost,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance returns index as DatetimeIndex; make it a normal column for file output
    df = df.reset_index()

    # Normalize column names
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    df.insert(0, "Ticker", ticker)

    return df


def save_df(df: pd.DataFrame, path: str, fmt: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        # parquet requires pyarrow or fastparquet
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            raise SystemExit(
                "Failed to write parquet. Install one of:\n"
                "  pip install pyarrow\n"
                "  pip install fastparquet\n"
                f"Original error: {e}"
            )


def main() -> int:
    args = parse_args()
    validate_dates(args.start, args.end)

    if args.period is None and (args.start is None or args.end is None):
        print("Error: Provide either --period OR both --start and --end.", file=sys.stderr)
        return 2

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[pd.DataFrame] = []

    for t in args.tickers:
        print(f"Downloading {t} ...")
        df = download_one(
            ticker=t,
            period=args.period,
            start=args.start,
            end=args.end,
            interval=args.interval,
            auto_adjust=args.adjust,
            actions=args.actions,
            prepost=args.prepost,
        )

        if df.empty:
            print(f"  -> No data returned for {t}", file=sys.stderr)
            continue

        if args.group == "separate":
            safe_t = t.replace("/", "_").replace(" ", "_")
            fname = f"{safe_t}_{args.interval}"
            if args.period:
                fname += f"_{args.period}"
            else:
                fname += f"_{args.start}_to_{args.end}"
            fpath = os.path.join(out_dir, f"{fname}.{args.format}")
            save_df(df, fpath, args.format)
            print(f"  -> saved: {fpath}")

        all_rows.append(df)

    if args.group == "combined" and all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        fname = f"combined_{args.interval}"
        if args.period:
            fname += f"_{args.period}"
        else:
            fname += f"_{args.start}_to_{args.end}"
        fpath = os.path.join(out_dir, f"{fname}.{args.format}")
        save_df(combined, fpath, args.format)
        print(f"Saved combined file: {fpath}")

    if not all_rows:
        print("No data downloaded (all tickers returned empty).", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#python fetch_yf.py --tickers AAPL MSFT --start 2020-01-01 --end 2025-12-23 --interval 1d --out data --group combined

#python fetch_yf.py --tickers AAPL --start 2020-01-01 --end 2025-12-23 --interval 1d


