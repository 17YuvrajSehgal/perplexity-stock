#!/usr/bin/env python3
"""
Helper script to download and prepare data files for cluster execution.
Run this on a machine with internet access before transferring to cluster.
"""

import argparse
import os
from data_collection import DataCollector


def prepare_data(ticker, start_date, end_date, output_dir='data', interval='1d'):
    """
    Download and prepare data file for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Directory to save processed data
        interval: Data interval (1d, 1h, etc.)
    """
    print(f"\n{'='*60}")
    print(f"Preparing data for {ticker}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize collector
    collector = DataCollector(
        ticker_symbol=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    # Download data
    print(f"Downloading data from {start_date} to {end_date}...")
    data = collector.download_data(interval=interval)
    
    if data is None or len(data) == 0:
        print(f"✗ Failed to download data for {ticker}")
        return False
    
    # Add technical indicators
    print("Adding technical indicators...")
    collector.add_technical_indicators()
    
    # Prepare data
    print("Preparing data...")
    df = collector.prepare_data(normalize=True)
    
    # Save to file
    filename = os.path.join(output_dir, f"{ticker.lower()}_processed.csv")
    collector.save_data(filename)
    
    print(f"✓ Data saved to: {filename}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare stock data files for cluster execution (offline mode)'
    )
    
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-01-01',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1h, etc.)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for data files')
    
    args = parser.parse_args()
    
    success = prepare_data(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        interval=args.interval
    )
    
    if success:
        print(f"\n{'='*60}")
        print("✓ Data preparation complete!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Transfer {args.output_dir}/{args.ticker.lower()}_processed.csv to cluster")
        print(f"2. Place in project directory or data/ subdirectory")
        print(f"3. Run training on cluster (it will automatically find the file)")
    else:
        print(f"\n✗ Data preparation failed for {args.ticker}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

