
# data_collection.py
# Module for downloading and preprocessing stock data

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    """
    Handles data collection from Yahoo Finance and preprocessing
    """

    def __init__(self, ticker_symbol, start_date=None, end_date=None):
        """
        Initialize data collector

        Args:
            ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
            start_date (str): Start date 'YYYY-MM-DD' 
            end_date (str): End date 'YYYY-MM-DD'
        """
        self.ticker = ticker_symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None

    def download_data(self, interval='1d'):
        """
        Download OHLCV data from Yahoo Finance

        Args:
            interval (str): Data interval ('1d', '1h', '1wk', etc.)

        Returns:
            pd.DataFrame: OHLCV data
        """
        print(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}...")

        try:
            self.data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                interval=interval,
                auto_adjust=True,  # Adjust for splits and dividends
                progress=False
            )

            # Clean column names if multi-index
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = [col[0] for col in self.data.columns]

            print(f"Downloaded {len(self.data)} rows of data")
            return self.data

        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

    def add_technical_indicators(self):
        """
        Add technical indicators to the dataset
        Uses pandas for calculations (alternative to TA-Lib for easier setup)

        Returns:
            pd.DataFrame: Data with technical indicators
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data available. Run download_data() first.")

        df = self.data.copy()

        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)

        # Drop NaN values
        df = df.dropna()

        self.data = df
        print(f"Added technical indicators. Shape: {df.shape}")
        return df

    def prepare_data(self, normalize=True):
        """
        Prepare data for RL environment

        Args:
            normalize (bool): Whether to normalize features

        Returns:
            pd.DataFrame: Prepared data
        """
        if self.data is None:
            self.download_data()
            self.add_technical_indicators()

        df = self.data.copy()

        if normalize:
            # Normalize features (except returns which are already normalized)
            feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

            # Use rolling normalization to avoid look-ahead bias
            for col in feature_cols:
                if col not in ['Returns', 'Log_Returns']:
                    rolling_mean = df[col].rolling(window=50, min_periods=1).mean()
                    rolling_std = df[col].rolling(window=50, min_periods=1).std()
                    df[f'{col}_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

        return df

    def save_data(self, filename='stock_data.csv'):
        """
        Save processed data to CSV

        Args:
            filename (str): Output filename
        """
        if self.data is not None:
            self.data.to_csv(filename)
            print(f"Data saved to {filename}")
        else:
            print("No data to save")


# Example usage
if __name__ == "__main__":
    # Initialize collector
    collector = DataCollector('AAPL', start_date='2020-01-01', end_date='2024-01-01')

    # Download and process data
    collector.download_data(interval='1d')
    collector.add_technical_indicators()
    data = collector.prepare_data(normalize=True)

    # Save to file
    collector.save_data('aapl_processed.csv')

    # Display summary
    print("\nData Summary:")
    print(data.describe())
    print("\nColumns:", data.columns.tolist())
