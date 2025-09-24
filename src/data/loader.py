"""
Data loading and preprocessing module.
"""
import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime, timedelta

from ..core.config import CONFIG
from ..core.logger import logger
from ..core.exceptions import DataLoadError, InsufficientDataError


class StockDataLoader:
    """Handles stock data loading and basic preprocessing."""
    
    def __init__(self):
        self.data = None
        self.symbol = None
    
    def load_data(self, 
                  symbol: str = None, 
                  period: str = None, 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            period: Time period ('1y', '2y', '3y', etc.)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with stock data
            
        Raises:
            DataLoadError: If data loading fails
            InsufficientDataError: If insufficient data is loaded
        """
        symbol = symbol or CONFIG.data.symbol
        period = period or CONFIG.data.period
        
        logger.info(f"Loading {symbol} data for period: {period}", "📊")
        
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period=period)
            
            if data.empty:
                raise DataLoadError(f"No data found for symbol: {symbol}")
            
            if len(data) < 100:  # Minimum required data points
                raise InsufficientDataError(
                    f"Insufficient data: {len(data)} rows (minimum: 100)"
                )
            
            # Clean data
            data = self._clean_data(data)
            
            self.data = data
            self.symbol = symbol
            
            logger.success(f"Loaded {len(data)} days of data")
            logger.info(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            return data
            
        except Exception as e:
            raise DataLoadError(f"Failed to load data for {symbol}: {str(e)}")
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate stock data.
        
        Args:
            data: Raw stock data
            
        Returns:
            Cleaned stock data
        """
        logger.info("Cleaning stock data", "🧹")
        
        # Remove rows with missing values
        initial_length = len(data)
        data = data.dropna()
        
        if len(data) < initial_length:
            logger.warning(f"Removed {initial_length - len(data)} rows with missing values")
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise DataLoadError(f"Missing required columns: {missing_columns}")
        
        # Remove rows where High < Low (data errors)
        invalid_rows = data['High'] < data['Low']
        if invalid_rows.any():
            logger.warning(f"Removing {invalid_rows.sum()} rows with High < Low")
            data = data[~invalid_rows]
        
        # Remove rows with zero or negative prices
        invalid_prices = (data['Close'] <= 0) | (data['Open'] <= 0)
        if invalid_prices.any():
            logger.warning(f"Removing {invalid_prices.sum()} rows with invalid prices")
            data = data[~invalid_prices]
        
        # Sort by date
        data = data.sort_index()
        
        logger.success(f"Data cleaning complete: {len(data)} valid rows")
        
        return data
    
    def get_basic_stats(self) -> dict:
        """Get basic statistics about loaded data."""
        if self.data is None:
            return {}
        
        stats = {
            'symbol': self.symbol,
            'total_days': len(self.data),
            'start_date': self.data.index[0].date(),
            'end_date': self.data.index[-1].date(),
            'price_range': {
                'min': self.data['Close'].min(),
                'max': self.data['Close'].max(),
                'current': self.data['Close'].iloc[-1]
            },
            'volume_stats': {
                'mean': self.data['Volume'].mean(),
                'max': self.data['Volume'].max()
            }
        }
        
        return stats


class TargetGenerator:
    """Generates prediction targets from stock data."""
    
    def __init__(self, threshold: float = None):
        self.threshold = threshold or CONFIG.data.target_threshold
    
    def create_binary_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Create binary classification target.
        
        Args:
            data: Stock data with 'Close' column
            
        Returns:
            Binary target (1 for significant upward movement, 0 otherwise)
        """
        logger.info(f"Creating binary target with {self.threshold:.1%} threshold", "🎯")
        
        # Calculate next day returns
        next_day_returns = data['Close'].pct_change().shift(-1)
        
        # Create binary target
        binary_target = (next_day_returns > self.threshold).astype(int)
        
        # Remove last row (no target available)
        binary_target = binary_target.iloc[:-1]
        
        # Calculate class distribution
        positive_ratio = binary_target.mean()
        negative_count = (binary_target == 0).sum()
        positive_count = (binary_target == 1).sum()
        
        logger.info(f"Target distribution:")
        logger.info(f"  Down/Sideways (0): {negative_count} ({1-positive_ratio:.1%})")
        logger.info(f"  Up {self.threshold:.1%}+ (1): {positive_count} ({positive_ratio:.1%})")
        
        return binary_target
    
    def create_multiclass_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Create multi-class classification target.
        
        Args:
            data: Stock data with 'Close' column
            
        Returns:
            Multi-class target (0: Strong Down, 1: Weak Down, 2: Sideways, 
                               3: Weak Up, 4: Strong Up)
        """
        logger.info("Creating multi-class target (5 classes)", "🎯")
        
        next_day_returns = data['Close'].pct_change().shift(-1)
        labels = pd.Series(2, index=next_day_returns.index)  # Default: Sideways
        
        # Define thresholds
        labels[next_day_returns < -0.02] = 0    # Strong Down
        labels[(next_day_returns >= -0.02) & (next_day_returns < 0.0)] = 1    # Weak Down
        labels[(next_day_returns >= -0.005) & (next_day_returns <= 0.005)] = 2  # Sideways
        labels[(next_day_returns > 0.0) & (next_day_returns <= 0.02)] = 3     # Weak Up
        labels[next_day_returns > 0.02] = 4     # Strong Up
        
        # Remove last row
        labels = labels.iloc[:-1]
        
        # Log distribution
        class_names = ['Strong_Down', 'Weak_Down', 'Sideways', 'Weak_Up', 'Strong_Up']
        for i, name in enumerate(class_names):
            count = (labels == i).sum()
            percentage = count / len(labels) * 100
            logger.info(f"  {i}. {name:12s}: {count:4d} ({percentage:.1f}%)")
        
        return labels