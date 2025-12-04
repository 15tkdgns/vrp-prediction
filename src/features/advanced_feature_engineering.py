#!/usr/bin/env python3
"""
고급 특성 엔지니어링 구현
변동성 예측 성능 개선을 위한 정교한 특성 생성
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import os

warnings.filterwarnings('ignore')

def load_enhanced_spy_data():
        
    def create_advanced_technical_indicators(self, data):
        """Create advanced technical indicators beyond basic RSI/MACD"""
        logger.info("Creating advanced technical indicators...")
        
        df = data.copy()
        
        # Price arrays for TA-Lib (ensure float64 type)
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        close = df['Close'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(close, timeperiod=20)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Stochastic Oscillators
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close)
        
        # Williams %R
        df['WILLIAMS_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Average True Range (Volatility)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['ATR_ratio'] = df['ATR'] / close
        
        # Commodity Channel Index
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        
        # Money Flow Index
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # Average Directional Index (Trend Strength)
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(high, low)
        df['SAR_signal'] = np.where(close > df['SAR'], 1, -1)
        
        logger.info(f"Added {len([col for col in df.columns if col not in data.columns])} advanced technical indicators")
        return df
    
    def create_volatility_features(self, data):
        """Create sophisticated volatility and momentum features"""
        logger.info("Creating volatility and momentum features...")
        
        df = data.copy()
        
        # Multiple timeframe volatilities
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['Returns'].rolling(window=period).std()
        
        # Volatility ratios (create after all volatilities are calculated)
        for period in [5, 10, 50]:
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df['volatility_20']
        
        # Volatility breakout signals
        df['vol_breakout_5'] = (df['volatility_5'] > df['volatility_20'] * 1.5).astype(int)
        df['vol_breakout_10'] = (df['volatility_10'] > df['volatility_20'] * 1.2).astype(int)
        
        # Price momentum features
        for period in [1, 3, 5, 10]:
            df[f'price_change_{period}'] = df['Close'].pct_change(periods=period)
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Volume-based features
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['volume_breakout'] = (df['volume_ratio'] > 2.0).astype(int)
        
        # Price-Volume interaction
        df['price_volume_trend'] = df['Returns'] * np.log1p(df['volume_ratio'])
        
        # Volatility clustering (GARCH-like features)
        df['vol_regime'] = (df['Volatility'] > df['Volatility'].rolling(50).quantile(0.75)).astype(int)
        
        logger.info(f"Added volatility and momentum features")
        return df
    
    def create_lag_features(self, data, max_lag=5):
        """Create time-series lag features"""
        logger.info(f"Creating lag features up to {max_lag} periods...")
        
        df = data.copy()
        
        # Key indicators to create lags for
        key_features = ['Returns', 'RSI', 'MACD', 'Volatility', 'volume_ratio', 'BB_position']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in range(1, max_lag + 1):
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Rolling averages of key features
        for feature in ['Returns', 'RSI', 'volume_ratio']:
            if feature in df.columns:
                df[f'{feature}_ma_3'] = df[feature].rolling(window=3).mean()
                df[f'{feature}_ma_7'] = df[feature].rolling(window=7).mean()
        
        logger.info(f"Added lag and rolling features")
        return df
    
    def create_market_structure_features(self, data):
        """Create market structure and regime features"""
        logger.info("Creating market structure features...")
        
        df = data.copy()
        
        # Market regime indicators
        df['trend_strength'] = np.abs(df['Close'].rolling(20).corr(pd.Series(range(20))))
        df['ma_20_slope'] = df['MA_20'].diff() / df['MA_20'].shift(1)
        df['ma_50_slope'] = df['MA_50'].diff() / df['MA_50'].shift(1)
        
        # Price action patterns
        df['doji'] = (np.abs(df['Open'] - df['Close']) / (df['High'] - df['Low']) < 0.1).astype(int)
        df['hammer'] = ((df['Close'] > df['Open']) & 
                       ((df['High'] - df['Close']) / (df['Close'] - df['Open']) < 0.5) & 
                       ((df['Open'] - df['Low']) / (df['Close'] - df['Open']) > 2)).astype(int)
        
        # Gap detection
        df['gap_up'] = (df['Open'] > df['High'].shift(1) * 1.005).astype(int)
        df['gap_down'] = (df['Open'] < df['Low'].shift(1) * 0.995).astype(int)
        
        # Multi-timeframe alignment
        df['ma_alignment'] = ((df['Close'] > df['MA_20']) & 
                             (df['MA_20'] > df['MA_50'])).astype(int)
        
        logger.info("Added market structure features")
        return df
    
    def create_statistical_features(self, data):
        """Create statistical and mathematical features"""
        logger.info("Creating statistical features...")
        
        df = data.copy()
        
        # Z-scores of key features
        for feature in ['Returns', 'Volume', 'RSI']:
            if feature in df.columns:
                rolling_mean = df[feature].rolling(window=50).mean()
                rolling_std = df[feature].rolling(window=50).std()
                df[f'{feature}_zscore'] = (df[feature] - rolling_mean) / rolling_std
        
        # Percentile ranks
        for feature in ['Volume', 'Volatility', 'ATR']:
            if feature in df.columns:
                df[f'{feature}_percentile'] = df[feature].rolling(window=252).rank(pct=True)
        
        # Autocorrelation features
        df['returns_autocorr_1'] = df['Returns'].rolling(window=50).apply(
            lambda x: x.autocorr(lag=1) if len(x) >= 10 else np.nan)
        
        logger.info("Added statistical features")
        return df
    
    def engineer_all_features(self, data):
        """Apply all feature engineering techniques"""
        logger.info("Starting comprehensive feature engineering...")
        
        # Start with original data
        df = data.copy()
        
        # Apply all feature engineering steps
        df = self.create_advanced_technical_indicators(df)
        df = self.create_volatility_features(df)
        df = self.create_lag_features(df)
        df = self.create_market_structure_features(df)
        df = self.create_statistical_features(df)
        
        # Fill NaN values created by rolling/lag operations
        # Forward fill first, then backward fill, then fill with 0
        df = df.ffill().bfill().fillna(0)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        logger.info(f"Original features: {len(data.columns)}")
        logger.info(f"New features added: {len(df.columns) - len(data.columns)}")
        
        return df
    
    def select_features_for_modeling(self, data):
        """Select the most relevant features for modeling"""
        # Exclude non-predictive columns
        exclude_cols = ['Date', 'Dividends', 'Stock Splits']
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        logger.info(f"Selected {len(feature_cols)} features for modeling")
        return feature_cols

def main():
    """Test the feature engineering pipeline"""
    logger.info("Testing advanced feature engineering pipeline...")
    
    # Load data
    data = pd.read_csv('/root/workspace/data/training/sp500_2020_2024.csv')
    logger.info(f"Loaded data shape: {data.shape}")
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Apply feature engineering
    enhanced_data = engineer.engineer_all_features(data)
    
    # Get feature list
    feature_cols = engineer.select_features_for_modeling(enhanced_data)
    
    # Save enhanced dataset
    output_path = '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'
    enhanced_data.to_csv(output_path, index=False)
    logger.info(f"Enhanced dataset saved to: {output_path}")
    
    # Print feature summary
    logger.info("\nFeature Categories Summary:")
    
    technical_indicators = [col for col in feature_cols if any(
        indicator in col.upper() for indicator in ['BB_', 'STOCH', 'WILLIAMS', 'ATR', 'CCI', 'MFI', 'ADX', 'SAR']
    )]
    volatility_features = [col for col in feature_cols if 'volatility' in col or 'momentum' in col or 'vol_' in col]
    lag_features = [col for col in feature_cols if '_lag_' in col or '_ma_' in col]
    structure_features = [col for col in feature_cols if any(
        pattern in col for pattern in ['trend_', 'slope', 'doji', 'hammer', 'gap_', 'alignment']
    )]
    statistical_features = [col for col in feature_cols if any(
        pattern in col for pattern in ['zscore', 'percentile', 'autocorr']
    )]
    
    logger.info(f"- Technical Indicators: {len(technical_indicators)}")
    logger.info(f"- Volatility Features: {len(volatility_features)}")
    logger.info(f"- Lag Features: {len(lag_features)}")
    logger.info(f"- Market Structure: {len(structure_features)}")
    logger.info(f"- Statistical Features: {len(statistical_features)}")
    
    return enhanced_data, feature_cols

if __name__ == "__main__":
    enhanced_data, features = main()