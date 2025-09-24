"""
Feature engineering module with modular, extensible design.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from ..core.config import CONFIG
from ..core.logger import logger
from ..core.exceptions import FeatureEngineeringError


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, name: str):
        self.name = name
        self.feature_names: List[str] = []
    
    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract features from stock data."""
        pass
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names


class TechnicalIndicatorExtractor(BaseFeatureExtractor):
    """Extract technical indicator features."""
    
    def __init__(self):
        super().__init__("TechnicalIndicators")
        self.config = CONFIG.features.technical_indicators
    
    def extract_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract technical indicator features."""
        logger.info(f"Extracting {self.name} features", "🔧")
        features = {}
        
        # RSI indicators
        features.update(self._calculate_rsi(data))
        
        # Moving averages
        features.update(self._calculate_moving_averages(data))
        
        # Bollinger Bands
        features.update(self._calculate_bollinger_bands(data))
        
        # MACD
        features.update(self._calculate_macd(data))
        
        # Stochastic
        features.update(self._calculate_stochastic(data))
        
        self.feature_names = list(features.keys())
        logger.success(f"Generated {len(features)} technical indicator features")
        
        return features
    
    def _calculate_rsi(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate RSI for multiple periods."""
        features = {}
        
        for period in self.config['rsi_periods']:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            features[f'rsi_{period}'] = rsi
            features[f'rsi_momentum_{period}'] = rsi.diff()
            features[f'rsi_oversold_{period}'] = (rsi < 30).astype(int)
            features[f'rsi_overbought_{period}'] = (rsi > 70).astype(int)
        
        return features
    
    def _calculate_moving_averages(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate moving averages and crossovers."""
        features = {}
        ma_data = {}
        
        # Calculate moving averages
        for period in self.config['ma_periods']:
            ma_data[f'sma_{period}'] = data['Close'].rolling(period).mean()
            ma_data[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
            
            features[f'sma_{period}'] = ma_data[f'sma_{period}']
            features[f'ema_{period}'] = ma_data[f'ema_{period}']
            
            if period <= 50:  # Only for shorter periods
                features[f'price_sma_ratio_{period}'] = data['Close'] / ma_data[f'sma_{period}']
                features[f'price_ema_ratio_{period}'] = data['Close'] / ma_data[f'ema_{period}']
        
        # Calculate crossovers
        periods = self.config['ma_periods']
        for i, short in enumerate(periods[:-1]):
            for long in periods[i+1:]:
                if f'sma_{short}' in ma_data and f'sma_{long}' in ma_data:
                    # SMA crossovers
                    cross = ma_data[f'sma_{short}'] > ma_data[f'sma_{long}']
                    features[f'sma_cross_{short}_{long}'] = cross.astype(int)
                    features[f'sma_cross_change_{short}_{long}'] = cross.astype(int).diff()
                    features[f'sma_distance_{short}_{long}'] = (
                        ma_data[f'sma_{short}'] - ma_data[f'sma_{long}']
                    ) / ma_data[f'sma_{long}']
                    
                    # EMA crossovers
                    ema_cross = ma_data[f'ema_{short}'] > ma_data[f'ema_{long}']
                    features[f'ema_cross_{short}_{long}'] = ema_cross.astype(int)
        
        return features
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands indicators."""
        features = {}
        
        for period in self.config['bollinger_periods']:
            sma = data['Close'].rolling(period).mean()
            std = data['Close'].rolling(period).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            features[f'bb_upper_{period}'] = upper
            features[f'bb_lower_{period}'] = lower
            features[f'bb_width_{period}'] = upper - lower
            features[f'bb_position_{period}'] = (data['Close'] - lower) / (upper - lower)
            features[f'bb_squeeze_{period}'] = ((upper - lower) / sma).rolling(10).rank(pct=True)
            
            # Band touches
            features[f'bb_touch_upper_{period}'] = (data['High'] >= upper).astype(int)
            features[f'bb_touch_lower_{period}'] = (data['Low'] <= lower).astype(int)
        
        return features
    
    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD indicators."""
        features = {}
        
        # Standard MACD (12, 26, 9)
        exp12 = data['Close'].ewm(span=12).mean()
        exp26 = data['Close'].ewm(span=26).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = histogram
        features['macd_cross'] = (macd_line > signal_line).astype(int)
        features['macd_divergence'] = histogram.diff()
        features['macd_momentum'] = macd_line.diff()
        
        return features
    
    def _calculate_stochastic(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Stochastic oscillator."""
        features = {}
        
        for period in self.config['stochastic_periods']:
            low_min = data['Low'].rolling(period).min()
            high_max = data['High'].rolling(period).max()
            k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(3).mean()
            
            features[f'stoch_k_{period}'] = k_percent
            features[f'stoch_d_{period}'] = d_percent
            features[f'stoch_cross_{period}'] = (k_percent > d_percent).astype(int)
            features[f'stoch_oversold_{period}'] = (k_percent < 20).astype(int)
            features[f'stoch_overbought_{period}'] = (k_percent > 80).astype(int)
        
        return features


class MomentumIndicatorExtractor(BaseFeatureExtractor):
    """Extract momentum indicator features."""
    
    def __init__(self):
        super().__init__("MomentumIndicators")
        self.config = CONFIG.features.momentum_indicators
    
    def extract_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract momentum indicator features."""
        logger.info(f"Extracting {self.name} features", "🔧")
        features = {}
        
        # Returns and rankings
        for period in self.config['return_periods']:
            ret = data['Close'].pct_change(period)
            features[f'return_{period}d'] = ret
            features[f'return_rank_{period}d'] = ret.rolling(60).rank(pct=True)
            features[f'return_zscore_{period}d'] = (
                ret - ret.rolling(60).mean()
            ) / ret.rolling(60).std()
        
        # Volatility features
        returns = data['Close'].pct_change()
        for window in self.config['volatility_windows']:
            vol = returns.rolling(window).std()
            features[f'volatility_{window}'] = vol
            features[f'volatility_rank_{window}'] = vol.rolling(60).rank(pct=True)
        
        # Momentum features
        for period in self.config['momentum_periods']:
            momentum = (data['Close'] / data['Close'].shift(period) - 1) * 100
            features[f'momentum_{period}'] = momentum
            features[f'momentum_rank_{period}'] = momentum.rolling(60).rank(pct=True)
        
        self.feature_names = list(features.keys())
        logger.success(f"Generated {len(features)} momentum indicator features")
        
        return features


class VolumeIndicatorExtractor(BaseFeatureExtractor):
    """Extract volume indicator features."""
    
    def __init__(self):
        super().__init__("VolumeIndicators")
        self.config = CONFIG.features.volume_indicators
    
    def extract_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract volume indicator features."""
        logger.info(f"Extracting {self.name} features", "🔧")
        features = {}
        
        # Volume moving averages
        for window in self.config['volume_windows']:
            vol_sma = data['Volume'].rolling(window).mean()
            features[f'volume_sma_{window}'] = vol_sma
            features[f'volume_ratio_{window}'] = data['Volume'] / vol_sma
            features[f'volume_spike_{window}'] = (data['Volume'] > vol_sma * 2).astype(int)
        
        # Price-volume relationships
        for window in self.config['price_volume_windows']:
            price_change = data['Close'].pct_change(window)
            volume_change = data['Volume'].pct_change(window)
            features[f'price_vol_corr_{window}'] = price_change.rolling(window).corr(volume_change)
        
        # Price-volume trend
        price_change = data['Close'].pct_change()
        volume_change = data['Volume'].pct_change()
        features['price_volume_trend'] = (price_change * volume_change).rolling(5).mean()
        
        # On Balance Volume (OBV)
        obv = self._calculate_obv(data)
        features['obv'] = obv
        features['obv_sma_10'] = obv.rolling(10).mean()
        features['obv_sma_20'] = obv.rolling(20).mean()
        
        self.feature_names = list(features.keys())
        logger.success(f"Generated {len(features)} volume indicator features")
        
        return features
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = pd.Series(0.0, index=data.index)
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv


class PatternIndicatorExtractor(BaseFeatureExtractor):
    """Extract pattern indicator features."""
    
    def __init__(self):
        super().__init__("PatternIndicators")
        self.config = CONFIG.features.pattern_indicators
    
    def extract_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract pattern indicator features."""
        logger.info(f"Extracting {self.name} features", "🔧")
        features = {}
        
        # Support and resistance levels
        for window in self.config['support_resistance_windows']:
            high_max = data['High'].rolling(window).max()
            low_min = data['Low'].rolling(window).min()
            
            features[f'resistance_{window}'] = high_max
            features[f'support_{window}'] = low_min
            features[f'resistance_distance_{window}'] = (high_max - data['Close']) / data['Close']
            features[f'support_distance_{window}'] = (data['Close'] - low_min) / data['Close']
            
            # Level breakouts
            features[f'resistance_break_{window}'] = (
                data['Close'] > high_max.shift(1)
            ).astype(int)
            features[f'support_break_{window}'] = (
                data['Close'] < low_min.shift(1)
            ).astype(int)
        
        # Candlestick patterns (simplified)
        features.update(self._calculate_candlestick_patterns(data))
        
        # Gap analysis
        features.update(self._calculate_gaps(data))
        
        self.feature_names = list(features.keys())
        logger.success(f"Generated {len(features)} pattern indicator features")
        
        return features
    
    def _calculate_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate basic candlestick patterns."""
        features = {}
        
        body = abs(data['Close'] - data['Open'])
        upper_shadow = data['High'] - np.maximum(data['Close'], data['Open'])
        lower_shadow = np.minimum(data['Close'], data['Open']) - data['Low']
        total_range = data['High'] - data['Low']
        
        # Avoid division by zero
        total_range = total_range.where(total_range > 0, 1e-10)
        
        features['body_ratio'] = body / total_range
        features['upper_shadow_ratio'] = upper_shadow / total_range
        features['lower_shadow_ratio'] = lower_shadow / total_range
        features['body_position'] = (
            np.minimum(data['Close'], data['Open']) - data['Low']
        ) / total_range
        
        # Doji pattern (small body relative to range)
        features['doji'] = (body / total_range < 0.1).astype(int)
        
        return features
    
    def _calculate_gaps(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate gap analysis features."""
        features = {}
        
        overnight_gap = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        threshold = self.config['gap_threshold']
        
        features['gap_up'] = np.maximum(overnight_gap, 0)
        features['gap_down'] = np.maximum(-overnight_gap, 0)
        features['significant_gap_up'] = (overnight_gap > threshold).astype(int)
        features['significant_gap_down'] = (overnight_gap < -threshold).astype(int)
        
        # Gap filling
        features['gap_filled'] = (
            (overnight_gap > 0) & (data['Low'] <= data['Close'].shift(1))
        ).astype(int)
        
        return features


class FeatureEngineering:
    """Main feature engineering orchestrator."""
    
    def __init__(self):
        self.extractors = [
            TechnicalIndicatorExtractor(),
            MomentumIndicatorExtractor(), 
            VolumeIndicatorExtractor(),
            PatternIndicatorExtractor()
        ]
        self.all_feature_names = []
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features using registered extractors.
        
        Args:
            data: Stock data
            
        Returns:
            DataFrame with all engineered features
        """
        logger.section("Feature Engineering", "🔧")
        
        all_features = {}
        
        try:
            for extractor in self.extractors:
                features = extractor.extract_features(data)
                all_features.update(features)
                
                # Log progress
                logger.info(f"{extractor.name}: {len(features)} features")
            
            # Convert to DataFrame
            feature_df = pd.DataFrame(all_features, index=data.index)
            
            # Clean features
            feature_df = self._clean_features(feature_df)
            
            self.all_feature_names = list(feature_df.columns)
            
            logger.success(f"Total features generated: {len(feature_df.columns)}")
            
            return feature_df
            
        except Exception as e:
            raise FeatureEngineeringError(f"Feature engineering failed: {str(e)}")
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features."""
        logger.info("Cleaning and normalizing features", "🧹")
        
        initial_cols = len(features.columns)
        
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove features with constant values
        constant_features = []
        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            features = features.drop(columns=constant_features)
            logger.warning(f"Removed {len(constant_features)} constant features")
        
        # Outlier clipping (5-sigma rule)
        for col in features.columns:
            mean_val = features[col].mean()
            std_val = features[col].std()
            if std_val > 0:
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                features[col] = features[col].clip(lower_bound, upper_bound)
        
        final_cols = len(features.columns)
        logger.success(f"Feature cleaning complete: {final_cols}/{initial_cols} features retained")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.all_feature_names