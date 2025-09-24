"""
Volatility-specific feature engineering.

This module provides feature engineering specifically designed for volatility prediction,
including technical indicators, volatility measures, and time-based features.
"""

from __future__ import annotations

from typing import Union, Optional, Dict, Any, List
import numpy as np
import pandas as pd

from ..core.interfaces import BaseFeatureEngineer, FeatureResult
from ..core.exceptions import FeatureEngineeringError
from ..core.types import DataFrame, Series


class VolatilityFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer specialized for volatility prediction.

    This class creates features that are specifically useful for predicting
    financial volatility, including various volatility measures, momentum indicators,
    and lagged features.

    Example:
        >>> engineer = VolatilityFeatureEngineer(
        ...     volatility_windows=[5, 10, 20],
        ...     include_lags=True,
        ...     max_lags=3
        ... )
        >>> result = engineer.fit_transform(price_data)
        >>> print(f"Created {len(result.feature_names)} features")
    """

    def __init__(
        self,
        volatility_windows: List[int] = [5, 10, 20],
        ma_windows: List[int] = [20, 50],
        include_lags: bool = True,
        max_lags: int = 3,
        include_volume: bool = True,
        **kwargs
    ):
        """
        Initialize the volatility feature engineer.

        Args:
            volatility_windows: Windows for volatility calculations
            ma_windows: Windows for moving averages
            include_lags: Whether to include lagged features
            max_lags: Maximum number of lags to include
            include_volume: Whether to include volume-based features
            **kwargs: Additional parameters
        """
        super().__init__(name="Volatility Feature Engineer", **kwargs)

        self.volatility_windows = volatility_windows
        self.ma_windows = ma_windows
        self.include_lags = include_lags
        self.max_lags = max_lags
        self.include_volume = include_volume

    def create_features(
        self,
        data: DataFrame,
        **kwargs
    ) -> FeatureResult:
        """
        Create volatility-focused features from price data.

        Args:
            data: OHLCV price data
            **kwargs: Additional parameters

        Returns:
            FeatureResult containing engineered features

        Raises:
            FeatureEngineeringError: If feature creation fails
        """
        try:
            self.validate_data(data)

            # Ensure columns are in the right case
            data_clean = data.copy()
            data_clean.columns = data_clean.columns.str.lower()

            features = pd.DataFrame(index=data_clean.index)
            feature_descriptions = {}

            # 1. Basic volatility measures
            features, descriptions = self._add_volatility_features(data_clean, features)
            feature_descriptions.update(descriptions)

            # 2. Moving averages and technical indicators
            features, descriptions = self._add_technical_indicators(data_clean, features)
            feature_descriptions.update(descriptions)

            # 3. Volume-based features
            if self.include_volume:
                features, descriptions = self._add_volume_features(data_clean, features)
                feature_descriptions.update(descriptions)

            # 4. Lagged features
            if self.include_lags:
                features, descriptions = self._add_lagged_features(data_clean, features)
                feature_descriptions.update(descriptions)

            # 5. Target variable (next-day volatility)
            target = self._calculate_target_volatility(data_clean)
            features['next_day_volatility'] = target

            # Remove any rows with NaN values
            features_clean = features.dropna()

            # Prepare metadata
            metadata = {
                'original_length': len(data),
                'final_length': len(features_clean),
                'features_created': len(features_clean.columns),
                'volatility_windows': self.volatility_windows,
                'ma_windows': self.ma_windows,
                'max_lags': self.max_lags,
                'feature_descriptions': feature_descriptions
            }

            return FeatureResult(
                features=features_clean,
                feature_names=list(features_clean.columns),
                metadata=metadata,
                creation_method="volatility_focused",
                quality_metrics=self._calculate_quality_metrics(features_clean)
            )

        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to create volatility features: {str(e)}",
                operation="create_features"
            ) from e

    def _add_volatility_features(
        self,
        data: DataFrame,
        features: DataFrame
    ) -> tuple[DataFrame, Dict[str, str]]:
        """Add various volatility measures."""
        descriptions = {}

        # Calculate returns
        returns = data['close'].pct_change()

        # Volatility measures for different windows
        for window in self.volatility_windows:
            # Rolling standard deviation of returns
            vol_name = f'volatility_{window}'
            features[vol_name] = returns.rolling(window=window).std()
            descriptions[vol_name] = f"{window}-day rolling volatility (std of returns)"

            # High-Low volatility (Garman-Klass estimator)
            hl_vol_name = f'hl_volatility_{window}'
            hl_ratio = np.log(data['high'] / data['low'])
            features[hl_vol_name] = hl_ratio.rolling(window=window).std()
            descriptions[hl_vol_name] = f"{window}-day high-low volatility"

        # Current day volatility proxies
        features['intraday_range'] = (data['high'] - data['low']) / data['close']
        descriptions['intraday_range'] = "Intraday price range normalized by close"

        features['true_range'] = self._calculate_true_range(data)
        descriptions['true_range'] = "True range (max of HL, HC, LC)"

        return features, descriptions

    def _add_technical_indicators(
        self,
        data: DataFrame,
        features: DataFrame
    ) -> tuple[DataFrame, Dict[str, str]]:
        """Add technical indicators useful for volatility prediction."""
        descriptions = {}

        # Moving averages
        for window in self.ma_windows:
            ma_name = f'ma_{window}'
            features[ma_name] = data['close'].rolling(window=window).mean()
            descriptions[ma_name] = f"{window}-day moving average"

        # RSI
        rsi = self._calculate_rsi(data['close'], window=14)
        features['rsi'] = rsi
        descriptions['rsi'] = "14-day Relative Strength Index"

        # Bollinger Band position
        if 20 in self.ma_windows:
            ma20 = features['ma_20']
            bb_std = data['close'].rolling(window=20).std()
            features['bb_position'] = (data['close'] - ma20) / (2 * bb_std)
            descriptions['bb_position'] = "Position within Bollinger Bands"

        # Price momentum
        features['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
        features['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
        descriptions['price_momentum_5'] = "5-day price momentum"
        descriptions['price_momentum_10'] = "10-day price momentum"

        return features, descriptions

    def _add_volume_features(
        self,
        data: DataFrame,
        features: DataFrame
    ) -> tuple[DataFrame, Dict[str, str]]:
        """Add volume-based features."""
        descriptions = {}

        # Volume ratios
        for window in [10, 20]:
            vol_ratio_name = f'volume_ratio_{window}'
            avg_volume = data['volume'].rolling(window=window).mean()
            features[vol_ratio_name] = data['volume'] / avg_volume
            descriptions[vol_ratio_name] = f"Current volume / {window}-day average volume"

        # Volume-price trend
        features['volume_price_trend'] = (
            data['volume'] * (data['close'] - data['close'].shift(1))
        ).rolling(window=10).sum()
        descriptions['volume_price_trend'] = "10-day volume-weighted price trend"

        return features, descriptions

    def _add_lagged_features(
        self,
        data: DataFrame,
        features: DataFrame
    ) -> tuple[DataFrame, Dict[str, str]]:
        """Add lagged features for time series modeling."""
        descriptions = {}

        # Lag important features
        returns = data['close'].pct_change()

        for lag in range(1, self.max_lags + 1):
            # Lagged returns
            lag_name = f'returns_lag_{lag}'
            features[lag_name] = returns.shift(lag)
            descriptions[lag_name] = f"Returns lagged by {lag} day(s)"

            # Lagged RSI
            if 'rsi' in features.columns:
                rsi_lag_name = f'rsi_lag_{lag}'
                features[rsi_lag_name] = features['rsi'].shift(lag)
                descriptions[rsi_lag_name] = f"RSI lagged by {lag} day(s)"

            # Lagged volatility
            if f'volatility_20' in features.columns:
                vol_lag_name = f'volatility_20_lag_{lag}'
                features[vol_lag_name] = features['volatility_20'].shift(lag)
                descriptions[vol_lag_name] = f"20-day volatility lagged by {lag} day(s)"

            # Lagged Bollinger Band position
            if 'bb_position' in features.columns:
                bb_lag_name = f'bb_position_lag_{lag}'
                features[bb_lag_name] = features['bb_position'].shift(lag)
                descriptions[bb_lag_name] = f"BB position lagged by {lag} day(s)"

        return features, descriptions

    def _calculate_target_volatility(self, data: DataFrame) -> Series:
        """Calculate the target variable: next-day volatility."""
        # Calculate next-day realized volatility using close-to-close returns
        returns = data['close'].pct_change()
        # Shift forward to get next day's volatility
        next_day_volatility = returns.abs().shift(-1)
        return next_day_volatility

    def _calculate_true_range(self, data: DataFrame) -> Series:
        """Calculate True Range indicator."""
        prev_close = data['close'].shift(1)
        tr1 = data['high'] - data['low']
        tr2 = np.abs(data['high'] - prev_close)
        tr3 = np.abs(data['low'] - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))

    def _calculate_rsi(self, prices: Series, window: int = 14) -> Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_quality_metrics(self, features: DataFrame) -> Dict[str, float]:
        """Calculate quality metrics for the generated features."""
        numeric_features = features.select_dtypes(include=[np.number])

        return {
            'completeness': 1 - (numeric_features.isnull().sum().sum() / numeric_features.size),
            'mean_correlation': np.abs(numeric_features.corr()).mean().mean(),
            'variance_threshold': (numeric_features.var() > 1e-6).mean(),
            'feature_count': len(numeric_features.columns)
        }