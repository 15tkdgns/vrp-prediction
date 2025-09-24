"""
Abstract base class for feature engineering components.

This module defines the contract for all feature engineering operations,
ensuring consistency and composability across different feature creation methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd
import numpy as np


@dataclass
class FeatureResult:
    """
    Standardized result structure for feature engineering operations.

    Attributes:
        features: DataFrame containing the engineered features
        feature_names: List of feature column names
        metadata: Additional information about the features
        creation_method: Method used to create the features
        quality_metrics: Quality metrics for the generated features
    """
    features: pd.DataFrame
    feature_names: List[str]
    metadata: Optional[Dict[str, Any]] = None
    creation_method: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None


class BaseFeatureEngineer(ABC):
    """
    Abstract base class for all feature engineering components.

    This class defines the interface for creating, transforming, and selecting
    features from raw financial data.

    Example:
        >>> class TechnicalIndicators(BaseFeatureEngineer):
        ...     def create_features(self, data):
        ...         # Create technical indicators
        ...         features = self._calculate_indicators(data)
        ...         return FeatureResult(features=features, feature_names=list(features.columns))
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the feature engineer.

        Args:
            name: Human-readable name for this feature engineer
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs
        self.is_fitted = False
        self.feature_columns: Optional[List[str]] = None
        self.scaler: Optional[Any] = None

    @abstractmethod
    def create_features(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> FeatureResult:
        """
        Create features from raw data.

        Args:
            data: Raw input data (typically OHLCV format)
            **kwargs: Additional parameters for feature creation

        Returns:
            FeatureResult containing engineered features and metadata

        Raises:
            ValueError: If input data is invalid
        """
        pass

    def fit_transform(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> FeatureResult:
        """
        Fit the feature engineer and transform data in one step.

        Args:
            data: Raw input data
            **kwargs: Additional parameters

        Returns:
            FeatureResult containing transformed features
        """
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)

    def fit(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> BaseFeatureEngineer:
        """
        Fit the feature engineer on training data.

        This method learns any necessary parameters for feature transformation,
        such as scaling parameters or feature selection criteria.

        Args:
            data: Training data
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        self.is_fitted = True
        return self

    def transform(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> FeatureResult:
        """
        Transform new data using fitted parameters.

        Args:
            data: Data to transform
            **kwargs: Additional transformation parameters

        Returns:
            FeatureResult containing transformed features

        Raises:
            ValueError: If feature engineer is not fitted
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before transform")

        return self.create_features(data, **kwargs)

    def select_features(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, np.ndarray],
        method: str = 'correlation',
        n_features: Optional[int] = None
    ) -> FeatureResult:
        """
        Select the most relevant features for prediction.

        Args:
            features: Feature matrix
            target: Target variable
            method: Feature selection method ('correlation', 'mutual_info', 'rfe')
            n_features: Number of features to select (None for automatic)

        Returns:
            FeatureResult with selected features

        Raises:
            ValueError: If method is not supported
        """
        if method == 'correlation':
            return self._select_by_correlation(features, target, n_features)
        elif method == 'mutual_info':
            return self._select_by_mutual_info(features, target, n_features)
        elif method == 'rfe':
            return self._select_by_rfe(features, target, n_features)
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")

    def _select_by_correlation(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, np.ndarray],
        n_features: Optional[int]
    ) -> FeatureResult:
        """Select features based on correlation with target."""
        correlations = features.corrwith(pd.Series(target))
        correlations = correlations.abs().sort_values(ascending=False)

        if n_features is None:
            # Select features with correlation > 0.1
            selected = correlations[correlations > 0.1].index.tolist()
        else:
            selected = correlations.head(n_features).index.tolist()

        selected_features = features[selected]
        return FeatureResult(
            features=selected_features,
            feature_names=selected,
            metadata={'selection_method': 'correlation', 'correlations': correlations.to_dict()},
            creation_method=f"{self.name}_correlation_selection"
        )

    def _select_by_mutual_info(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, np.ndarray],
        n_features: Optional[int]
    ) -> FeatureResult:
        """Select features based on mutual information."""
        from sklearn.feature_selection import mutual_info_regression

        # Handle any NaN values
        features_clean = features.fillna(features.mean())
        mi_scores = mutual_info_regression(features_clean, target)

        feature_scores = pd.Series(mi_scores, index=features.columns)
        feature_scores = feature_scores.sort_values(ascending=False)

        if n_features is None:
            # Select features with MI score > median
            threshold = feature_scores.median()
            selected = feature_scores[feature_scores > threshold].index.tolist()
        else:
            selected = feature_scores.head(n_features).index.tolist()

        selected_features = features[selected]
        return FeatureResult(
            features=selected_features,
            feature_names=selected,
            metadata={'selection_method': 'mutual_info', 'mi_scores': feature_scores.to_dict()},
            creation_method=f"{self.name}_mutual_info_selection"
        )

    def _select_by_rfe(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, np.ndarray],
        n_features: Optional[int]
    ) -> FeatureResult:
        """Select features using Recursive Feature Elimination."""
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LinearRegression

        estimator = LinearRegression()
        n_features = n_features or max(5, len(features.columns) // 2)

        # Handle any NaN values
        features_clean = features.fillna(features.mean())

        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(features_clean, target)

        selected = features.columns[selector.support_].tolist()
        selected_features = features[selected]

        return FeatureResult(
            features=selected_features,
            feature_names=selected,
            metadata={'selection_method': 'rfe', 'ranking': dict(zip(features.columns, selector.ranking_))},
            creation_method=f"{self.name}_rfe_selection"
        )

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data format and content.

        Args:
            data: Input data to validate

        Raises:
            ValueError: If data is invalid
        """
        if data is None or len(data) == 0:
            raise ValueError("Input data cannot be empty")

        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(data.columns.str.lower()):
            missing = required_columns - set(data.columns.str.lower())
            raise ValueError(f"Missing required columns: {missing}")

        if data.isnull().any().any():
            raise ValueError("Input data contains null values")

    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get human-readable descriptions of created features.

        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {}

    def save_feature_config(self, path: str) -> None:
        """
        Save feature engineering configuration.

        Args:
            path: File path to save configuration
        """
        import json
        config_data = {
            'name': self.name,
            'config': self.config,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)

    def load_feature_config(self, path: str) -> None:
        """
        Load feature engineering configuration.

        Args:
            path: File path to load configuration from
        """
        import json
        with open(path, 'r') as f:
            config_data = json.load(f)

        self.name = config_data['name']
        self.config = config_data['config']
        self.feature_columns = config_data['feature_columns']
        self.is_fitted = config_data['is_fitted']

    def __repr__(self) -> str:
        """String representation of the feature engineer."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', status='{fitted_status}')"