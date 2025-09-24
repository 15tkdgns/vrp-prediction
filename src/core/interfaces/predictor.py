"""
Abstract base class for all prediction models.

This module defines the contract that all predictors must implement,
ensuring consistency across different prediction algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


@dataclass
class PredictionResult:
    """
    Standardized result structure for all predictions.

    Attributes:
        predictions: Array of predicted values
        confidence: Confidence scores for each prediction (0-1)
        metadata: Additional information about the prediction
        model_name: Name of the model that made the prediction
        feature_importance: Importance scores for features used
    """
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None


class BasePredictor(ABC):
    """
    Abstract base class for all prediction models.

    This class defines the interface that all predictors must implement,
    including volatility predictors, return predictors, and ensemble methods.

    Example:
        >>> class MyPredictor(BasePredictor):
        ...     def fit(self, X, y):
        ...         # Training logic here
        ...         pass
        ...
        ...     def predict(self, X):
        ...         # Prediction logic here
        ...         return PredictionResult(predictions=predictions)
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the predictor.

        Args:
            name: Human-readable name for this predictor
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.model: Optional[Any] = None

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> BasePredictor:
        """
        Train the model on the provided data.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is invalid or incompatible
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> PredictionResult:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult containing predictions and metadata

        Raises:
            ValueError: If model is not fitted or data is invalid
        """
        pass

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> PredictionResult:
        """
        Make probabilistic predictions (optional).

        Args:
            X: Feature matrix
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult with probability distributions

        Raises:
            NotImplementedError: If the model doesn't support probabilities
        """
        raise NotImplementedError(f"{self.name} does not support probabilistic predictions")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available
        """
        return None

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: File path to save the model

        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        import joblib
        joblib.dump({
            'model': self.model,
            'name': self.name,
            'config': self.config,
            'feature_names': self.feature_names
        }, path)

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path: File path to load the model from

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        import joblib
        saved_data = joblib.load(path)

        self.model = saved_data['model']
        self.name = saved_data['name']
        self.config = saved_data['config']
        self.feature_names = saved_data['feature_names']
        self.is_fitted = True

    def validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Validate input data format and content.

        Args:
            X: Input feature matrix

        Raises:
            ValueError: If data is invalid
        """
        if X is None or len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if isinstance(X, pd.DataFrame):
            if X.isnull().any().any():
                raise ValueError("Input data contains null values")
        elif isinstance(X, np.ndarray):
            if np.isnan(X).any():
                raise ValueError("Input data contains NaN values")

    def __repr__(self) -> str:
        """String representation of the predictor."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', status='{fitted_status}')"