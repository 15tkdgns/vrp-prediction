"""
Ridge regression-based volatility predictor.

This module implements a Ridge regression model for volatility prediction,
serving as an alternative to ElasticNet with L2-only regularization.
"""

from __future__ import annotations

from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ...core.interfaces import BasePredictor, PredictionResult
from ...core.exceptions import ModelTrainingError, ModelPredictionError, ModelNotFittedError


class RidgeVolatilityPredictor(BasePredictor):
    """
    Ridge regression volatility predictor.

    This model uses L2 regularization for volatility prediction,
    providing a simpler alternative to ElasticNet with good baseline performance.

    Example:
        >>> predictor = RidgeVolatilityPredictor(alpha=0.001)
        >>> predictor.fit(X_train, y_train)
        >>> result = predictor.predict(X_test)
    """

    def __init__(
        self,
        alpha: float = 0.001,
        max_iter: int = 1000,
        random_state: Optional[int] = 42,
        normalize_features: bool = True,
        **kwargs
    ):
        """
        Initialize the Ridge volatility predictor.

        Args:
            alpha: Regularization strength
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            normalize_features: Whether to normalize features
            **kwargs: Additional parameters
        """
        super().__init__(name="Ridge Volatility Predictor", **kwargs)

        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.normalize_features = normalize_features

        self.model = Ridge(alpha=alpha, max_iter=max_iter, random_state=random_state)
        self.scaler = StandardScaler() if normalize_features else None
        self.training_metrics: Dict[str, float] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> RidgeVolatilityPredictor:
        """Train the Ridge model on volatility data."""
        try:
            self.validate_input(X)

            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
                X_array = X.values
            else:
                X_array = np.array(X)
                self.feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

            y_array = np.array(y)

            # Scale features if requested
            if self.scaler is not None:
                X_array = self.scaler.fit_transform(X_array)

            # Train the model
            self.model.fit(X_array, y_array)

            # Calculate training metrics
            predictions = self.model.predict(X_array)
            predictions = np.maximum(predictions, 0.0)

            self.training_metrics = {
                'train_r2': r2_score(y_array, predictions),
                'train_mae': mean_absolute_error(y_array, predictions),
                'train_mse': mean_squared_error(y_array, predictions)
            }

            self.is_fitted = True
            return self

        except Exception as e:
            raise ModelTrainingError(
                f"Failed to train Ridge volatility predictor: {str(e)}",
                model_name=self.name
            ) from e

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> PredictionResult:
        """Predict volatility for new data."""
        if not self.is_fitted:
            raise ModelNotFittedError(
                "Ridge model must be fitted before making predictions",
                model_name=self.name
            )

        try:
            self.validate_input(X)

            # Convert and scale
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = np.array(X)

            if self.scaler is not None:
                X_array = self.scaler.transform(X_array)

            # Make predictions
            predictions = self.model.predict(X_array)
            predictions = np.maximum(predictions, 0.0)

            metadata = {
                'model_type': 'Ridge',
                'target_type': 'next_day_volatility',
                'regularization_alpha': self.alpha,
                'n_features': X_array.shape[1],
                'n_predictions': len(predictions)
            }
            metadata.update(self.training_metrics)

            return PredictionResult(
                predictions=predictions,
                metadata=metadata,
                model_name=self.name,
                feature_importance=self.get_feature_importance()
            )

        except Exception as e:
            raise ModelPredictionError(
                f"Failed to predict with Ridge volatility predictor: {str(e)}",
                model_name=self.name
            ) from e

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from Ridge coefficients."""
        if not self.is_fitted or self.feature_names is None:
            return None

        coefficients = np.abs(self.model.coef_)
        if coefficients.sum() > 0:
            normalized_coefficients = coefficients / coefficients.sum()
        else:
            normalized_coefficients = coefficients

        return dict(zip(self.feature_names, normalized_coefficients))