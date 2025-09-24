"""
Volatility prediction module.

This module contains all components for predicting financial volatility,
including feature engineering, models, and evaluation methods specifically
designed for volatility forecasting.

The module is organized into the following submodules:
- predictors: Various volatility prediction models
- features: Volatility-specific feature engineering
- ensemble: Ensemble methods for volatility prediction
- evaluation: Volatility-specific evaluation metrics
"""

from .predictors import (
    ElasticNetVolatilityPredictor,
    RidgeVolatilityPredictor
)
from .features import VolatilityFeatureEngineer
from .evaluation import VolatilityEvaluator

__all__ = [
    'ElasticNetVolatilityPredictor',
    'RidgeVolatilityPredictor',
    'VolatilityFeatureEngineer',
    'VolatilityEvaluator'
]

__version__ = "2.0.0"