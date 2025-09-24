"""
Custom exceptions for the volatility prediction system.

This module defines specific exceptions that can be raised throughout the system,
providing better error handling and debugging capabilities.
"""

from .base import VolatilityPredictionError
from .data import DataValidationError, DataLoadError, FeatureEngineeringError
from .model import ModelTrainingError, ModelPredictionError, ModelValidationError, ModelNotFittedError
from .config import ConfigurationError

__all__ = [
    'VolatilityPredictionError',
    'DataValidationError',
    'DataLoadError',
    'FeatureEngineeringError',
    'ModelTrainingError',
    'ModelPredictionError',
    'ModelValidationError',
    'ModelNotFittedError',
    'ConfigurationError'
]