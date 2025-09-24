"""
Core interfaces and abstract base classes for the volatility prediction system.

This module defines the fundamental contracts that all components must implement,
ensuring consistency and enabling polymorphism throughout the system.
"""

from .predictor import BasePredictor, PredictionResult
from .feature_engineer import BaseFeatureEngineer, FeatureResult
from .evaluator import BaseEvaluator, EvaluationResult

__all__ = [
    'BasePredictor',
    'PredictionResult',
    'BaseFeatureEngineer',
    'FeatureResult',
    'BaseEvaluator',
    'EvaluationResult'
]