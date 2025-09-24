"""
Volatility prediction models.

This module contains various models specifically designed for volatility prediction,
implementing the BasePredictor interface for consistency.
"""

from .elasticnet_predictor import ElasticNetVolatilityPredictor
from .ridge_predictor import RidgeVolatilityPredictor

__all__ = [
    'ElasticNetVolatilityPredictor',
    'RidgeVolatilityPredictor'
]