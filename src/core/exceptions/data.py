"""
Data-related exception classes.
"""

from typing import Optional, Any, Dict, List
from .base import VolatilityPredictionError


class DataValidationError(VolatilityPredictionError):
    """
    Exception raised when data validation fails.

    This exception is raised when input data doesn't meet the required
    format, contains invalid values, or fails other validation checks.
    """

    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the data validation error.

        Args:
            message: Error message
            validation_errors: List of specific validation errors
            **kwargs: Additional context
        """
        super().__init__(message, error_code="DATA_VALIDATION", **kwargs)
        self.validation_errors = validation_errors or []

    def __str__(self) -> str:
        """Return string representation including validation errors."""
        error_str = super().__str__()
        if self.validation_errors:
            errors_str = "; ".join(self.validation_errors)
            error_str = f"{error_str} Validation errors: {errors_str}"
        return error_str


class DataLoadError(VolatilityPredictionError):
    """
    Exception raised when data loading fails.

    This exception is raised when data cannot be loaded from the specified
    source, whether it's a file, database, or API.
    """

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the data load error.

        Args:
            message: Error message
            source: Data source that failed to load
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if source:
            context['source'] = source
        kwargs['context'] = context
        
        super().__init__(message, error_code="DATA_LOAD", **kwargs)
        self.source = source


class FeatureEngineeringError(VolatilityPredictionError):
    """
    Exception raised when feature engineering fails.

    This exception is raised when feature creation, transformation,
    or selection operations encounter errors.
    """

    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the feature engineering error.

        Args:
            message: Error message
            feature_name: Name of the feature that caused the error
            operation: Feature engineering operation that failed
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if feature_name:
            context['feature_name'] = feature_name
        if operation:
            context['operation'] = operation
        kwargs['context'] = context
        
        super().__init__(message, error_code="FEATURE_ENGINEERING", **kwargs)
        self.feature_name = feature_name
        self.operation = operation


class DataLeakageError(VolatilityPredictionError):
    """
    Exception raised when potential data leakage is detected.

    This exception is raised when the system detects features or operations
    that might cause data leakage, compromising model validity.
    """

    def __init__(
        self,
        message: str,
        leakage_type: Optional[str] = None,
        affected_features: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the data leakage error.

        Args:
            message: Error message
            leakage_type: Type of data leakage detected
            affected_features: Features that might cause leakage
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if leakage_type:
            context['leakage_type'] = leakage_type
        if affected_features:
            context['affected_features'] = affected_features
        kwargs['context'] = context
        
        super().__init__(message, error_code="DATA_LEAKAGE", **kwargs)
        self.leakage_type = leakage_type
        self.affected_features = affected_features or []
