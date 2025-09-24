"""
Model-related exception classes.
"""

from typing import Optional, Any, Dict
from .base import VolatilityPredictionError


class ModelTrainingError(VolatilityPredictionError):
    """
    Exception raised when model training fails.

    This exception is raised when model training encounters errors,
    such as convergence issues, invalid parameters, or data problems.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        training_step: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the model training error.

        Args:
            message: Error message
            model_name: Name of the model being trained
            training_step: Specific training step that failed
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if model_name:
            context['model_name'] = model_name
        if training_step:
            context['training_step'] = training_step
        kwargs['context'] = context
        
        super().__init__(message, error_code="MODEL_TRAINING", **kwargs)
        self.model_name = model_name
        self.training_step = training_step


class ModelPredictionError(VolatilityPredictionError):
    """
    Exception raised when model prediction fails.

    This exception is raised when a trained model fails to generate
    predictions, either due to input issues or model state problems.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        **kwargs
    ):
        """
        Initialize the model prediction error.

        Args:
            message: Error message
            model_name: Name of the model making predictions
            input_shape: Shape of the input data
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if model_name:
            context['model_name'] = model_name
        if input_shape:
            context['input_shape'] = input_shape
        kwargs['context'] = context
        
        super().__init__(message, error_code="MODEL_PREDICTION", **kwargs)
        self.model_name = model_name
        self.input_shape = input_shape


class ModelValidationError(VolatilityPredictionError):
    """
    Exception raised when model validation fails.

    This exception is raised when model validation checks fail,
    such as performance thresholds not being met or validation data issues.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        validation_metric: Optional[str] = None,
        expected_value: Optional[float] = None,
        actual_value: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the model validation error.

        Args:
            message: Error message
            model_name: Name of the model being validated
            validation_metric: Metric that failed validation
            expected_value: Expected metric value
            actual_value: Actual metric value
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if model_name:
            context['model_name'] = model_name
        if validation_metric:
            context['validation_metric'] = validation_metric
        if expected_value is not None:
            context['expected_value'] = expected_value
        if actual_value is not None:
            context['actual_value'] = actual_value
        kwargs['context'] = context
        
        super().__init__(message, error_code="MODEL_VALIDATION", **kwargs)
        self.model_name = model_name
        self.validation_metric = validation_metric
        self.expected_value = expected_value
        self.actual_value = actual_value


class ModelNotFittedError(VolatilityPredictionError):
    """
    Exception raised when trying to use an unfitted model.

    This exception is raised when operations requiring a fitted model
    are called on a model that hasn't been trained yet.
    """

    def __init__(
        self,
        message: str = "Model must be fitted before making predictions",
        model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the model not fitted error.

        Args:
            message: Error message
            model_name: Name of the unfitted model
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if model_name:
            context['model_name'] = model_name
        kwargs['context'] = context
        
        super().__init__(message, error_code="MODEL_NOT_FITTED", **kwargs)
        self.model_name = model_name
