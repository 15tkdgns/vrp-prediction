"""
Configuration-related exception classes.
"""

from typing import Optional, Any, Dict, List
from .base import VolatilityPredictionError


class ConfigurationError(VolatilityPredictionError):
    """
    Exception raised when configuration is invalid or missing.

    This exception is raised when configuration files are malformed,
    required parameters are missing, or configuration values are invalid.
    """

    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        invalid_parameters: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the configuration error.

        Args:
            message: Error message
            config_section: Configuration section that has the error
            invalid_parameters: List of invalid parameter names
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if config_section:
            context['config_section'] = config_section
        if invalid_parameters:
            context['invalid_parameters'] = invalid_parameters
        kwargs['context'] = context
        
        super().__init__(message, error_code="CONFIGURATION", **kwargs)
        self.config_section = config_section
        self.invalid_parameters = invalid_parameters or []


class APIConfigurationError(ConfigurationError):
    """
    Exception raised when API configuration is invalid.

    This exception is raised when API keys are missing, invalid,
    or when API endpoints are misconfigured.
    """

    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        missing_keys: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the API configuration error.

        Args:
            message: Error message
            api_name: Name of the API with configuration issues
            missing_keys: List of missing API keys
            **kwargs: Additional context
        """
        context = kwargs.get('context', {})
        if api_name:
            context['api_name'] = api_name
        if missing_keys:
            context['missing_keys'] = missing_keys
        kwargs['context'] = context
        
        super().__init__(
            message,
            config_section='api',
            invalid_parameters=missing_keys,
            **kwargs
        )
        self.api_name = api_name
        self.missing_keys = missing_keys or []
