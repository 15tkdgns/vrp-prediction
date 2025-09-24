"""
Base exception classes for the volatility prediction system.
"""

from typing import Optional, Any, Dict


class VolatilityPredictionError(Exception):
    """
    Base exception class for all volatility prediction system errors.

    This is the parent class for all custom exceptions in the system,
    providing a common interface and additional context information.

    Attributes:
        message: Error message
        error_code: Optional error code for categorization
        context: Additional context information
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            error_code: Optional code for error categorization
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        error_str = self.message
        if self.error_code:
            error_str = f"[{self.error_code}] {error_str}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            error_str = f"{error_str} (Context: {context_str})"
        return error_str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary representation.

        Returns:
            Dictionary containing error information
        """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context
        }