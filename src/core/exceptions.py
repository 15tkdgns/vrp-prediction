"""
Enhanced custom exceptions for the SPY Analysis system.
Provides a hierarchical exception structure for better error handling and debugging.
"""

from typing import Any, Dict, Optional


class SPYAnalysisError(Exception):
    """Base exception for all SPY Analysis errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class ConfigurationError(SPYAnalysisError):
    """Raised when there are configuration-related issues."""
    pass


class DataError(SPYAnalysisError):
    """Base class for data-related errors."""
    pass


class DataCollectionError(DataError):
    """Raised when data collection fails."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataProcessingError(DataError):
    """Raised when data processing fails."""
    pass


class InsufficientDataError(DataError):
    """Raised when there's insufficient data for processing."""
    pass


class ModelError(SPYAnalysisError):
    """Base class for model-related errors."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    pass


class FeatureEngineeringError(SPYAnalysisError):
    """Raised when feature engineering fails."""
    pass


class AnalysisError(SPYAnalysisError):
    """Base class for analysis-related errors."""
    pass


class XAIAnalysisError(AnalysisError):
    """Raised when XAI analysis fails."""
    pass


class PerformanceAnalysisError(AnalysisError):
    """Raised when performance analysis fails."""
    pass


class TradingError(SPYAnalysisError):
    """Base class for trading-related errors."""
    pass


class BacktestError(TradingError):
    """Raised when backtesting fails."""
    pass


class StrategyError(TradingError):
    """Raised when trading strategy execution fails."""
    pass


class APIError(SPYAnalysisError):
    """Base class for API-related errors."""
    pass


class ExternalAPIError(APIError):
    """Raised when external API calls fail."""
    pass


class WebAPIError(APIError):
    """Raised when web API errors occur."""
    pass


class DatabaseError(SPYAnalysisError):
    """Base class for database-related errors."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class QueryError(DatabaseError):
    """Raised when database query fails."""
    pass


# Error code mappings for structured error handling
ERROR_CODES = {
    # Configuration errors (1000-1099)
    "CONFIG_FILE_NOT_FOUND": 1001,
    "INVALID_CONFIG_FORMAT": 1002,
    "MISSING_REQUIRED_CONFIG": 1003,
    
    # Data errors (2000-2099)
    "DATA_SOURCE_UNAVAILABLE": 2001,
    "DATA_VALIDATION_FAILED": 2002,
    "INSUFFICIENT_DATA": 2003,
    "CORRUPT_DATA": 2004,
    
    # Model errors (3000-3099)
    "MODEL_TRAINING_FAILED": 3001,
    "MODEL_NOT_FOUND": 3002,
    "PREDICTION_FAILED": 3003,
    "MODEL_VALIDATION_FAILED": 3004,
    
    # Analysis errors (4000-4099)
    "XAI_ANALYSIS_FAILED": 4001,
    "PERFORMANCE_ANALYSIS_FAILED": 4002,
    "INSUFFICIENT_ANALYSIS_DATA": 4003,
    
    # Trading errors (5000-5099)
    "BACKTEST_FAILED": 5001,
    "STRATEGY_EXECUTION_FAILED": 5002,
    "INSUFFICIENT_MARKET_DATA": 5003,
    
    # API errors (6000-6099)
    "EXTERNAL_API_TIMEOUT": 6001,
    "API_RATE_LIMIT_EXCEEDED": 6002,
    "API_AUTHENTICATION_FAILED": 6003,
    "INVALID_API_RESPONSE": 6004,
    
    # Database errors (7000-7099)
    "DATABASE_CONNECTION_FAILED": 7001,
    "QUERY_EXECUTION_FAILED": 7002,
    "DATABASE_TIMEOUT": 7003,
}


def get_error_code(error_name: str) -> Optional[int]:
    """Get numeric error code from error name."""
    return ERROR_CODES.get(error_name)


def create_error_with_code(
    exception_class: type[SPYAnalysisError],
    message: str,
    error_name: str,
    **kwargs
) -> SPYAnalysisError:
    """Create an exception with proper error code."""
    error_code = get_error_code(error_name)
    return exception_class(
        message=message,
        error_code=str(error_code) if error_code else None,
        **kwargs
    )


# Legacy compatibility - 기존 코드 호환성을 위한 매핑
StockPredictionError = SPYAnalysisError
DataLoadError = DataCollectionError