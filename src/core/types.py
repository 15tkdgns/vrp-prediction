"""
Type definitions for the volatility prediction system.

This module provides type aliases and custom types used throughout the system,
improving code clarity and enabling better static type checking.
"""

from __future__ import annotations

from typing import TypeVar, Union, Dict, List, Tuple, Any, Optional, Callable, Protocol
from typing_extensions import TypedDict
import pandas as pd
import numpy as np
from numpy.typing import NDArray

# Basic type aliases
Number = Union[int, float]
ArrayLike = Union[np.ndarray, pd.Series, List[Number]]
DataFrame = pd.DataFrame
Series = pd.Series

# NumPy array types
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]

# Generic type variables
T = TypeVar('T')
ModelType = TypeVar('ModelType')
PredictorType = TypeVar('PredictorType')

# Configuration types
class DataConfig(TypedDict):
    """Configuration for data loading and processing."""
    symbol: str
    period: str
    start_date: Optional[str]
    end_date: Optional[str]
    data_source: str


class ModelConfig(TypedDict):
    """Configuration for model training."""
    model_type: str
    parameters: Dict[str, Any]
    validation_method: str
    test_size: float


class FeatureConfig(TypedDict):
    """Configuration for feature engineering."""
    feature_types: List[str]
    window_sizes: List[int]
    selection_method: str
    max_features: Optional[int]


class SystemConfig(TypedDict):
    """Overall system configuration."""
    data: DataConfig
    model: ModelConfig
    features: FeatureConfig
    logging_level: str
    output_dir: str


# Financial data types
class PriceData(TypedDict):
    """Standard OHLCV price data structure."""
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: pd.Timestamp


class VolatilityData(TypedDict):
    """Volatility calculation data."""
    value: float
    window: int
    method: str
    timestamp: pd.Timestamp


# Model result types
class TrainingResult(TypedDict):
    """Result of model training."""
    model: Any
    training_score: float
    validation_score: float
    test_score: float
    feature_importance: Optional[Dict[str, float]]
    training_time: float


class PredictionOutput(TypedDict):
    """Standardized prediction output."""
    predictions: FloatArray
    confidence: Optional[FloatArray]
    model_name: str
    prediction_time: float


class EvaluationMetrics(TypedDict):
    """Standard evaluation metrics."""
    r2: float
    mae: float
    mse: float
    rmse: float
    mape: float
    correlation: float


class VolatilityMetrics(EvaluationMetrics):
    """Volatility-specific evaluation metrics."""
    direction_accuracy: float
    high_vol_precision: float
    high_vol_recall: float
    high_vol_f1: float


# Protocol definitions for duck typing
class Predictable(Protocol):
    """Protocol for objects that can make predictions."""

    def predict(self, X: ArrayLike) -> PredictionOutput:
        """Make predictions on input data."""
        ...

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit the model to training data."""
        ...


class Evaluable(Protocol):
    """Protocol for objects that can be evaluated."""

    def evaluate(
        self,
        predictions: ArrayLike,
        actuals: ArrayLike
    ) -> Dict[str, float]:
        """Evaluate predictions against actual values."""
        ...


class Transformable(Protocol):
    """Protocol for objects that can transform data."""

    def transform(self, data: DataFrame) -> DataFrame:
        """Transform input data."""
        ...

    def fit_transform(self, data: DataFrame) -> DataFrame:
        """Fit and transform data in one step."""
        ...


# Function type aliases
FeatureFunction = Callable[[DataFrame], Series]
ValidationFunction = Callable[[ArrayLike, ArrayLike], Dict[str, float]]
ModelFactory = Callable[..., Predictable]

# Ensemble types
EnsembleWeights = Dict[str, float]
EnsemblePredictions = Dict[str, FloatArray]

class EnsembleResult(TypedDict):
    """Result from ensemble prediction."""
    final_prediction: FloatArray
    individual_predictions: EnsemblePredictions
    weights: EnsembleWeights
    confidence: Optional[FloatArray]


# Time series types
TimeSeriesData = pd.DataFrame  # DataFrame with DatetimeIndex
Window = Tuple[pd.Timestamp, pd.Timestamp]
DateRange = Tuple[Optional[str], Optional[str]]

# Validation types
class ValidationSplit(TypedDict):
    """Time series validation split."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


CrossValidationResult = List[TrainingResult]

# API types
class APIResponse(TypedDict):
    """Standard API response structure."""
    success: bool
    data: Optional[Any]
    error: Optional[str]
    timestamp: str


class HealthCheckResult(TypedDict):
    """Health check result structure."""
    status: str
    version: str
    uptime: float
    components: Dict[str, str]


# File path types
FilePath = Union[str, 'pathlib.Path']
DataFilePath = FilePath
ModelFilePath = FilePath
ConfigFilePath = FilePath

# Logging types
LogLevel = Union[str, int]
LogRecord = Dict[str, Any]

# Custom exceptions that we want to type
ExceptionContext = Dict[str, Any]

# Feature engineering types
FeatureMatrix = DataFrame
TargetVector = Series
FeatureNames = List[str]
FeatureImportance = Dict[str, float]

class FeatureSetResult(TypedDict):
    """Result of feature set creation."""
    features: FeatureMatrix
    feature_names: FeatureNames
    metadata: Dict[str, Any]


# Model selection types
ModelName = str
ModelParameters = Dict[str, Any]
ModelScore = float

class ModelCandidate(TypedDict):
    """Model candidate for selection."""
    name: ModelName
    model: Any
    parameters: ModelParameters
    score: ModelScore


ModelCandidates = List[ModelCandidate]

# Data source types
class DataSource(TypedDict):
    """Data source configuration."""
    name: str
    type: str
    connection_params: Dict[str, Any]
    cache_enabled: bool


# Performance monitoring types
class PerformanceMetrics(TypedDict):
    """Performance monitoring metrics."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float


# Risk management types
class RiskMetrics(TypedDict):
    """Risk assessment metrics."""
    value_at_risk: float
    expected_shortfall: float
    maximum_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float


# Market regime types
MarketRegime = str  # 'bull', 'bear', 'sideways', 'volatile', 'calm'

class RegimeClassification(TypedDict):
    """Market regime classification result."""
    regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    timestamp: pd.Timestamp


# Feature selection types
FeatureSelector = Callable[[FeatureMatrix, TargetVector], FeatureNames]
SelectionCriteria = Dict[str, Any]

class FeatureSelectionResult(TypedDict):
    """Result of feature selection."""
    selected_features: FeatureNames
    feature_scores: Dict[str, float]
    selection_method: str
    criteria: SelectionCriteria