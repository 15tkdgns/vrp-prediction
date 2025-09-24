"""
Pydantic models for API request/response validation.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import time


class PredictionRequest(BaseModel):
    """Request model for stock predictions."""
    symbol: str = Field(..., description="Stock symbol (e.g., 'SPY')")
    period: Optional[str] = Field("1y", description="Data period (e.g., '1y', '2y', '5y')")
    models: Optional[List[str]] = Field(None, description="Models to use for prediction")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 1 or len(v) > 10:
            raise ValueError('Symbol must be between 1 and 10 characters')
        return v.upper()
    
    @validator('period')
    def validate_period(cls, v):
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
        if v not in valid_periods:
            raise ValueError(f'Period must be one of: {valid_periods}')
        return v


class ModelPrediction(BaseModel):
    """Individual model prediction result."""
    class_: int = Field(..., alias="class", description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    confidence: float = Field(..., description="Prediction confidence")
    
    class Config:
        allow_population_by_field_name = True


class PredictionResponse(BaseModel):
    """Response model for stock predictions."""
    symbol: str
    predictions: Dict[str, Union[ModelPrediction, Dict[str, Any]]]
    timestamp: float
    features_count: int
    data_points_used: int


class HealthCheck(BaseModel):
    """Individual health check status."""
    status: str = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Additional information")


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str = Field(..., description="Overall system status")
    checks: Dict[str, str] = Field(..., description="Individual component checks")
    timestamp: float = Field(default_factory=time.time)


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    name: str
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    auc_score: float = Field(..., ge=0, le=1)
    training_date: str
    training_samples: int = Field(..., gt=0)


class TrainingRequest(BaseModel):
    """Request model for model training."""
    symbol: str = Field(..., description="Stock symbol to train on")
    period: str = Field("3y", description="Training data period")
    models: Optional[List[str]] = Field(None, description="Models to train")
    features_count: int = Field(30, description="Number of features to use")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set size")


class TrainingResponse(BaseModel):
    """Response model for training requests."""
    task_id: str
    status: str
    message: str
    timestamp: float = Field(default_factory=time.time)


class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    symbol: str
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(10000, gt=0)
    models: Optional[List[str]] = None


class BacktestResponse(BaseModel):
    """Response model for backtesting results."""
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    trades_count: int
    win_rate: float
    timestamp: float = Field(default_factory=time.time)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)