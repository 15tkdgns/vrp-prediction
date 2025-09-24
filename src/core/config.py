"""
Configuration management for stock prediction system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    params: Dict
    enabled: bool = True
    
    
@dataclass  
class FeatureConfig:
    """Feature engineering configuration."""
    technical_indicators: Dict = field(default_factory=lambda: {
        'rsi_periods': [5, 10, 14, 21],
        'ma_periods': [5, 10, 20, 50],
        'bollinger_periods': [10, 20],
        'stochastic_periods': [14, 21]
    })
    
    momentum_indicators: Dict = field(default_factory=lambda: {
        'return_periods': [1, 2, 3, 5, 8, 13, 21],  # Fibonacci sequence
        'volatility_windows': [5, 10, 20],
        'momentum_periods': [5, 10, 20]
    })
    
    volume_indicators: Dict = field(default_factory=lambda: {
        'volume_windows': [10, 20],
        'price_volume_windows': [5, 10, 15, 20, 30]
    })
    
    pattern_indicators: Dict = field(default_factory=lambda: {
        'support_resistance_windows': [10, 20],
        'gap_threshold': 0.01
    })


@dataclass
class DataConfig:
    """Data processing configuration."""
    symbol: str = 'SPY'
    period: str = '3y'
    target_threshold: float = 0.01  # 1% threshold for binary classification
    test_size: float = 0.15
    validation_size: float = 0.15
    feature_selection_k: int = 30
    

@dataclass
class TrainingConfig:
    """Training configuration."""
    random_state: int = 42
    balance_method: str = 'smote'  # 'smote', 'adasyn', 'smoteenn', None
    scaling_method: str = 'robust'  # 'robust', 'standard', 'minmax'
    cross_validation_folds: int = 5
    

@dataclass
class SystemConfig:
    """Main system configuration."""
    # Paths
    base_path: Path = field(default_factory=lambda: Path("/root/workspace"))
    data_path: Path = field(default_factory=lambda: Path("/root/workspace/data"))
    models_path: Path = field(default_factory=lambda: Path("/root/workspace/data/models"))
    results_path: Path = field(default_factory=lambda: Path("/root/workspace/results"))
    
    # Configuration objects
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=lambda: {
        'RandomForest': ModelConfig(
            name='RandomForest_Optimized',
            params={
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        ),
        'XGBoost': ModelConfig(
            name='XGBoost_Optimized', 
            params={
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'random_state': 42,
                'n_jobs': -1
            }
        ),
        'GradientBoosting': ModelConfig(
            name='GradientBoosting_Optimized',
            params={
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'random_state': 42
            }
        ),
        'AdaBoost': ModelConfig(
            name='AdaBoost_Optimized',
            params={
                'n_estimators': 200,
                'learning_rate': 0.05,
                'random_state': 42
            }
        ),
        'NeuralNetwork': ModelConfig(
            name='NeuralNetwork_Optimized',
            params={
                'hidden_layers': [512, 256, 128, 64],
                'dropout_rates': [0.4, 0.3, 0.2, 0.2],
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 200,
                'patience': 20
            }
        )
    })
    
    # Performance targets (시장 예측 현실적 기준)
    target_accuracy: float = 0.65  # S&P500 예측의 현실적 타겟
    target_f1: float = 0.55  # 금융 시장 예측의 합리적 F1 스코어
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.data_path, self.models_path, self.results_path]:
            path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
CONFIG = SystemConfig()