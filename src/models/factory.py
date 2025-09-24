"""
Model factory for creating and managing different ML models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..core.config import CONFIG, ModelConfig
from ..core.logger import logger
from ..core.exceptions import ModelTrainingError, ConfigurationError


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: str, config: ModelConfig):
        self.name = name
        self.config = config
        self.model = None
        self.is_fitted = False
        self.training_history = {}
    
    @abstractmethod
    def create_model(self, **kwargs) -> Any:
        """Create the underlying model."""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances if available."""
        return None
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        raise NotImplementedError("Save method not implemented")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        raise NotImplementedError("Load method not implemented")


class SklearnModel(BaseModel):
    """Wrapper for sklearn-compatible models."""
    
    def __init__(self, name: str, config: ModelConfig, model_class):
        super().__init__(name, config)
        self.model_class = model_class
    
    def create_model(self, **kwargs) -> Any:
        """Create sklearn model."""
        params = {**self.config.params, **kwargs}
        self.model = self.model_class(**params)
        return self.model
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SklearnModel':
        """Train sklearn model."""
        if self.model is None:
            self.create_model()
        
        logger.info(f"Training {self.name}...", "🔧")
        
        try:
            # Handle class weight for imbalanced data
            if hasattr(self.model, 'set_params') and 'scale_pos_weight' in self.model.get_params():
                # XGBoost specific handling
                if len(np.unique(y)) == 2:
                    pos_weight = len(y[y == 0]) / len(y[y == 1])
                    self.model.set_params(scale_pos_weight=pos_weight)
            
            self.model.fit(X, y, **kwargs)
            self.is_fitted = True
            
            logger.success(f"{self.name} training completed")
            
        except Exception as e:
            raise ModelTrainingError(f"{self.name} training failed: {str(e)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelTrainingError(f"{self.name} is not fitted")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ModelTrainingError(f"{self.name} is not fitted")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
        return None


class NeuralNetworkModel(BaseModel):
    """Deep learning model using TensorFlow/Keras."""
    
    def __init__(self, name: str, config: ModelConfig):
        super().__init__(name, config)
        self.scaler = None
    
    def create_model(self, input_dim: int, **kwargs) -> keras.Model:
        """Create neural network model."""
        params = {**self.config.params, **kwargs}
        
        # Build architecture
        inputs = layers.Input(shape=(input_dim,))
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(params['hidden_layers']):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            dropout_rate = params['dropout_rates'][i] if i < len(params['dropout_rates']) else 0.3
            x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data=None, **kwargs) -> 'NeuralNetworkModel':
        """Train neural network."""
        if self.model is None:
            self.create_model(input_dim=X.shape[1])
        
        logger.info(f"Training {self.name}...", "🤖")
        
        params = self.config.params
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=params.get('patience', 20),
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=params.get('patience', 20) // 2,
                min_lr=1e-6
            )
        ]
        
        # Class weights for imbalanced data
        if len(np.unique(y)) == 2:
            class_weight = {
                0: len(y) / (2 * np.sum(y == 0)),
                1: len(y) / (2 * np.sum(y == 1))
            }
        else:
            class_weight = None
        
        try:
            history = self.model.fit(
                X, y,
                validation_data=validation_data,
                epochs=params.get('epochs', 200),
                batch_size=params.get('batch_size', 64),
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=0,
                **kwargs
            )
            
            self.is_fitted = True
            self.training_history = history.history
            
            best_val_acc = max(history.history.get('val_accuracy', [0]))
            logger.success(f"{self.name} training completed (best val_acc: {best_val_acc:.4f})")
            
        except Exception as e:
            raise ModelTrainingError(f"{self.name} training failed: {str(e)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ModelTrainingError(f"{self.name} is not fitted")
        
        pred_proba = self.model.predict(X, verbose=0)
        return (pred_proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ModelTrainingError(f"{self.name} is not fitted")
        
        proba = self.model.predict(X, verbose=0)
        # Convert to 2D array for sklearn compatibility
        return np.column_stack([1 - proba.flatten(), proba.flatten()])


class ModelFactory:
    """Factory for creating and managing models."""
    
    def __init__(self):
        self.model_registry = {
            'RandomForest': (SklearnModel, RandomForestClassifier),
            'XGBoost': (SklearnModel, xgb.XGBClassifier),
            'GradientBoosting': (SklearnModel, GradientBoostingClassifier),
            'AdaBoost': (SklearnModel, AdaBoostClassifier),
            'LogisticRegression': (SklearnModel, LogisticRegression),
            'SVM': (SklearnModel, SVC),
            'NeuralNetwork': (NeuralNetworkModel, None)
        }
    
    def create_model(self, model_type: str, config: Optional[ModelConfig] = None) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            config: Model configuration (uses default if None)
            
        Returns:
            Model instance
        """
        if model_type not in self.model_registry:
            available = list(self.model_registry.keys())
            raise ConfigurationError(f"Unknown model type: {model_type}. Available: {available}")
        
        # Get configuration
        if config is None:
            if model_type not in CONFIG.models:
                raise ConfigurationError(f"No configuration found for model: {model_type}")
            config = CONFIG.models[model_type]
        
        # Create model
        model_class, sklearn_class = self.model_registry[model_type]
        
        if model_class == SklearnModel:
            # Add probability=True for SVM
            if sklearn_class == SVC and 'probability' not in config.params:
                config.params['probability'] = True
            
            model = model_class(config.name, config, sklearn_class)
        else:
            model = model_class(config.name, config)
        
        logger.info(f"Created {model_type} model: {config.name}")
        
        return model
    
    def create_all_models(self, model_types: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """
        Create multiple models.
        
        Args:
            model_types: List of model types to create (creates all if None)
            
        Returns:
            Dictionary of model instances
        """
        if model_types is None:
            model_types = [name for name, config in CONFIG.models.items() if config.enabled]
        
        models = {}
        for model_type in model_types:
            try:
                models[model_type] = self.create_model(model_type)
            except Exception as e:
                logger.error(f"Failed to create {model_type}: {str(e)}")
        
        logger.success(f"Created {len(models)} models: {list(models.keys())}")
        
        return models
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.model_registry.keys())
    
    def register_model(self, name: str, model_class, sklearn_class=None) -> None:
        """Register a new model type."""
        self.model_registry[name] = (model_class, sklearn_class)
        logger.info(f"Registered new model type: {name}")


# Global factory instance
model_factory = ModelFactory()