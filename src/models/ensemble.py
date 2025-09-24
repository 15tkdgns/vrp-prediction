"""
Advanced ensemble methods for combining multiple models.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler

from .factory import BaseModel, model_factory
from ..core.config import CONFIG
from ..core.logger import logger
from ..core.exceptions import ModelTrainingError


class EnsembleMethod:
    """Base class for ensemble methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.models = {}
        self.weights = {}
        self.is_fitted = False
    
    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'EnsembleMethod':
        """Train all models in the ensemble."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        raise NotImplementedError


class VotingEnsemble(EnsembleMethod):
    """Voting ensemble implementation."""
    
    def __init__(self, voting_type: str = 'soft'):
        super().__init__(f"VotingEnsemble_{voting_type}")
        self.voting_type = voting_type
        self.voting_classifier = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None, **kwargs) -> 'VotingEnsemble':
        """Train all models and create voting classifier."""
        logger.section(f"Training {self.name}", "🗳️")
        
        if not self.models:
            raise ModelTrainingError("No models added to ensemble")
        
        trained_models = []
        model_performances = {}
        
        # Train individual models
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                if hasattr(model, 'fit'):
                    if 'NeuralNetwork' in name and validation_data is not None:
                        model.fit(X, y, validation_data=validation_data, **kwargs)
                    else:
                        model.fit(X, y, **kwargs)
                else:
                    # Handle sklearn models wrapped in our BaseModel
                    model.create_model()
                    model.fit(X, y, **kwargs)
                
                # Evaluate on validation data if provided
                if validation_data is not None:
                    X_val, y_val = validation_data
                    val_pred = model.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                    val_f1 = f1_score(y_val, val_pred, average='binary')
                    
                    model_performances[name] = val_acc
                    
                    logger.performance(f"{name} Validation Accuracy", val_acc)
                    logger.performance(f"{name} Validation F1", val_f1)
                
                # Prepare for voting classifier
                if hasattr(model, 'model'):
                    trained_models.append((name, model.model))
                else:
                    trained_models.append((name, model))
                    
            except Exception as e:
                logger.error(f"{name} training failed: {str(e)}")
                continue
        
        if len(trained_models) < 2:
            raise ModelTrainingError("Need at least 2 trained models for ensemble")
        
        # Create voting classifier
        logger.info(f"Creating voting classifier with {len(trained_models)} models")
        
        try:
            self.voting_classifier = VotingClassifier(
                estimators=trained_models,
                voting=self.voting_type
            )
            self.voting_classifier.fit(X, y)
            
            # Evaluate ensemble if validation data provided
            if validation_data is not None:
                X_val, y_val = validation_data
                ensemble_pred = self.voting_classifier.predict(X_val)
                ensemble_acc = accuracy_score(y_val, ensemble_pred)
                ensemble_f1 = f1_score(y_val, ensemble_pred, average='binary')
                
                logger.performance("Ensemble Validation Accuracy", ensemble_acc)
                logger.performance("Ensemble Validation F1", ensemble_f1)
                
                model_performances['Ensemble'] = ensemble_acc
            
            self.is_fitted = True
            logger.success(f"{self.name} training completed")
            
            return model_performances
            
        except Exception as e:
            raise ModelTrainingError(f"Voting classifier creation failed: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ModelTrainingError(f"{self.name} is not fitted")
        
        return self.voting_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        if not self.is_fitted:
            raise ModelTrainingError(f"{self.name} is not fitted")
        
        if self.voting_type == 'soft':
            return self.voting_classifier.predict_proba(X)
        else:
            # For hard voting, return binary predictions as probabilities
            pred = self.voting_classifier.predict(X)
            return np.column_stack([1 - pred, pred])


class StackingEnsemble(EnsembleMethod):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, meta_model: BaseModel):
        super().__init__("StackingEnsemble")
        self.meta_model = meta_model
        self.base_models = {}
        self.scaler = RobustScaler()
    
    def add_base_model(self, name: str, model: BaseModel) -> None:
        """Add a base model to the stacking ensemble."""
        self.base_models[name] = model
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None, **kwargs) -> 'StackingEnsemble':
        """Train base models and meta-learner."""
        logger.section("Training Stacking Ensemble", "🥞")
        
        if not self.base_models:
            raise ModelTrainingError("No base models added to stacking ensemble")
        
        # Split data for base models and meta-learner
        from sklearn.model_selection import train_test_split
        X_base, X_meta, y_base, y_meta = train_test_split(
            X, y, test_size=0.3, random_state=CONFIG.training.random_state, stratify=y
        )
        
        logger.info(f"Base models training: {len(X_base)}, Meta model training: {len(X_meta)}")
        
        # Train base models
        base_predictions = []
        
        for name, model in self.base_models.items():
            logger.info(f"Training base model: {name}")
            
            try:
                model.fit(X_base, y_base)
                
                # Get predictions for meta-learner
                pred_proba = model.predict_proba(X_meta)
                base_predictions.append(pred_proba)
                
                # Evaluate base model
                pred = model.predict(X_meta)
                acc = accuracy_score(y_meta, pred)
                logger.performance(f"{name} Meta Accuracy", acc)
                
            except Exception as e:
                logger.error(f"Base model {name} training failed: {str(e)}")
                continue
        
        if not base_predictions:
            raise ModelTrainingError("No base models trained successfully")
        
        # Create meta features
        meta_features = np.hstack(base_predictions)
        logger.info(f"Meta features shape: {meta_features.shape}")
        
        # Scale meta features
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        
        # Train meta-learner
        logger.info("Training meta-learner")
        
        try:
            if 'NeuralNetwork' in self.meta_model.name and validation_data is not None:
                # For neural networks, use part of meta data for validation
                val_split = 0.2
                self.meta_model.fit(
                    meta_features_scaled, y_meta, 
                    validation_split=val_split
                )
            else:
                self.meta_model.fit(meta_features_scaled, y_meta)
            
            self.is_fitted = True
            logger.success("Stacking ensemble training completed")
            
        except Exception as e:
            raise ModelTrainingError(f"Meta-learner training failed: {str(e)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make stacking ensemble predictions."""
        if not self.is_fitted:
            raise ModelTrainingError("Stacking ensemble is not fitted")
        
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            try:
                pred_proba = model.predict_proba(X)
                base_predictions.append(pred_proba)
            except Exception as e:
                logger.warning(f"Base model {name} prediction failed: {str(e)}")
                continue
        
        if not base_predictions:
            raise ModelTrainingError("No base model predictions available")
        
        # Create meta features
        meta_features = np.hstack(base_predictions)
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Meta-learner prediction
        return self.meta_model.predict(meta_features_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get stacking ensemble prediction probabilities."""
        if not self.is_fitted:
            raise ModelTrainingError("Stacking ensemble is not fitted")
        
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            try:
                pred_proba = model.predict_proba(X)
                base_predictions.append(pred_proba)
            except:
                continue
        
        if not base_predictions:
            raise ModelTrainingError("No base model predictions available")
        
        # Create meta features
        meta_features = np.hstack(base_predictions)
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Meta-learner probability prediction
        return self.meta_model.predict_proba(meta_features_scaled)


class AdaptiveEnsemble(EnsembleMethod):
    """Adaptive ensemble that dynamically weights models based on performance."""
    
    def __init__(self, adaptation_window: int = 50):
        super().__init__("AdaptiveEnsemble")
        self.adaptation_window = adaptation_window
        self.performance_history = {}
        self.dynamic_weights = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None, **kwargs) -> 'AdaptiveEnsemble':
        """Train models and initialize adaptive weighting."""
        logger.section("Training Adaptive Ensemble", "🔄")
        
        # Train all models
        for name, model in self.models.items():
            logger.info(f"Training {name}")
            
            try:
                model.fit(X, y, **kwargs)
                
                # Initialize performance history
                self.performance_history[name] = []
                self.dynamic_weights[name] = 1.0
                
            except Exception as e:
                logger.error(f"{name} training failed: {str(e)}")
                continue
        
        self.is_fitted = True
        logger.success("Adaptive ensemble training completed")
        
        return self
    
    def _update_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update model weights based on recent performance."""
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                accuracy = accuracy_score(y, pred)
                
                # Update performance history
                self.performance_history[name].append(accuracy)
                
                # Keep only recent performance
                if len(self.performance_history[name]) > self.adaptation_window:
                    self.performance_history[name] = self.performance_history[name][-self.adaptation_window:]
                
                # Calculate dynamic weight (recent average performance)
                recent_performance = np.mean(self.performance_history[name])
                self.dynamic_weights[name] = max(recent_performance, 0.1)  # Minimum weight of 0.1
                
            except Exception as e:
                logger.warning(f"Weight update failed for {name}: {str(e)}")
                self.dynamic_weights[name] = 0.1
    
    def predict_proba(self, X: np.ndarray, update_weights: bool = False) -> np.ndarray:
        """Get adaptive ensemble prediction probabilities."""
        if not self.is_fitted:
            raise ModelTrainingError("Adaptive ensemble is not fitted")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(X)
                predictions.append(pred_proba)
                weights.append(self.dynamic_weights[name])
            except:
                continue
        
        if not predictions:
            raise ModelTrainingError("No model predictions available")
        
        # Weighted average of predictions
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        return weighted_pred
    
    def predict(self, X: np.ndarray, update_weights: bool = False) -> np.ndarray:
        """Make adaptive ensemble predictions."""
        pred_proba = self.predict_proba(X, update_weights)
        return (pred_proba[:, 1] > 0.5).astype(int)


class EnsembleManager:
    """Manager for creating and comparing different ensemble methods."""
    
    def __init__(self):
        self.ensembles = {}
        self.performance_results = {}
    
    def create_voting_ensemble(self, model_types: List[str], voting_type: str = 'soft') -> VotingEnsemble:
        """Create a voting ensemble with specified models."""
        ensemble = VotingEnsemble(voting_type)
        
        for model_type in model_types:
            model = model_factory.create_model(model_type)
            ensemble.add_model(model_type, model)
        
        ensemble_name = f"Voting_{voting_type}_{len(model_types)}models"
        self.ensembles[ensemble_name] = ensemble
        
        return ensemble
    
    def create_stacking_ensemble(self, base_model_types: List[str], meta_model_type: str) -> StackingEnsemble:
        """Create a stacking ensemble."""
        meta_model = model_factory.create_model(meta_model_type)
        ensemble = StackingEnsemble(meta_model)
        
        for model_type in base_model_types:
            model = model_factory.create_model(model_type)
            ensemble.add_base_model(model_type, model)
        
        ensemble_name = f"Stacking_{len(base_model_types)}base_{meta_model_type}meta"
        self.ensembles[ensemble_name] = ensemble
        
        return ensemble
    
    def compare_ensembles(self, X: np.ndarray, y: np.ndarray, 
                         validation_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compare performance of different ensembles."""
        logger.section("Ensemble Comparison", "🏆")
        
        X_val, y_val = validation_data
        results = {}
        
        for name, ensemble in self.ensembles.items():
            logger.info(f"Evaluating {name}")
            
            try:
                # Train ensemble
                if hasattr(ensemble, 'fit'):
                    ensemble.fit(X, y, validation_data=validation_data)
                
                # Make predictions
                pred = ensemble.predict(X_val)
                accuracy = accuracy_score(y_val, pred)
                f1 = f1_score(y_val, pred, average='binary')
                
                results[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1
                }
                
                logger.performance(f"{name} Accuracy", accuracy)
                logger.performance(f"{name} F1-Score", f1)
                
            except Exception as e:
                logger.error(f"{name} evaluation failed: {str(e)}")
                results[name] = {'accuracy': 0.0, 'f1_score': 0.0}
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        logger.info("Ensemble Performance Ranking:")
        for i, (name, metrics) in enumerate(sorted_results, 1):
            logger.info(f"  {i}. {name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        self.performance_results = results
        return results
    
    def get_best_ensemble(self) -> Tuple[str, EnsembleMethod]:
        """Get the best performing ensemble."""
        if not self.performance_results:
            raise ValueError("No ensemble results available. Run compare_ensembles first.")
        
        best_name = max(self.performance_results.items(), key=lambda x: x[1]['accuracy'])[0]
        return best_name, self.ensembles[best_name]