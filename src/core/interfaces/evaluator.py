"""
Abstract base class for model evaluation components.

This module defines the contract for evaluating prediction models,
ensuring consistent evaluation metrics across different prediction tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np


@dataclass
class EvaluationResult:
    """
    Standardized result structure for model evaluation.

    Attributes:
        metrics: Dictionary of evaluation metrics
        predictions: Model predictions used in evaluation
        actuals: Actual values used in evaluation
        metadata: Additional evaluation information
        model_name: Name of the evaluated model
        evaluation_method: Method used for evaluation
    """
    metrics: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    metadata: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    evaluation_method: Optional[str] = None


class BaseEvaluator(ABC):
    """
    Abstract base class for all model evaluation components.

    This class defines the interface for evaluating prediction models,
    supporting different types of metrics for regression and classification tasks.

    Example:
        >>> class VolatilityEvaluator(BaseEvaluator):
        ...     def evaluate(self, predictions, actuals):
        ...         metrics = self._calculate_volatility_metrics(predictions, actuals)
        ...         return EvaluationResult(metrics=metrics, predictions=predictions, actuals=actuals)
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the evaluator.

        Args:
            name: Human-readable name for this evaluator
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs

    @abstractmethod
    def evaluate(
        self,
        predictions: Union[np.ndarray, pd.Series],
        actuals: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate model predictions against actual values.

        Args:
            predictions: Model predictions
            actuals: Actual/true values
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult containing metrics and metadata

        Raises:
            ValueError: If predictions and actuals don't match in length
        """
        pass

    def cross_validate(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv_folds: int = 5,
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Perform cross-validation evaluation.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target values
            cv_folds: Number of cross-validation folds
            **kwargs: Additional parameters

        Returns:
            List of EvaluationResult for each fold
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Prepare data for this fold
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

            # Train and predict
            model.fit(X_train, y_train)
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
                if hasattr(predictions, 'predictions'):  # PredictionResult object
                    predictions = predictions.predictions
            else:
                raise ValueError("Model must have a predict method")

            # Evaluate this fold
            result = self.evaluate(predictions, y_test, fold=fold, **kwargs)
            result.metadata = result.metadata or {}
            result.metadata['fold'] = fold
            results.append(result)

        return results

    def compare_models(
        self,
        model_results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Compare evaluation results across multiple models.

        Args:
            model_results: Dictionary mapping model names to their evaluation results

        Returns:
            Dictionary containing comparison metrics and rankings
        """
        comparison = {
            'model_rankings': {},
            'metric_comparison': {},
            'best_model': None,
            'performance_summary': {}
        }

        # Extract metrics for comparison
        all_metrics = set()
        for result in model_results.values():
            all_metrics.update(result.metrics.keys())

        # Compare each metric
        for metric in all_metrics:
            metric_values = {}
            for model_name, result in model_results.items():
                if metric in result.metrics:
                    metric_values[model_name] = result.metrics[metric]

            if metric_values:
                # Sort by metric value (higher is better for most metrics)
                if metric.lower() in ['mse', 'mae', 'rmse', 'loss']:
                    # Lower is better for error metrics
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
                else:
                    # Higher is better for other metrics
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)

                comparison['metric_comparison'][metric] = {
                    'best_model': sorted_models[0][0],
                    'best_value': sorted_models[0][1],
                    'all_values': dict(sorted_models)
                }

        # Overall ranking (based on primary metric)
        primary_metric = self._get_primary_metric(all_metrics)
        if primary_metric and primary_metric in comparison['metric_comparison']:
            comparison['best_model'] = comparison['metric_comparison'][primary_metric]['best_model']
            comparison['model_rankings'] = comparison['metric_comparison'][primary_metric]['all_values']

        return comparison

    def _get_primary_metric(self, available_metrics: set) -> Optional[str]:
        """
        Determine the primary metric for model comparison.

        Args:
            available_metrics: Set of available metric names

        Returns:
            Name of the primary metric, or None if no suitable metric found
        """
        # Priority order for different metric types
        priority_metrics = [
            'r2', 'r2_score', 'r_squared',  # R-squared for regression
            'accuracy',  # Accuracy for classification
            'f1', 'f1_score',  # F1 score
            'auc', 'roc_auc',  # AUC scores
            'mae', 'mse', 'rmse'  # Error metrics (lower is better)
        ]

        for metric in priority_metrics:
            if metric in available_metrics:
                return metric

        return None

    def calculate_regression_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate standard regression metrics.

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            Dictionary of regression metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {}

        # Basic error metrics
        metrics['mae'] = mean_absolute_error(actuals, predictions)
        metrics['mse'] = mean_squared_error(actuals, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(actuals, predictions)

        # Additional metrics
        metrics['mape'] = np.mean(np.abs((actuals - predictions) / np.abs(actuals))) * 100
        metrics['max_error'] = np.max(np.abs(actuals - predictions))
        metrics['mean_error'] = np.mean(predictions - actuals)

        # Correlation metrics
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0

        return metrics

    def calculate_volatility_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate metrics specific to volatility prediction.

        Args:
            predictions: Predicted volatility values
            actuals: Actual volatility values

        Returns:
            Dictionary of volatility-specific metrics
        """
        # Start with standard regression metrics
        metrics = self.calculate_regression_metrics(predictions, actuals)

        # Add volatility-specific metrics
        # Hit rate for volatility direction (increasing vs decreasing)
        if len(predictions) > 1:
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            direction_accuracy = np.mean(pred_direction == actual_direction)
            metrics['direction_accuracy'] = direction_accuracy

        # Volatility clustering detection
        actual_high_vol = actuals > np.percentile(actuals, 75)
        pred_high_vol = predictions > np.percentile(predictions, 75)
        high_vol_precision = np.sum(actual_high_vol & pred_high_vol) / np.sum(pred_high_vol) if np.sum(pred_high_vol) > 0 else 0
        high_vol_recall = np.sum(actual_high_vol & pred_high_vol) / np.sum(actual_high_vol) if np.sum(actual_high_vol) > 0 else 0

        metrics['high_vol_precision'] = high_vol_precision
        metrics['high_vol_recall'] = high_vol_recall
        metrics['high_vol_f1'] = 2 * (high_vol_precision * high_vol_recall) / (high_vol_precision + high_vol_recall) if (high_vol_precision + high_vol_recall) > 0 else 0

        return metrics

    def validate_inputs(
        self,
        predictions: Union[np.ndarray, pd.Series],
        actuals: Union[np.ndarray, pd.Series]
    ) -> None:
        """
        Validate evaluation inputs.

        Args:
            predictions: Model predictions
            actuals: Actual values

        Raises:
            ValueError: If inputs are invalid
        """
        if predictions is None or actuals is None:
            raise ValueError("Predictions and actuals cannot be None")

        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have the same length")

        if len(predictions) == 0:
            raise ValueError("Cannot evaluate with empty arrays")

        # Convert to numpy arrays for validation
        pred_array = np.asarray(predictions)
        actual_array = np.asarray(actuals)

        if np.any(np.isnan(pred_array)) or np.any(np.isnan(actual_array)):
            raise ValueError("Predictions and actuals cannot contain NaN values")

    def generate_report(
        self,
        evaluation_result: EvaluationResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            evaluation_result: Evaluation result to report on
            output_path: Optional path to save the report

        Returns:
            Formatted report string
        """
        report_lines = [
            f"Evaluation Report: {evaluation_result.model_name or 'Unknown Model'}",
            "=" * 60,
            f"Evaluation Method: {evaluation_result.evaluation_method or 'Standard'}",
            f"Number of Predictions: {len(evaluation_result.predictions)}",
            "",
            "Metrics:",
            "-" * 20
        ]

        # Format metrics
        for metric_name, value in evaluation_result.metrics.items():
            formatted_value = f"{value:.6f}" if isinstance(value, float) else str(value)
            report_lines.append(f"{metric_name:20s}: {formatted_value}")

        # Add metadata if available
        if evaluation_result.metadata:
            report_lines.extend([
                "",
                "Additional Information:",
                "-" * 30
            ])
            for key, value in evaluation_result.metadata.items():
                report_lines.append(f"{key:20s}: {value}")

        report = "\n".join(report_lines)

        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def __repr__(self) -> str:
        """String representation of the evaluator."""
        return f"{self.__class__.__name__}(name='{self.name}')"