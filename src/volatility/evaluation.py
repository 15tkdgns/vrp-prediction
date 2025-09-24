"""
Volatility prediction evaluation.

This module provides evaluation methods specifically designed for volatility prediction,
including volatility-specific metrics and risk management assessments.
"""

from __future__ import annotations

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ..core.interfaces import BaseEvaluator, EvaluationResult
from ..core.types import FloatArray


class VolatilityEvaluator(BaseEvaluator):
    """
    Evaluator specialized for volatility prediction.

    This class provides comprehensive evaluation metrics for volatility prediction models,
    including standard regression metrics and volatility-specific measures.

    Example:
        >>> evaluator = VolatilityEvaluator()
        >>> result = evaluator.evaluate(predictions, actual_volatility)
        >>> print(f"R² Score: {result.metrics['r2']:.4f}")
        >>> print(f"Volatility Accuracy: {result.metrics['direction_accuracy']:.4f}")
    """

    def __init__(self, **kwargs):
        """Initialize the volatility evaluator."""
        super().__init__(name="Volatility Evaluator", **kwargs)

    def evaluate(
        self,
        predictions: Union[np.ndarray, pd.Series],
        actuals: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate volatility predictions against actual values.

        Args:
            predictions: Predicted volatility values
            actuals: Actual volatility values
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult containing comprehensive volatility metrics
        """
        self.validate_inputs(predictions, actuals)

        # Convert to numpy arrays
        pred_array = np.asarray(predictions)
        actual_array = np.asarray(actuals)

        # Calculate comprehensive metrics
        metrics = {}

        # Standard regression metrics
        metrics.update(self.calculate_regression_metrics(pred_array, actual_array))

        # Volatility-specific metrics
        metrics.update(self.calculate_volatility_metrics(pred_array, actual_array))

        # Risk management metrics
        metrics.update(self._calculate_risk_metrics(pred_array, actual_array))

        # Prepare metadata
        metadata = {
            'evaluation_type': 'volatility_prediction',
            'n_observations': len(pred_array),
            'prediction_range': {
                'min': float(pred_array.min()),
                'max': float(pred_array.max()),
                'mean': float(pred_array.mean())
            },
            'actual_range': {
                'min': float(actual_array.min()),
                'max': float(actual_array.max()),
                'mean': float(actual_array.mean())
            }
        }

        return EvaluationResult(
            metrics=metrics,
            predictions=pred_array,
            actuals=actual_array,
            metadata=metadata,
            evaluation_method="comprehensive_volatility"
        )

    def _calculate_risk_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate risk management specific metrics."""
        metrics = {}

        # Volatility forecasting accuracy for different percentiles
        for percentile in [75, 90, 95]:
            threshold = np.percentile(actuals, percentile)
            high_vol_actual = actuals >= threshold
            high_vol_pred = predictions >= threshold

            if np.sum(high_vol_actual) > 0 and np.sum(high_vol_pred) > 0:
                precision = np.sum(high_vol_actual & high_vol_pred) / np.sum(high_vol_pred)
                recall = np.sum(high_vol_actual & high_vol_pred) / np.sum(high_vol_actual)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                metrics[f'high_vol_{percentile}p_precision'] = precision
                metrics[f'high_vol_{percentile}p_recall'] = recall
                metrics[f'high_vol_{percentile}p_f1'] = f1

        # Volatility regime accuracy
        metrics.update(self._calculate_regime_accuracy(predictions, actuals))

        # Prediction bias in different volatility environments
        metrics.update(self._calculate_conditional_bias(predictions, actuals))

        return metrics

    def _calculate_regime_accuracy(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate accuracy in different volatility regimes."""
        metrics = {}

        # Define volatility regimes based on percentiles
        low_threshold = np.percentile(actuals, 33)
        high_threshold = np.percentile(actuals, 67)

        # Low volatility regime
        low_vol_mask = actuals <= low_threshold
        if np.sum(low_vol_mask) > 0:
            low_vol_r2 = r2_score(actuals[low_vol_mask], predictions[low_vol_mask])
            low_vol_mae = mean_absolute_error(actuals[low_vol_mask], predictions[low_vol_mask])
            metrics['low_vol_r2'] = max(low_vol_r2, -1.0)  # Cap at -1 for extreme cases
            metrics['low_vol_mae'] = low_vol_mae

        # Medium volatility regime
        med_vol_mask = (actuals > low_threshold) & (actuals <= high_threshold)
        if np.sum(med_vol_mask) > 0:
            med_vol_r2 = r2_score(actuals[med_vol_mask], predictions[med_vol_mask])
            med_vol_mae = mean_absolute_error(actuals[med_vol_mask], predictions[med_vol_mask])
            metrics['med_vol_r2'] = max(med_vol_r2, -1.0)
            metrics['med_vol_mae'] = med_vol_mae

        # High volatility regime
        high_vol_mask = actuals > high_threshold
        if np.sum(high_vol_mask) > 0:
            high_vol_r2 = r2_score(actuals[high_vol_mask], predictions[high_vol_mask])
            high_vol_mae = mean_absolute_error(actuals[high_vol_mask], predictions[high_vol_mask])
            metrics['high_vol_r2'] = max(high_vol_r2, -1.0)
            metrics['high_vol_mae'] = high_vol_mae

        return metrics

    def _calculate_conditional_bias(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate prediction bias in different conditions."""
        metrics = {}

        # Overall bias
        bias = np.mean(predictions - actuals)
        metrics['overall_bias'] = bias

        # Bias in high vs low volatility periods
        median_vol = np.median(actuals)
        high_vol_mask = actuals > median_vol
        low_vol_mask = actuals <= median_vol

        if np.sum(high_vol_mask) > 0:
            high_vol_bias = np.mean(predictions[high_vol_mask] - actuals[high_vol_mask])
            metrics['high_vol_bias'] = high_vol_bias

        if np.sum(low_vol_mask) > 0:
            low_vol_bias = np.mean(predictions[low_vol_mask] - actuals[low_vol_mask])
            metrics['low_vol_bias'] = low_vol_bias

        # Relative bias (percentage)
        mean_actual = np.mean(actuals)
        if mean_actual > 0:
            metrics['relative_bias'] = (bias / mean_actual) * 100

        return metrics

    def evaluate_trading_performance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate how well volatility predictions translate to trading performance.

        Args:
            predictions: Predicted volatility
            actuals: Actual volatility
            returns: Asset returns (if available)

        Returns:
            Dictionary of trading-relevant metrics
        """
        metrics = {}

        # Volatility timing accuracy
        pred_changes = np.diff(predictions)
        actual_changes = np.diff(actuals)

        # Direction accuracy for volatility changes
        if len(pred_changes) > 0:
            direction_accuracy = np.mean(np.sign(pred_changes) == np.sign(actual_changes))
            metrics['vol_change_direction_accuracy'] = direction_accuracy

        # Volatility forecasting for risk management
        # How well do we predict high volatility periods?
        vol_threshold = np.percentile(actuals, 80)
        high_vol_periods = actuals > vol_threshold
        predicted_high_vol = predictions > vol_threshold

        if np.sum(high_vol_periods) > 0:
            vol_timing_precision = np.sum(high_vol_periods & predicted_high_vol) / np.sum(predicted_high_vol) if np.sum(predicted_high_vol) > 0 else 0
            vol_timing_recall = np.sum(high_vol_periods & predicted_high_vol) / np.sum(high_vol_periods)

            metrics['vol_timing_precision'] = vol_timing_precision
            metrics['vol_timing_recall'] = vol_timing_recall
            metrics['vol_timing_f1'] = 2 * (vol_timing_precision * vol_timing_recall) / (vol_timing_precision + vol_timing_recall) if (vol_timing_precision + vol_timing_recall) > 0 else 0

        # If returns are provided, calculate portfolio-relevant metrics
        if returns is not None:
            returns_array = np.asarray(returns)
            if len(returns_array) == len(predictions):
                metrics.update(self._calculate_portfolio_metrics(predictions, actuals, returns_array))

        return metrics

    def _calculate_portfolio_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate portfolio-relevant metrics."""
        metrics = {}

        # Correlation between predicted volatility and subsequent returns
        if len(predictions) > 1:
            # Shift returns to align with volatility predictions
            future_returns = np.roll(returns, -1)[:-1]
            pred_vol = predictions[:-1]

            vol_return_corr = np.corrcoef(pred_vol, np.abs(future_returns))[0, 1]
            if not np.isnan(vol_return_corr):
                metrics['vol_return_correlation'] = vol_return_corr

        # Volatility clustering detection
        # How well do we predict volatility persistence?
        high_vol_persistence = self._calculate_volatility_persistence(actuals)
        pred_vol_persistence = self._calculate_volatility_persistence(predictions)

        metrics['actual_vol_persistence'] = high_vol_persistence
        metrics['predicted_vol_persistence'] = pred_vol_persistence
        metrics['persistence_difference'] = abs(high_vol_persistence - pred_vol_persistence)

        return metrics

    def _calculate_volatility_persistence(self, volatility: np.ndarray) -> float:
        """Calculate volatility clustering/persistence measure."""
        high_vol_threshold = np.percentile(volatility, 75)
        high_vol_periods = volatility > high_vol_threshold

        # Count consecutive high volatility periods
        consecutive_periods = []
        current_streak = 0

        for is_high_vol in high_vol_periods:
            if is_high_vol:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive_periods.append(current_streak)
                current_streak = 0

        if current_streak > 0:
            consecutive_periods.append(current_streak)

        # Return average length of high volatility streaks
        return np.mean(consecutive_periods) if consecutive_periods else 0.0