"""
Advanced Statistical Validation Framework.

This module implements state-of-the-art statistical validation methods for financial ML,
including Combinatorial Purged Cross-Validation (CPCV) and Deflated Sharpe Ratio (DSR).

These methods address the critical problem of selection bias and overfitting in financial
time series modeling, providing robust statistical evidence for model performance.

References:
- Bailey & López de Prado (2014): "The Deflated Sharpe Ratio"
- López de Prado (2018): "Advances in Financial Machine Learning"
- Bailey et al. (2017): "The Probability of Backtest Overfitting"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from itertools import combinations
import warnings
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import math

from ..core.types import FloatArray, DataFrame, ModelType
from ..core.exceptions import ModelValidationError


@dataclass
class CPCVResult:
    """
    Result structure for Combinatorial Purged Cross-Validation.

    Attributes:
        performance_distribution: Array of performance scores across all path combinations
        mean_performance: Mean performance across all paths
        std_performance: Standard deviation of performance
        confidence_intervals: Confidence intervals for performance
        worst_case_performance: Worst case scenario performance
        probability_positive: Probability that performance > 0
        n_combinations: Number of path combinations tested
        methodology: Description of validation methodology used
    """
    performance_distribution: FloatArray
    mean_performance: float
    std_performance: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    worst_case_performance: float
    probability_positive: float
    n_combinations: int
    methodology: str


@dataclass
class DSRResult:
    """
    Result structure for Deflated Sharpe Ratio analysis.

    Attributes:
        deflated_sharpe: Deflated Sharpe ratio accounting for selection bias
        original_sharpe: Original (unadjusted) Sharpe ratio
        probability_skill: Probability that observed performance represents skill vs luck
        selection_bias_adjustment: Adjustment factor for multiple testing
        n_trials: Number of trials/models tested
        variance_inflation: Variance inflation due to multiple testing
        statistical_significance: Whether result is statistically significant
    """
    deflated_sharpe: float
    original_sharpe: float
    probability_skill: float
    selection_bias_adjustment: float
    n_trials: int
    variance_inflation: float
    statistical_significance: bool


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation implementation.

    This class implements CPCV to generate hundreds of possible backtest paths,
    creating a distribution of performance metrics rather than a single point estimate.
    This provides much more robust statistical inference about model performance.

    The key innovation is testing all possible combinations of time-based splits,
    with proper purging to prevent data leakage, creating a comprehensive view
    of how the model performs across different market regimes and time periods.
    """

    def __init__(
        self,
        n_splits: int = 10,
        min_train_size: int = 252,  # 1 year minimum
        purge_gap: int = 5,  # 5 days gap to prevent leakage
        embargo_pct: float = 0.01,  # 1% embargo
        max_combinations: int = 1000  # Limit for computational efficiency
    ):
        """
        Initialize CPCV validator.

        Args:
            n_splits: Number of base splits to generate
            min_train_size: Minimum training set size
            purge_gap: Number of days to purge between train/test
            embargo_pct: Percentage of data to embargo after test set
            max_combinations: Maximum number of combinations to test
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.max_combinations = max_combinations

    def generate_time_splits(
        self,
        data_length: int,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate base time-based splits with proper purging.

        Args:
            data_length: Length of the dataset
            timestamps: Optional timestamps for temporal splits

        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []

        # Calculate split sizes
        test_size = max(50, data_length // (self.n_splits + 1))
        embargo_size = int(data_length * self.embargo_pct)

        for i in range(self.n_splits):
            # Calculate test period
            test_start = self.min_train_size + i * test_size
            test_end = min(test_start + test_size, data_length - embargo_size)

            if test_start >= test_end:
                break

            # Training period (everything before test, with purge gap)
            train_end = test_start - self.purge_gap
            train_start = max(0, train_end - self.min_train_size * 2)

            if train_start >= train_end or train_end <= self.min_train_size:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            # Ensure minimum sizes
            if len(train_indices) >= self.min_train_size and len(test_indices) >= 20:
                splits.append((train_indices, test_indices))

        return splits

    def generate_combinations(
        self,
        base_splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[List[int]]:
        """
        Generate all valid combinations of base splits.

        Args:
            base_splits: Base time splits

        Returns:
            List of split combination indices
        """
        n_base_splits = len(base_splits)

        if n_base_splits < 3:
            warnings.warn("Too few base splits for reliable CPCV")
            return [[i] for i in range(n_base_splits)]

        all_combinations = []

        # Generate combinations of different sizes
        for combo_size in range(1, min(n_base_splits + 1, 8)):  # Limit combo size
            for combo in combinations(range(n_base_splits), combo_size):
                # Check temporal consistency
                if self._is_valid_combination(combo, base_splits):
                    all_combinations.append(list(combo))

                    if len(all_combinations) >= self.max_combinations:
                        break

            if len(all_combinations) >= self.max_combinations:
                break

        return all_combinations

    def _is_valid_combination(
        self,
        combination: Tuple[int, ...],
        base_splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """
        Check if a combination of splits is temporally valid.

        Args:
            combination: Tuple of split indices
            base_splits: Base splits

        Returns:
            True if combination is valid
        """
        if len(combination) == 1:
            return True

        # Check for temporal overlap in test periods
        test_periods = []
        for split_idx in combination:
            _, test_indices = base_splits[split_idx]
            test_periods.append((test_indices[0], test_indices[-1]))

        # Sort by start time
        test_periods.sort()

        # Check for overlaps
        for i in range(len(test_periods) - 1):
            if test_periods[i][1] >= test_periods[i + 1][0]:
                return False

        return True

    def evaluate_model_combination(
        self,
        model_class: type,
        model_params: Dict[str, Any],
        X: DataFrame,
        y: pd.Series,
        combination: List[int],
        base_splits: List[Tuple[np.ndarray, np.ndarray]],
        metric_func: callable = None
    ) -> float:
        """
        Evaluate model performance for a specific combination of splits.

        Args:
            model_class: Model class to instantiate
            model_params: Model parameters
            X: Feature matrix
            y: Target vector
            combination: List of split indices to use
            base_splits: Base splits
            metric_func: Metric function (default: R²)

        Returns:
            Performance score for this combination
        """
        if metric_func is None:
            metric_func = lambda y_true, y_pred: stats.pearsonr(y_true, y_pred)[0] ** 2

        all_predictions = []
        all_actuals = []

        for split_idx in combination:
            train_indices, test_indices = base_splits[split_idx]

            # Prepare data
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]

            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)

            # Predict
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
                if hasattr(predictions, 'predictions'):
                    predictions = predictions.predictions
            else:
                raise ValueError("Model must have predict method")

            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)

        # Calculate performance metric
        if len(all_predictions) < 10:  # Minimum sample size
            return -999  # Invalid score

        try:
            score = metric_func(all_actuals, all_predictions)
            return score if not np.isnan(score) else -999
        except:
            return -999

    def run_cpcv(
        self,
        model_class: type,
        model_params: Dict[str, Any],
        X: DataFrame,
        y: pd.Series,
        metric_func: callable = None,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> CPCVResult:
        """
        Run complete Combinatorial Purged Cross-Validation.

        Args:
            model_class: Model class to test
            model_params: Model parameters
            X: Feature matrix
            y: Target vector
            metric_func: Performance metric function
            confidence_levels: Confidence levels for intervals

        Returns:
            CPCVResult with comprehensive statistics
        """
        print(f"🔬 Starting Combinatorial Purged Cross-Validation...")
        print(f"   Data shape: {X.shape}")
        print(f"   Target: {y.name if hasattr(y, 'name') else 'unnamed'}")

        # Generate base splits
        base_splits = self.generate_time_splits(len(X), X.index if hasattr(X, 'index') else None)
        print(f"   Base splits generated: {len(base_splits)}")

        if len(base_splits) < 3:
            raise ModelValidationError(
                "Insufficient base splits for CPCV",
                validation_metric="cpcv_splits",
                expected_value=3,
                actual_value=len(base_splits)
            )

        # Generate all valid combinations
        combinations = self.generate_combinations(base_splits)
        print(f"   Valid combinations: {len(combinations)}")

        if len(combinations) == 0:
            raise ModelValidationError("No valid combinations found for CPCV")

        # Evaluate each combination
        performance_scores = []
        valid_combinations = 0

        for i, combination in enumerate(combinations):
            if i % 50 == 0:
                print(f"   Progress: {i}/{len(combinations)} combinations")

            try:
                score = self.evaluate_model_combination(
                    model_class, model_params, X, y, combination, base_splits, metric_func
                )

                if score > -999:  # Valid score
                    performance_scores.append(score)
                    valid_combinations += 1

            except Exception as e:
                warnings.warn(f"Combination {i} failed: {e}")
                continue

        if len(performance_scores) < 10:
            raise ModelValidationError(
                f"Too few valid combinations: {len(performance_scores)}",
                validation_metric="valid_combinations",
                expected_value=10,
                actual_value=len(performance_scores)
            )

        performance_array = np.array(performance_scores)

        # Calculate statistics
        mean_perf = np.mean(performance_array)
        std_perf = np.std(performance_array)
        worst_case = np.min(performance_array)
        prob_positive = np.mean(performance_array > 0)

        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower = np.percentile(performance_array, 100 * alpha / 2)
            upper = np.percentile(performance_array, 100 * (1 - alpha / 2))
            confidence_intervals[f"{conf_level:.0%}"] = (lower, upper)

        print(f"✅ CPCV completed:")
        print(f"   Valid combinations: {valid_combinations}")
        print(f"   Mean performance: {mean_perf:.4f}")
        print(f"   Std performance: {std_perf:.4f}")
        print(f"   Worst case: {worst_case:.4f}")
        print(f"   P(positive): {prob_positive:.1%}")

        return CPCVResult(
            performance_distribution=performance_array,
            mean_performance=mean_perf,
            std_performance=std_perf,
            confidence_intervals=confidence_intervals,
            worst_case_performance=worst_case,
            probability_positive=prob_positive,
            n_combinations=valid_combinations,
            methodology=f"CPCV with {len(base_splits)} base splits, {valid_combinations} valid combinations"
        )


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio calculator.

    This class implements the deflated Sharpe ratio to account for selection bias
    that occurs when testing multiple models, parameters, or strategies.

    The DSR provides a more conservative and realistic assessment of whether
    observed performance represents genuine skill versus statistical luck.
    """

    def __init__(self):
        """Initialize DSR calculator."""
        pass

    def calculate_dsr(
        self,
        returns: Union[np.ndarray, pd.Series],
        n_trials: int,
        benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
        annual_factor: float = 252.0
    ) -> DSRResult:
        """
        Calculate the Deflated Sharpe Ratio.

        Args:
            returns: Strategy returns
            n_trials: Number of trials/models tested (critical for bias adjustment)
            benchmark_returns: Benchmark returns (optional)
            annual_factor: Annualization factor (252 for daily data)

        Returns:
            DSRResult with comprehensive DSR analysis
        """
        returns_array = np.asarray(returns)

        # Handle benchmark
        if benchmark_returns is not None:
            benchmark_array = np.asarray(benchmark_returns)
            excess_returns = returns_array - benchmark_array
        else:
            excess_returns = returns_array

        # Calculate original Sharpe ratio
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            raise ModelValidationError("Invalid returns for Sharpe ratio calculation")

        sharpe_original = (np.mean(excess_returns) * annual_factor) / (np.std(excess_returns) * np.sqrt(annual_factor))

        # Calculate selection bias adjustment
        # This accounts for testing multiple models/parameters
        if n_trials <= 1:
            n_trials = 1
            warnings.warn("n_trials should be > 1 for meaningful DSR calculation")

        # Variance inflation due to multiple testing
        variance_inflation = self._calculate_variance_inflation(len(excess_returns), n_trials)

        # Selection bias adjustment factor
        selection_bias = np.sqrt(2 * np.log(n_trials))

        # Deflated Sharpe ratio
        sharpe_deflated = (sharpe_original - selection_bias) / np.sqrt(variance_inflation)

        # Probability that the observed performance represents skill
        # Using normal CDF
        prob_skill = 1 - stats.norm.cdf(0, sharpe_deflated, 1)

        # Statistical significance (typically p < 0.05)
        is_significant = prob_skill > 0.95

        return DSRResult(
            deflated_sharpe=sharpe_deflated,
            original_sharpe=sharpe_original,
            probability_skill=prob_skill,
            selection_bias_adjustment=selection_bias,
            n_trials=n_trials,
            variance_inflation=variance_inflation,
            statistical_significance=is_significant
        )

    def _calculate_variance_inflation(self, n_observations: int, n_trials: int) -> float:
        """
        Calculate variance inflation factor due to multiple testing.

        Args:
            n_observations: Number of observations
            n_trials: Number of trials tested

        Returns:
            Variance inflation factor
        """
        # Simplified variance inflation calculation
        # In practice, this can be more sophisticated based on correlation structure
        base_inflation = 1.0
        trial_penalty = np.log(n_trials) / np.log(n_observations) if n_observations > 1 else 1.0

        return base_inflation + trial_penalty

    def calculate_haircut_sharpe(
        self,
        sharpe_ratio: float,
        n_trials: int,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate a conservative "haircut" Sharpe ratio.

        This provides a more conservative estimate by applying a penalty
        for multiple testing and model selection.

        Args:
            sharpe_ratio: Original Sharpe ratio
            n_trials: Number of trials
            confidence_level: Confidence level for haircut

        Returns:
            Conservative haircut Sharpe ratio
        """
        # Z-score for confidence level
        z_score = stats.norm.ppf(confidence_level)

        # Multiple testing penalty
        multiple_testing_penalty = np.sqrt(2 * np.log(n_trials))

        # Apply haircut
        haircut_sharpe = sharpe_ratio - (z_score * multiple_testing_penalty / np.sqrt(252))

        return max(haircut_sharpe, 0.0)  # Floor at 0

    def backtest_overfitting_probability(
        self,
        sharpe_ratio: float,
        n_trials: int,
        n_observations: int
    ) -> float:
        """
        Calculate the probability of backtest overfitting.

        This estimates the probability that the observed Sharpe ratio
        is due to overfitting rather than genuine skill.

        Args:
            sharpe_ratio: Observed Sharpe ratio
            n_trials: Number of trials
            n_observations: Number of observations

        Returns:
            Probability of overfitting (0-1)
        """
        # Expected maximum Sharpe ratio under null hypothesis (no skill)
        expected_max_sharpe = np.sqrt(2 * np.log(n_trials) / n_observations)

        # If observed Sharpe is much higher than expected under null,
        # it's more likely to be overfitting
        if sharpe_ratio <= 0:
            return 1.0  # Definitely overfitting if negative or zero

        # Probability calculation based on extreme value theory
        overfitting_prob = min(1.0, expected_max_sharpe / sharpe_ratio)

        return overfitting_prob


class AdvancedStatisticalValidator:
    """
    Combined advanced statistical validation framework.

    This class orchestrates both CPCV and DSR analysis to provide
    comprehensive statistical validation of financial ML models.
    """

    def __init__(self):
        """Initialize the advanced validator."""
        self.cpcv = CombinatorialPurgedCV()
        self.dsr = DeflatedSharpeRatio()

    def comprehensive_validation(
        self,
        model_class: type,
        model_params: Dict[str, Any],
        X: DataFrame,
        y: pd.Series,
        returns: Optional[pd.Series] = None,
        n_trials: int = 100,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive statistical validation combining CPCV and DSR.

        Args:
            model_class: Model class to validate
            model_params: Model parameters
            X: Feature matrix
            y: Target vector
            returns: Strategy returns (for DSR)
            n_trials: Number of trials for DSR calculation
            benchmark_returns: Benchmark returns

        Returns:
            Comprehensive validation results
        """
        print("🏛️ Starting Comprehensive Statistical Validation")
        print("="*60)

        results = {}

        # 1. CPCV Analysis
        print("\n🔬 Phase 1: Combinatorial Purged Cross-Validation")
        cpcv_result = self.cpcv.run_cpcv(model_class, model_params, X, y)
        results['cpcv'] = cpcv_result

        # 2. DSR Analysis (if returns provided)
        if returns is not None:
            print("\n📊 Phase 2: Deflated Sharpe Ratio Analysis")
            dsr_result = self.dsr.calculate_dsr(returns, n_trials, benchmark_returns)
            results['dsr'] = dsr_result

            # Additional DSR metrics
            haircut_sharpe = self.dsr.calculate_haircut_sharpe(
                dsr_result.original_sharpe, n_trials
            )
            overfitting_prob = self.dsr.backtest_overfitting_probability(
                dsr_result.original_sharpe, n_trials, len(returns)
            )

            results['haircut_sharpe'] = haircut_sharpe
            results['overfitting_probability'] = overfitting_prob

        # 3. Combined Analysis
        print("\n🎯 Phase 3: Combined Statistical Assessment")

        # Statistical significance across methods
        statistical_evidence = self._assess_statistical_evidence(results)
        results['statistical_evidence'] = statistical_evidence

        # Generate final recommendation
        recommendation = self._generate_recommendation(results)
        results['recommendation'] = recommendation

        self._print_summary(results)

        return results

    def _assess_statistical_evidence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall statistical evidence across methods."""
        evidence = {
            'cpcv_evidence': 'strong',
            'dsr_evidence': 'pending',
            'overall_confidence': 'medium'
        }

        # CPCV evidence assessment
        cpcv = results['cpcv']
        if cpcv.probability_positive > 0.95 and cpcv.worst_case_performance > 0:
            evidence['cpcv_evidence'] = 'very_strong'
        elif cpcv.probability_positive > 0.90:
            evidence['cpcv_evidence'] = 'strong'
        elif cpcv.probability_positive > 0.75:
            evidence['cpcv_evidence'] = 'moderate'
        else:
            evidence['cpcv_evidence'] = 'weak'

        # DSR evidence assessment
        if 'dsr' in results:
            dsr = results['dsr']
            if dsr.statistical_significance and dsr.probability_skill > 0.95:
                evidence['dsr_evidence'] = 'very_strong'
            elif dsr.probability_skill > 0.90:
                evidence['dsr_evidence'] = 'strong'
            elif dsr.probability_skill > 0.75:
                evidence['dsr_evidence'] = 'moderate'
            else:
                evidence['dsr_evidence'] = 'weak'

        # Overall confidence
        if evidence['cpcv_evidence'] in ['strong', 'very_strong']:
            if evidence['dsr_evidence'] in ['strong', 'very_strong', 'pending']:
                evidence['overall_confidence'] = 'high'
            else:
                evidence['overall_confidence'] = 'medium'
        else:
            evidence['overall_confidence'] = 'low'

        return evidence

    def _generate_recommendation(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate actionable recommendations based on validation results."""
        evidence = results['statistical_evidence']
        cpcv = results['cpcv']

        recommendation = {
            'deployment': 'not_recommended',
            'research': 'continue',
            'risk_management': 'high_caution'
        }

        if evidence['overall_confidence'] == 'high':
            if cpcv.worst_case_performance > 0:
                recommendation['deployment'] = 'recommended'
                recommendation['risk_management'] = 'standard'
            else:
                recommendation['deployment'] = 'conditional'
                recommendation['risk_management'] = 'elevated_caution'

        elif evidence['overall_confidence'] == 'medium':
            recommendation['deployment'] = 'conditional'
            recommendation['research'] = 'expand_validation'

        return recommendation

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive validation summary."""
        print("\n" + "="*80)
        print("📊 ADVANCED STATISTICAL VALIDATION SUMMARY")
        print("="*80)

        # CPCV Summary
        cpcv = results['cpcv']
        print(f"\n🔬 COMBINATORIAL PURGED CROSS-VALIDATION:")
        print(f"   Combinations tested: {cpcv.n_combinations:,}")
        print(f"   Mean performance: {cpcv.mean_performance:.4f}")
        print(f"   Performance std: {cpcv.std_performance:.4f}")
        print(f"   Worst case: {cpcv.worst_case_performance:.4f}")
        print(f"   P(positive): {cpcv.probability_positive:.1%}")

        for conf_level, (lower, upper) in cpcv.confidence_intervals.items():
            print(f"   {conf_level} CI: [{lower:.4f}, {upper:.4f}]")

        # DSR Summary
        if 'dsr' in results:
            dsr = results['dsr']
            print(f"\n📈 DEFLATED SHARPE RATIO ANALYSIS:")
            print(f"   Original Sharpe: {dsr.original_sharpe:.4f}")
            print(f"   Deflated Sharpe: {dsr.deflated_sharpe:.4f}")
            print(f"   Probability of skill: {dsr.probability_skill:.1%}")
            print(f"   Selection bias adjustment: {dsr.selection_bias_adjustment:.4f}")
            print(f"   Statistically significant: {dsr.statistical_significance}")

            if 'haircut_sharpe' in results:
                print(f"   Conservative (haircut) Sharpe: {results['haircut_sharpe']:.4f}")
            if 'overfitting_probability' in results:
                print(f"   Overfitting probability: {results['overfitting_probability']:.1%}")

        # Overall Assessment
        evidence = results['statistical_evidence']
        recommendation = results['recommendation']

        print(f"\n🎯 OVERALL STATISTICAL ASSESSMENT:")
        print(f"   CPCV evidence: {evidence['cpcv_evidence'].upper()}")
        print(f"   DSR evidence: {evidence['dsr_evidence'].upper()}")
        print(f"   Overall confidence: {evidence['overall_confidence'].upper()}")

        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   Deployment: {recommendation['deployment'].upper()}")
        print(f"   Research: {recommendation['research'].upper()}")
        print(f"   Risk management: {recommendation['risk_management'].upper()}")

        print("\n✅ Advanced statistical validation completed!")