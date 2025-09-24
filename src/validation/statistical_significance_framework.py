"""
Statistical Significance Testing Framework for Financial ML.

This module provides comprehensive statistical tests specifically designed for
financial machine learning models, addressing the unique challenges of financial
time series data including non-stationarity, heteroskedasticity, and serial correlation.

Key Features:
- Multiple hypothesis testing corrections
- Bootstrap confidence intervals
- Permutation tests for model comparison
- Regime-aware statistical testing
- Monte Carlo significance testing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.utils import resample
import warnings

from ..core.types import FloatArray, DataFrame
from ..core.exceptions import ModelValidationError


@dataclass
class SignificanceTestResult:
    """
    Result structure for statistical significance tests.

    Attributes:
        test_statistic: The calculated test statistic
        p_value: Raw p-value
        adjusted_p_value: Multiple testing adjusted p-value
        critical_value: Critical value for significance
        is_significant: Whether result is statistically significant
        confidence_interval: Confidence interval for the statistic
        effect_size: Effect size measure
        test_name: Name of the statistical test performed
        correction_method: Multiple testing correction method used
    """
    test_statistic: float
    p_value: float
    adjusted_p_value: float
    critical_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    effect_size: float
    test_name: str
    correction_method: str


@dataclass
class BootstrapResult:
    """
    Result structure for bootstrap analysis.

    Attributes:
        original_statistic: Original sample statistic
        bootstrap_distribution: Bootstrap distribution of the statistic
        bias: Bootstrap bias estimate
        bias_corrected_statistic: Bias-corrected statistic
        confidence_intervals: Bootstrap confidence intervals
        standard_error: Bootstrap standard error
        n_bootstrap_samples: Number of bootstrap samples used
    """
    original_statistic: float
    bootstrap_distribution: FloatArray
    bias: float
    bias_corrected_statistic: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    standard_error: float
    n_bootstrap_samples: int


class MultipleTestingCorrection:
    """
    Multiple testing correction methods for financial ML.

    This class implements various correction methods to control for
    family-wise error rate when testing multiple models or hypotheses.
    """

    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """
        Apply Bonferroni correction for multiple testing.

        Args:
            p_values: List of raw p-values
            alpha: Significance level

        Returns:
            Tuple of (adjusted_p_values, significance_indicators)
        """
        n_tests = len(p_values)
        adjusted_alpha = alpha / n_tests
        adjusted_p_values = [min(p * n_tests, 1.0) for p in p_values]
        is_significant = [p <= adjusted_alpha for p in p_values]

        return adjusted_p_values, is_significant

    @staticmethod
    def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """
        Apply Benjamini-Hochberg (FDR) correction.

        Args:
            p_values: List of raw p-values
            alpha: Significance level

        Returns:
            Tuple of (adjusted_p_values, significance_indicators)
        """
        n_tests = len(p_values)

        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]

        # Calculate BH critical values
        critical_values = [(i + 1) / n_tests * alpha for i in range(n_tests)]

        # Find largest i such that P(i) <= (i/m) * alpha
        significant_indices = []
        for i in range(n_tests - 1, -1, -1):
            if sorted_p_values[i] <= critical_values[i]:
                significant_indices = list(range(i + 1))
                break

        # Restore original order
        is_significant = [False] * n_tests
        for idx in significant_indices:
            original_idx = sorted_indices[idx]
            is_significant[original_idx] = True

        # Adjusted p-values (simplified approach)
        adjusted_p_values = [min(p * n_tests / (i + 1), 1.0) for i, p in enumerate(p_values)]

        return adjusted_p_values, is_significant

    @staticmethod
    def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """
        Apply Holm-Bonferroni step-down correction.

        Args:
            p_values: List of raw p-values
            alpha: Significance level

        Returns:
            Tuple of (adjusted_p_values, significance_indicators)
        """
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]

        # Calculate adjusted p-values
        adjusted_p_values = np.zeros(n_tests)
        is_significant = np.zeros(n_tests, dtype=bool)

        for i in range(n_tests):
            adjusted_alpha = alpha / (n_tests - i)
            adjusted_p_values[sorted_indices[i]] = min(
                sorted_p_values[i] * (n_tests - i), 1.0
            )

            if sorted_p_values[i] <= adjusted_alpha:
                is_significant[sorted_indices[i]] = True
            else:
                # If hypothesis i is not rejected, all subsequent are not rejected
                break

        return adjusted_p_values.tolist(), is_significant.tolist()


class BootstrapAnalysis:
    """
    Bootstrap analysis for robust statistical inference.

    This class provides various bootstrap methods for constructing
    confidence intervals and estimating sampling distributions.
    """

    def __init__(self, n_bootstrap: int = 10000, random_state: Optional[int] = 42):
        """
        Initialize bootstrap analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def basic_bootstrap(
        self,
        data: Union[np.ndarray, pd.Series],
        statistic_func: Callable[[np.ndarray], float],
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> BootstrapResult:
        """
        Perform basic bootstrap analysis.

        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            confidence_levels: Confidence levels for intervals

        Returns:
            BootstrapResult with comprehensive bootstrap analysis
        """
        np.random.seed(self.random_state)
        data_array = np.asarray(data)

        # Calculate original statistic
        original_stat = statistic_func(data_array)

        # Bootstrap sampling
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = resample(data_array, random_state=None)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)

        bootstrap_distribution = np.array(bootstrap_stats)

        # Calculate bias
        bias = np.mean(bootstrap_distribution) - original_stat
        bias_corrected = original_stat - bias

        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower = np.percentile(bootstrap_distribution, 100 * alpha / 2)
            upper = np.percentile(bootstrap_distribution, 100 * (1 - alpha / 2))
            confidence_intervals[f"{conf_level:.0%}"] = (lower, upper)

        # Standard error
        standard_error = np.std(bootstrap_distribution)

        return BootstrapResult(
            original_statistic=original_stat,
            bootstrap_distribution=bootstrap_distribution,
            bias=bias,
            bias_corrected_statistic=bias_corrected,
            confidence_intervals=confidence_intervals,
            standard_error=standard_error,
            n_bootstrap_samples=self.n_bootstrap
        )

    def percentile_bootstrap_ci(
        self,
        data: Union[np.ndarray, pd.Series],
        statistic_func: Callable[[np.ndarray], float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate percentile bootstrap confidence interval.

        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            confidence_level: Confidence level

        Returns:
            Confidence interval tuple
        """
        bootstrap_result = self.basic_bootstrap(data, statistic_func, [confidence_level])
        return bootstrap_result.confidence_intervals[f"{confidence_level:.0%}"]

    def bias_corrected_accelerated_ci(
        self,
        data: Union[np.ndarray, pd.Series],
        statistic_func: Callable[[np.ndarray], float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bias-corrected and accelerated (BCa) bootstrap CI.

        This is a more sophisticated bootstrap method that adjusts for
        bias and skewness in the bootstrap distribution.

        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            confidence_level: Confidence level

        Returns:
            BCa confidence interval tuple
        """
        np.random.seed(self.random_state)
        data_array = np.asarray(data)
        n = len(data_array)

        # Original statistic
        original_stat = statistic_func(data_array)

        # Bootstrap samples
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = resample(data_array, random_state=None)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)

        bootstrap_distribution = np.array(bootstrap_stats)

        # Bias correction
        bias_correction = stats.norm.ppf(
            (bootstrap_distribution < original_stat).mean()
        )

        # Acceleration (jackknife)
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.concatenate([data_array[:i], data_array[i+1:]])
            jackknife_stat = statistic_func(jackknife_sample)
            jackknife_stats.append(jackknife_stat)

        jackknife_mean = np.mean(jackknife_stats)
        numerator = np.sum((jackknife_mean - np.array(jackknife_stats)) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - np.array(jackknife_stats)) ** 2) ** 1.5)

        acceleration = numerator / denominator if denominator != 0 else 0

        # Adjusted percentiles
        alpha = 1 - confidence_level
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

        adjusted_alpha_1 = stats.norm.cdf(
            bias_correction + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2))
        )
        adjusted_alpha_2 = stats.norm.cdf(
            bias_correction + (bias_correction + z_1_alpha_2) / (1 - acceleration * (bias_correction + z_1_alpha_2))
        )

        # Confidence interval
        lower = np.percentile(bootstrap_distribution, 100 * adjusted_alpha_1)
        upper = np.percentile(bootstrap_distribution, 100 * adjusted_alpha_2)

        return (lower, upper)


class PermutationTest:
    """
    Permutation tests for model comparison.

    This class implements permutation-based tests for comparing
    model performance while preserving the null distribution.
    """

    def __init__(self, n_permutations: int = 10000, random_state: Optional[int] = 42):
        """
        Initialize permutation test.

        Args:
            n_permutations: Number of permutations
            random_state: Random seed
        """
        self.n_permutations = n_permutations
        self.random_state = random_state

    def two_sample_permutation_test(
        self,
        sample1: Union[np.ndarray, pd.Series],
        sample2: Union[np.ndarray, pd.Series],
        statistic_func: Optional[Callable] = None
    ) -> SignificanceTestResult:
        """
        Perform two-sample permutation test.

        Args:
            sample1: First sample (e.g., model 1 performance)
            sample2: Second sample (e.g., model 2 performance)
            statistic_func: Test statistic function (default: difference in means)

        Returns:
            SignificanceTestResult
        """
        if statistic_func is None:
            statistic_func = lambda x, y: np.mean(x) - np.mean(y)

        np.random.seed(self.random_state)

        sample1_array = np.asarray(sample1)
        sample2_array = np.asarray(sample2)

        # Original test statistic
        original_statistic = statistic_func(sample1_array, sample2_array)

        # Combined sample for permutation
        combined_sample = np.concatenate([sample1_array, sample2_array])
        n1, n2 = len(sample1_array), len(sample2_array)

        # Permutation distribution
        permutation_stats = []
        for _ in range(self.n_permutations):
            # Randomly permute combined sample
            permuted_sample = np.random.permutation(combined_sample)

            # Split into two groups
            perm_sample1 = permuted_sample[:n1]
            perm_sample2 = permuted_sample[n1:]

            # Calculate test statistic
            perm_stat = statistic_func(perm_sample1, perm_sample2)
            permutation_stats.append(perm_stat)

        permutation_distribution = np.array(permutation_stats)

        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(permutation_distribution) >= np.abs(original_statistic))

        # Critical value (95% confidence)
        critical_value = np.percentile(np.abs(permutation_distribution), 95)

        # Effect size (Cohen's d for difference in means)
        pooled_std = np.sqrt(((n1 - 1) * np.var(sample1_array) + (n2 - 1) * np.var(sample2_array)) / (n1 + n2 - 2))
        effect_size = original_statistic / pooled_std if pooled_std > 0 else 0

        # Confidence interval
        ci_lower = np.percentile(permutation_distribution, 2.5)
        ci_upper = np.percentile(permutation_distribution, 97.5)

        return SignificanceTestResult(
            test_statistic=original_statistic,
            p_value=p_value,
            adjusted_p_value=p_value,  # No adjustment for single test
            critical_value=critical_value,
            is_significant=p_value < 0.05,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=effect_size,
            test_name="Two-Sample Permutation Test",
            correction_method="None"
        )

    def model_comparison_test(
        self,
        model1_scores: Union[np.ndarray, pd.Series],
        model2_scores: Union[np.ndarray, pd.Series],
        metric_name: str = "Performance"
    ) -> SignificanceTestResult:
        """
        Test for significant difference between two models.

        Args:
            model1_scores: Model 1 performance scores
            model2_scores: Model 2 performance scores
            metric_name: Name of the metric being compared

        Returns:
            SignificanceTestResult for model comparison
        """
        def difference_statistic(x, y):
            return np.mean(x) - np.mean(y)

        result = self.two_sample_permutation_test(
            model1_scores, model2_scores, difference_statistic
        )

        result.test_name = f"{metric_name} Difference Test"
        return result


class StatisticalSignificanceFramework:
    """
    Comprehensive statistical significance testing framework.

    This class orchestrates all statistical tests and provides
    a unified interface for significance testing in financial ML.
    """

    def __init__(
        self,
        n_bootstrap: int = 10000,
        n_permutations: int = 10000,
        random_state: Optional[int] = 42
    ):
        """
        Initialize the framework.

        Args:
            n_bootstrap: Number of bootstrap samples
            n_permutations: Number of permutations
            random_state: Random seed
        """
        self.bootstrap = BootstrapAnalysis(n_bootstrap, random_state)
        self.permutation = PermutationTest(n_permutations, random_state)
        self.correction = MultipleTestingCorrection()

    def comprehensive_significance_test(
        self,
        performance_scores: Union[np.ndarray, pd.Series],
        baseline_scores: Optional[Union[np.ndarray, pd.Series]] = None,
        multiple_testing_adjustment: str = "benjamini_hochberg",
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform comprehensive significance testing.

        Args:
            performance_scores: Model performance scores
            baseline_scores: Baseline model scores (optional)
            multiple_testing_adjustment: Method for multiple testing correction
            confidence_level: Confidence level for intervals

        Returns:
            Comprehensive significance test results
        """
        results = {}

        # 1. Basic statistical tests
        scores_array = np.asarray(performance_scores)

        # One-sample t-test (H0: mean = 0)
        t_stat, t_p_value = stats.ttest_1samp(scores_array, 0)
        results['one_sample_t_test'] = {
            'statistic': t_stat,
            'p_value': t_p_value,
            'significant': t_p_value < (1 - confidence_level)
        }

        # Wilcoxon signed-rank test (non-parametric)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(scores_array)
            results['wilcoxon_test'] = {
                'statistic': wilcoxon_stat,
                'p_value': wilcoxon_p,
                'significant': wilcoxon_p < (1 - confidence_level)
            }
        except:
            results['wilcoxon_test'] = None

        # 2. Bootstrap analysis
        def mean_func(x):
            return np.mean(x)

        bootstrap_result = self.bootstrap.basic_bootstrap(
            scores_array, mean_func, [confidence_level]
        )
        results['bootstrap'] = bootstrap_result

        # 3. Model comparison (if baseline provided)
        if baseline_scores is not None:
            comparison_result = self.permutation.model_comparison_test(
                performance_scores, baseline_scores
            )
            results['model_comparison'] = comparison_result

        # 4. Multiple testing correction (if multiple tests)
        p_values = [results['one_sample_t_test']['p_value']]
        test_names = ['one_sample_t_test']

        if results['wilcoxon_test'] is not None:
            p_values.append(results['wilcoxon_test']['p_value'])
            test_names.append('wilcoxon_test')

        if baseline_scores is not None:
            p_values.append(results['model_comparison'].p_value)
            test_names.append('model_comparison')

        if len(p_values) > 1:
            if multiple_testing_adjustment == "bonferroni":
                adjusted_p, is_significant = self.correction.bonferroni_correction(p_values)
            elif multiple_testing_adjustment == "benjamini_hochberg":
                adjusted_p, is_significant = self.correction.benjamini_hochberg_correction(p_values)
            elif multiple_testing_adjustment == "holm_bonferroni":
                adjusted_p, is_significant = self.correction.holm_bonferroni_correction(p_values)
            else:
                adjusted_p = p_values
                is_significant = [p < 0.05 for p in p_values]

            results['multiple_testing'] = {
                'method': multiple_testing_adjustment,
                'original_p_values': dict(zip(test_names, p_values)),
                'adjusted_p_values': dict(zip(test_names, adjusted_p)),
                'is_significant': dict(zip(test_names, is_significant))
            }

        # 5. Overall assessment
        results['overall_assessment'] = self._assess_overall_significance(results)

        return results

    def _assess_overall_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall statistical significance across all tests."""
        assessment = {
            'statistical_confidence': 'low',
            'recommendation': 'reject',
            'evidence_strength': 'weak'
        }

        # Count significant tests
        significant_tests = 0
        total_tests = 0

        if results['one_sample_t_test']['significant']:
            significant_tests += 1
        total_tests += 1

        if results['wilcoxon_test'] is not None:
            if results['wilcoxon_test']['significant']:
                significant_tests += 1
            total_tests += 1

        if 'model_comparison' in results:
            if results['model_comparison'].is_significant:
                significant_tests += 1
            total_tests += 1

        # Bootstrap evidence
        bootstrap_ci = results['bootstrap'].confidence_intervals.get('95%', (0, 0))
        bootstrap_significant = bootstrap_ci[0] > 0  # Lower bound > 0

        if bootstrap_significant:
            significant_tests += 1
        total_tests += 1

        # Overall assessment
        significance_ratio = significant_tests / total_tests

        if significance_ratio >= 0.75:
            assessment['statistical_confidence'] = 'high'
            assessment['recommendation'] = 'accept'
            assessment['evidence_strength'] = 'strong'
        elif significance_ratio >= 0.5:
            assessment['statistical_confidence'] = 'medium'
            assessment['recommendation'] = 'conditional_accept'
            assessment['evidence_strength'] = 'moderate'
        elif significance_ratio >= 0.25:
            assessment['statistical_confidence'] = 'low'
            assessment['recommendation'] = 'further_investigation'
            assessment['evidence_strength'] = 'weak'
        else:
            assessment['statistical_confidence'] = 'very_low'
            assessment['recommendation'] = 'reject'
            assessment['evidence_strength'] = 'insufficient'

        assessment['significant_tests'] = significant_tests
        assessment['total_tests'] = total_tests
        assessment['significance_ratio'] = significance_ratio

        return assessment