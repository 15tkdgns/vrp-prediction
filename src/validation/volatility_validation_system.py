"""
Volatility prediction validation system integrating CPCV and DSR.

This module integrates the advanced statistical validation methods (CPCV and DSR)
with the volatility prediction system to provide robust statistical evidence
for model performance claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.validation.advanced_statistical_validation import (
    CombinatorialPurgedCV,
    DeflatedSharpeRatio,
    AdvancedStatisticalValidator
)
from src.validation.statistical_significance_framework import (
    StatisticalSignificanceFramework,
    BootstrapAnalysis,
    PermutationTest
)
from src.volatility.predictors.elasticnet_predictor import ElasticNetVolatilityPredictor
from src.volatility.features import VolatilityFeatureEngineer
from src.core.exceptions import DataValidationError


@dataclass
class VolatilityValidationResults:
    """Comprehensive validation results for volatility prediction."""

    # Basic performance metrics
    base_r2: float
    base_mae: float
    base_rmse: float

    # CPCV results
    cpcv_r2_distribution: List[float]
    cpcv_r2_mean: float
    cpcv_r2_std: float
    cpcv_r2_ci_lower: float
    cpcv_r2_ci_upper: float
    cpcv_paths_count: int

    # DSR results
    dsr_value: float
    dsr_pvalue: float
    dsr_is_significant: bool
    selection_bias_trials: int

    # Statistical significance
    bootstrap_r2_distribution: List[float]
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    permutation_pvalue: float
    statistical_significance: bool

    # Multiple testing corrections
    bonferroni_pvalue: float
    benjamini_hochberg_pvalue: float
    holm_bonferroni_pvalue: float

    # Summary metrics
    validation_score: float  # Overall validation confidence score
    recommendation: str     # Practical recommendation

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        result = asdict(self)
        # Convert numpy types to Python native types for JSON serialization
        for key, value in result.items():
            if hasattr(value, 'item'):  # numpy scalar
                result[key] = value.item()
            elif isinstance(value, np.bool_):
                result[key] = bool(value)
            elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'item'):
                result[key] = [v.item() if hasattr(v, 'item') else v for v in value]
        return result

    def summary_report(self) -> str:
        """Generate human-readable summary report."""
        return f"""
=== 변동성 예측 모델 통계적 검증 결과 ===

🎯 기본 성능:
   R² Score: {self.base_r2:.4f} ({self.base_r2*100:.2f}%)
   MAE: {self.base_mae:.6f}
   RMSE: {self.base_rmse:.6f}

📊 CPCV 분석 ({self.cpcv_paths_count}개 경로):
   R² 평균: {self.cpcv_r2_mean:.4f} ± {self.cpcv_r2_std:.4f}
   95% 신뢰구간: [{self.cpcv_r2_ci_lower:.4f}, {self.cpcv_r2_ci_upper:.4f}]
   분포 안정성: {'우수' if self.cpcv_r2_std < 0.05 else '보통' if self.cpcv_r2_std < 0.1 else '불안정'}

🔍 DSR 검증:
   DSR 값: {self.dsr_value:.4f}
   p-value: {self.dsr_pvalue:.6f}
   선택편향 보정: {'통과' if self.dsr_is_significant else '실패'}
   테스트 시행수: {self.selection_bias_trials}

📈 통계적 유의성:
   Bootstrap 95% CI: [{self.bootstrap_ci_lower:.4f}, {self.bootstrap_ci_upper:.4f}]
   Permutation p-value: {self.permutation_pvalue:.6f}
   유의성: {'통계적으로 유의함' if self.statistical_significance else '유의하지 않음'}

🛡️ 다중검정 보정:
   Bonferroni p-value: {self.bonferroni_pvalue:.6f}
   Benjamini-Hochberg p-value: {self.benjamini_hochberg_pvalue:.6f}
   Holm-Bonferroni p-value: {self.holm_bonferroni_pvalue:.6f}

✅ 최종 평가:
   검증 점수: {self.validation_score:.0f}/100
   권장사항: {self.recommendation}
"""


class VolatilityValidationSystem:
    """
    Comprehensive validation system for volatility prediction models.

    This class integrates all advanced statistical validation methods to provide
    robust evidence for volatility prediction model performance claims.
    """

    def __init__(self,
                 cpcv_n_combinations: int = 500,
                 bootstrap_n_samples: int = 1000,
                 permutation_n_tests: int = 1000,
                 confidence_level: float = 0.95):
        """
        Initialize validation system.

        Args:
            cpcv_n_combinations: Number of CPCV combinations to generate
            bootstrap_n_samples: Number of bootstrap samples
            permutation_n_tests: Number of permutation tests
            confidence_level: Confidence level for intervals
        """
        self.cpcv_n_combinations = cpcv_n_combinations
        self.bootstrap_n_samples = bootstrap_n_samples
        self.permutation_n_tests = permutation_n_tests
        self.confidence_level = confidence_level

        # Initialize validation components
        self.cpcv = CombinatorialPurgedCV(
            n_splits=5,
            min_train_size=252,
            purge_gap=5,
            embargo_pct=0.01,
            max_combinations=self.cpcv_n_combinations
        )

        self.dsr = DeflatedSharpeRatio()
        self.significance_framework = StatisticalSignificanceFramework()

    def validate_volatility_model(self,
                                 data: pd.DataFrame,
                                 model_params: Optional[Dict[str, Any]] = None,
                                 selection_bias_trials: int = 100) -> VolatilityValidationResults:
        """
        Perform comprehensive validation of volatility prediction model.

        Args:
            data: Historical price data with OHLCV columns
            model_params: Model parameters for ElasticNet
            selection_bias_trials: Number of trials for DSR calculation

        Returns:
            Comprehensive validation results
        """
        print("🔬 변동성 예측 모델 종합 검증 시작...")
        print(f"   📊 데이터: {len(data)} 일간")
        print(f"   🔀 CPCV 조합: {self.cpcv_n_combinations}개")
        print(f"   🎲 Bootstrap 샘플: {self.bootstrap_n_samples}개")
        print(f"   🧪 순열검정: {self.permutation_n_tests}회")

        try:
            # 1. 기본 특성 준비
            print("\n🔧 특성 엔지니어링...")
            features, target = self._prepare_features(data)

            # 2. 기본 모델 성능
            print("🤖 기본 모델 성능 평가...")
            base_metrics = self._evaluate_base_model(features, target, model_params)

            # 3. CPCV 분석
            print("📊 CPCV 분석 (수백 개의 백테스트 경로 생성)...")
            cpcv_results = self._run_cpcv_analysis(features, target, model_params)

            # 4. DSR 계산
            print("🔍 선택편향 보정 (DSR) 계산...")
            dsr_results = self._calculate_dsr(features, target, model_params, selection_bias_trials)

            # 5. 통계적 유의성 검증
            print("📈 통계적 유의성 검증...")
            significance_results = self._test_statistical_significance(features, target, model_params)

            # 6. 다중검정 보정
            print("🛡️ 다중검정 보정...")
            corrected_pvalues = self._apply_multiple_testing_corrections([
                significance_results['permutation_pvalue'],
                dsr_results['dsr_pvalue']
            ])

            # 7. 종합 평가
            validation_score, recommendation = self._calculate_validation_score(
                base_metrics, cpcv_results, dsr_results, significance_results
            )

            # 결과 정리
            results = VolatilityValidationResults(
                # Basic metrics
                base_r2=base_metrics['r2'],
                base_mae=base_metrics['mae'],
                base_rmse=base_metrics['rmse'],

                # CPCV results
                cpcv_r2_distribution=cpcv_results['r2_distribution'],
                cpcv_r2_mean=cpcv_results['r2_mean'],
                cpcv_r2_std=cpcv_results['r2_std'],
                cpcv_r2_ci_lower=cpcv_results['r2_ci_lower'],
                cpcv_r2_ci_upper=cpcv_results['r2_ci_upper'],
                cpcv_paths_count=len(cpcv_results['r2_distribution']),

                # DSR results
                dsr_value=dsr_results['dsr_value'],
                dsr_pvalue=dsr_results['dsr_pvalue'],
                dsr_is_significant=dsr_results['is_significant'],
                selection_bias_trials=selection_bias_trials,

                # Statistical significance
                bootstrap_r2_distribution=significance_results['bootstrap_distribution'],
                bootstrap_ci_lower=significance_results['bootstrap_ci_lower'],
                bootstrap_ci_upper=significance_results['bootstrap_ci_upper'],
                permutation_pvalue=significance_results['permutation_pvalue'],
                statistical_significance=significance_results['is_significant'],

                # Multiple testing corrections
                bonferroni_pvalue=corrected_pvalues['bonferroni'],
                benjamini_hochberg_pvalue=corrected_pvalues['benjamini_hochberg'],
                holm_bonferroni_pvalue=corrected_pvalues['holm_bonferroni'],

                # Summary
                validation_score=validation_score,
                recommendation=recommendation
            )

            print("\n✅ 종합 검증 완료!")
            return results

        except Exception as e:
            raise DataValidationError(
                f"Volatility validation failed: {str(e)}"
            ) from e

    def _prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for validation."""
        engineer = VolatilityFeatureEngineer(
            volatility_windows=[5, 10, 20],
            ma_windows=[20, 50],
            include_lags=True,
            max_lags=3
        )

        feature_result = engineer.create_features(data)
        features_df = feature_result.features

        # Extract target and features
        target = features_df['next_day_volatility']
        feature_columns = [col for col in features_df.columns if col != 'next_day_volatility']
        features = features_df[feature_columns]

        # Clean data
        mask = ~(features.isnull().any(axis=1) | target.isnull())
        clean_features = features[mask]
        clean_target = target[mask]

        print(f"   ✅ 특성 {len(feature_columns)}개, 샘플 {len(clean_features)}개")
        return clean_features, clean_target

    def _evaluate_base_model(self,
                           features: pd.DataFrame,
                           target: pd.Series,
                           model_params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Evaluate base model performance."""
        if model_params is None:
            model_params = {'alpha': 0.001, 'l1_ratio': 0.5, 'random_state': 42}

        # Simple train-test split
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

        # Train model
        model = ElasticNetVolatilityPredictor(**model_params)
        model.fit(X_train, y_train)

        # Predictions
        pred_result = model.predict(X_test)
        predictions = pred_result.predictions

        # Metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"   ✅ 기본 성능: R²={r2:.4f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }

    def _run_cpcv_analysis(self,
                          features: pd.DataFrame,
                          target: pd.Series,
                          model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run CPCV analysis to generate performance distribution."""
        if model_params is None:
            model_params = {'alpha': 0.001, 'l1_ratio': 0.5, 'random_state': 42}

        # Use CPCV's run_cpcv method
        try:
            from sklearn.metrics import r2_score

            def r2_metric_func(y_true, y_pred):
                return r2_score(y_true, y_pred)

            cpcv_result = self.cpcv.run_cpcv(
                model_class=ElasticNetVolatilityPredictor,
                model_params=model_params,
                X=features,
                y=target,
                metric_func=r2_metric_func
            )

            print(f"   ✅ CPCV 완료: {len(cpcv_result.performance_distribution)}개 경로 생성")
            print(f"   📊 R² 분포: {cpcv_result.mean_performance:.4f} ± {cpcv_result.std_performance:.4f}")

            return {
                'r2_distribution': cpcv_result.performance_distribution.tolist(),
                'r2_mean': float(cpcv_result.mean_performance),
                'r2_std': float(cpcv_result.std_performance),
                'r2_ci_lower': float(cpcv_result.confidence_intervals['95%'][0]),
                'r2_ci_upper': float(cpcv_result.confidence_intervals['95%'][1])
            }

        except Exception as e:
            # Fallback to simple time series split
            warnings.warn(f"CPCV failed, using fallback: {e}")

            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=min(5, len(features) // 100))

            r2_scores = []
            for train_idx, test_idx in tscv.split(features):
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

                try:
                    model = ElasticNetVolatilityPredictor(**model_params)
                    model.fit(X_train, y_train)
                    pred_result = model.predict(X_test)
                    predictions = pred_result.predictions

                    from sklearn.metrics import r2_score
                    r2 = r2_score(y_test, predictions)
                    r2_scores.append(r2)

                except Exception:
                    continue

            if r2_scores:
                r2_array = np.array(r2_scores)
                alpha = 1 - self.confidence_level
                ci_lower = np.percentile(r2_array, 100 * alpha/2)
                ci_upper = np.percentile(r2_array, 100 * (1 - alpha/2))

                print(f"   ✅ Fallback CV 완료: {len(r2_scores)}개 폴드")
                print(f"   📊 R² 분포: {r2_array.mean():.4f} ± {r2_array.std():.4f}")

                return {
                    'r2_distribution': r2_scores,
                    'r2_mean': float(r2_array.mean()),
                    'r2_std': float(r2_array.std()),
                    'r2_ci_lower': float(ci_lower),
                    'r2_ci_upper': float(ci_upper)
                }
            else:
                raise DataValidationError("CPCV analysis failed completely")

    def _calculate_dsr(self,
                      features: pd.DataFrame,
                      target: pd.Series,
                      model_params: Optional[Dict[str, Any]] = None,
                      n_trials: int = 100) -> Dict[str, Any]:
        """Calculate Deflated Sharpe Ratio for selection bias correction."""
        if model_params is None:
            model_params = {'alpha': 0.001, 'l1_ratio': 0.5, 'random_state': 42}

        # Generate multiple model performances to simulate selection bias
        performances = []

        print(f"   🔍 {n_trials}회 시행으로 선택편향 추정...")

        for trial in range(n_trials):
            # Add noise to model parameters to simulate different trials
            trial_params = model_params.copy()
            trial_params['alpha'] = model_params['alpha'] * np.random.uniform(0.5, 2.0)
            trial_params['l1_ratio'] = np.clip(
                model_params['l1_ratio'] + np.random.normal(0, 0.1), 0.01, 0.99
            )
            trial_params['random_state'] = trial

            # Simple validation
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

            try:
                model = ElasticNetVolatilityPredictor(**trial_params)
                model.fit(X_train, y_train)
                pred_result = model.predict(X_test)
                predictions = pred_result.predictions

                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, predictions)
                performances.append(r2)

            except Exception:
                continue

            if trial % 20 == 0:
                print(f"   진행률: {trial}/{n_trials}")

        if len(performances) == 0:
            raise DataValidationError("DSR calculation failed - no successful trials")

        # Calculate DSR
        performances_array = np.array(performances)
        best_performance = np.max(performances_array)

        # Convert R² to pseudo-returns for DSR calculation
        # Higher R² means better "returns"
        pseudo_returns = performances_array * 100  # Scale for better numerical properties

        dsr_result = self.dsr.calculate_dsr(
            returns=pseudo_returns,
            n_trials=len(performances)
        )

        print(f"   ✅ DSR 계산 완료: {dsr_result.deflated_sharpe:.4f} (p={dsr_result.probability_skill:.6f})")

        return {
            'dsr_value': dsr_result.deflated_sharpe,
            'dsr_pvalue': dsr_result.probability_skill,
            'is_significant': dsr_result.probability_skill < 0.05,
            'best_performance': best_performance,
            'n_trials_successful': len(performances)
        }

    def _test_statistical_significance(self,
                                     features: pd.DataFrame,
                                     target: pd.Series,
                                     model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Test statistical significance using bootstrap and permutation tests."""
        if model_params is None:
            model_params = {'alpha': 0.001, 'l1_ratio': 0.5, 'random_state': 42}

        # Simple train-test split
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

        def score_func(X_tr, y_tr, X_te, y_te):
            """Scoring function for significance tests."""
            model = ElasticNetVolatilityPredictor(**model_params)
            model.fit(X_tr, y_tr)
            pred_result = model.predict(X_te)
            predictions = pred_result.predictions

            from sklearn.metrics import r2_score
            return r2_score(y_te, predictions)

        # Bootstrap analysis
        print("   🎲 Bootstrap 분석...")
        bootstrap = BootstrapAnalysis(
            n_bootstrap=self.bootstrap_n_samples,
            random_state=42
        )

        # Calculate base score
        base_score = score_func(X_train, y_train, X_test, y_test)

        # Bootstrap the score
        def bootstrap_statistic(indices):
            X_bootstrap = X_train.iloc[indices]
            y_bootstrap = y_train.iloc[indices]
            return score_func(X_bootstrap, y_bootstrap, X_test, y_test)

        bootstrap_result = bootstrap.basic_bootstrap(
            data=np.arange(len(X_train)),
            statistic_func=bootstrap_statistic,
            confidence_levels=[self.confidence_level]
        )

        # Permutation test
        print("   🧪 순열검정...")
        permutation = PermutationTest(
            n_permutations=self.permutation_n_tests
        )

        # Create simple permutation test
        permutation_scores = []
        for _ in range(self.permutation_n_tests):
            y_permuted = y_train.sample(frac=1).reset_index(drop=True)
            perm_score = score_func(X_train, y_permuted, X_test, y_test)
            permutation_scores.append(perm_score)

        # Calculate p-value
        permutation_pvalue = np.mean(np.array(permutation_scores) >= base_score)

        ci_key = f"{self.confidence_level:.0%}"
        if ci_key in bootstrap_result.confidence_intervals:
            ci_lower, ci_upper = bootstrap_result.confidence_intervals[ci_key]
        else:
            ci_lower, ci_upper = bootstrap_result.confidence_intervals['95%']

        print(f"   ✅ 통계적 유의성: Bootstrap CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   ✅ 순열검정: p-value={permutation_pvalue:.6f}")

        return {
            'bootstrap_distribution': bootstrap_result.bootstrap_distribution.tolist(),
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper,
            'permutation_pvalue': permutation_pvalue,
            'is_significant': permutation_pvalue < 0.05
        }

    def _apply_multiple_testing_corrections(self, p_values: List[float]) -> Dict[str, float]:
        """Apply multiple testing corrections."""
        from scipy.stats import false_discovery_control

        p_array = np.array(p_values)

        # Bonferroni correction
        bonferroni = np.minimum(p_array * len(p_array), 1.0)

        # Benjamini-Hochberg (FDR)
        benjamini_hochberg = false_discovery_control(p_array, method='bh')

        # Holm-Bonferroni
        sorted_indices = np.argsort(p_array)
        holm_bonferroni = np.zeros_like(p_array)
        for i, idx in enumerate(sorted_indices):
            holm_bonferroni[idx] = min(p_array[idx] * (len(p_array) - i), 1.0)

        return {
            'bonferroni': float(bonferroni[0]),  # Use first p-value for main result
            'benjamini_hochberg': float(benjamini_hochberg[0]),
            'holm_bonferroni': float(holm_bonferroni[0])
        }

    def _calculate_validation_score(self,
                                  base_metrics: Dict[str, float],
                                  cpcv_results: Dict[str, Any],
                                  dsr_results: Dict[str, Any],
                                  significance_results: Dict[str, Any]) -> Tuple[float, str]:
        """Calculate overall validation score and recommendation."""
        score = 0
        criteria = []

        # Base performance (30 points)
        r2 = base_metrics['r2']
        if r2 > 0.20:
            score += 30
            criteria.append("✅ 강력한 기본 성능 (R² > 0.20)")
        elif r2 > 0.10:
            score += 20
            criteria.append("✅ 양호한 기본 성능 (R² > 0.10)")
        elif r2 > 0.05:
            score += 10
            criteria.append("⚠️ 보통 기본 성능 (R² > 0.05)")
        else:
            criteria.append("❌ 낮은 기본 성능 (R² ≤ 0.05)")

        # CPCV stability (25 points)
        cpcv_std = cpcv_results['r2_std']
        if cpcv_std < 0.05:
            score += 25
            criteria.append("✅ 매우 안정적인 CPCV 분포")
        elif cpcv_std < 0.10:
            score += 15
            criteria.append("✅ 안정적인 CPCV 분포")
        elif cpcv_std < 0.20:
            score += 10
            criteria.append("⚠️ 보통 CPCV 분포")
        else:
            criteria.append("❌ 불안정한 CPCV 분포")

        # DSR significance (20 points)
        if dsr_results['is_significant']:
            score += 20
            criteria.append("✅ DSR 검증 통과 (선택편향 없음)")
        else:
            criteria.append("❌ DSR 검증 실패 (선택편향 우려)")

        # Statistical significance (25 points)
        if significance_results['is_significant']:
            score += 25
            criteria.append("✅ 통계적 유의성 확인")
        else:
            criteria.append("❌ 통계적 유의성 없음")

        # Generate recommendation
        if score >= 85:
            recommendation = "🟢 실전 투자 권장 - 모든 검증 기준 통과"
        elif score >= 70:
            recommendation = "🟡 제한적 사용 권장 - 대부분 검증 기준 통과"
        elif score >= 50:
            recommendation = "🟠 신중한 연구 사용 - 일부 검증 기준 통과"
        else:
            recommendation = "🔴 사용 비권장 - 검증 기준 미달"

        return score, recommendation


def main():
    """Main function for testing the validation system."""
    import argparse

    parser = argparse.ArgumentParser(description="Volatility Model Validation System")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--full", action="store_true", help="Run full validation")
    parser.add_argument("--days", type=int, default=1000, help="Number of days to simulate")

    args = parser.parse_args()

    # Generate sample data
    print("📊 샘플 데이터 생성...")
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=args.days, freq='D')

    # Simulate realistic stock data with volatility clustering
    initial_price = 100.0
    returns = []
    current_vol = 0.02

    for i in range(args.days):
        vol_shock = np.random.normal(0, 0.001)
        current_vol = 0.95 * current_vol + 0.05 * 0.02 + vol_shock
        current_vol = max(0.005, min(0.1, current_vol))

        daily_return = np.random.normal(0.0005, current_vol)
        returns.append(daily_return)

    prices = [initial_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    prices = prices[1:]

    data = pd.DataFrame({
        'open': [p * np.random.uniform(0.995, 1.005) for p in prices],
        'high': [p * np.random.uniform(1.005, 1.025) for p in prices],
        'low': [p * np.random.uniform(0.975, 0.995) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(13, 0.5, args.days).astype(int)
    }, index=dates)

    # Initialize validation system
    if args.demo:
        validator = VolatilityValidationSystem(
            cpcv_n_combinations=50,
            bootstrap_n_samples=100,
            permutation_n_tests=100
        )
    else:
        validator = VolatilityValidationSystem(
            cpcv_n_combinations=500,
            bootstrap_n_samples=1000,
            permutation_n_tests=1000
        )

    # Run validation
    results = validator.validate_volatility_model(
        data=data,
        selection_bias_trials=50 if args.demo else 100
    )

    # Display results
    print(results.summary_report())

    # Save results
    output_path = Path("/root/workspace/data/raw/volatility_validation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n💾 결과 저장됨: {output_path}")

    return results


if __name__ == "__main__":
    main()