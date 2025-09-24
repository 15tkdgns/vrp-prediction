#!/usr/bin/env python3
"""
📏 효과 크기 분석기
학술 논문을 위한 실용적 유의성 평가 도구

주요 기능:
- Cohen's d 및 변형들
- 상관계수 기반 효과 크기
- 금융 특화 효과 크기 지표
- 효과 크기 해석 가이드
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class EffectSizeAnalyzer:
    """
    다양한 효과 크기 지표 계산 및 해석 클래스

    통계적 유의성을 넘어 실용적 유의성을 평가하는 도구
    """

    def __init__(self):
        """초기화"""
        self.cohen_guidelines = {
            'd': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
            'r': {'small': 0.1, 'medium': 0.3, 'large': 0.5},
            'eta_squared': {'small': 0.01, 'medium': 0.06, 'large': 0.14},
            'omega_squared': {'small': 0.01, 'medium': 0.06, 'large': 0.14}
        }

        # 금융 특화 해석 기준
        self.financial_guidelines = {
            'sharpe_ratio_diff': {'negligible': 0.1, 'small': 0.3, 'medium': 0.5, 'large': 0.7},
            'mdd_reduction': {'negligible': 0.01, 'small': 0.05, 'medium': 0.10, 'large': 0.20},
            'accuracy_improvement': {'negligible': 0.01, 'small': 0.02, 'medium': 0.05, 'large': 0.10}
        }

    def cohens_d_paired(self, before: np.ndarray, after: np.ndarray) -> Dict:
        """
        쌍체 데이터의 Cohen's d 계산

        Args:
            before: 처치 전 데이터
            after: 처치 후 데이터

        Returns:
            Cohen's d 결과 딕셔너리
        """
        if len(before) != len(after):
            raise ValueError("before와 after 배열의 길이가 일치하지 않습니다")

        # 차이 점수
        diff = after - before
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)

        # 쌍체 Cohen's d
        cohens_d = mean_diff / std_diff if std_diff != 0 else 0

        # 편향 보정된 Cohen's d (Hedges' g)
        n = len(diff)
        correction_factor = 1 - (3 / (4 * n - 9)) if n > 3 else 1
        hedges_g = cohens_d * correction_factor

        # 효과 크기 해석
        interpretation = self._interpret_cohens_d(abs(cohens_d))

        return {
            'effect_size_type': "Cohen's d (paired)",
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'interpretation': interpretation,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'sample_size': n,
            'practical_significance': abs(cohens_d) >= 0.2  # 실용적 유의성 임계값
        }

    def cohens_d_independent(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """
        독립표본의 Cohen's d 계산

        Args:
            group1: 그룹 1 데이터
            group2: 그룹 2 데이터

        Returns:
            Cohen's d 결과 딕셔너리
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # 합동 표준편차 (pooled standard deviation)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

        # 편향 보정된 Cohen's d (Hedges' g)
        df = n1 + n2 - 2
        correction_factor = 1 - (3 / (4 * df - 1)) if df > 1 else 1
        hedges_g = cohens_d * correction_factor

        # Glass's delta (그룹 2의 표준편차 사용)
        glass_delta = (mean1 - mean2) / np.sqrt(var2) if var2 != 0 else 0

        # 효과 크기 해석
        interpretation = self._interpret_cohens_d(abs(cohens_d))

        return {
            'effect_size_type': "Cohen's d (independent)",
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'glass_delta': glass_delta,
            'interpretation': interpretation,
            'group1_mean': mean1,
            'group2_mean': mean2,
            'mean_difference': mean1 - mean2,
            'pooled_std': pooled_std,
            'sample_size_1': n1,
            'sample_size_2': n2,
            'practical_significance': abs(cohens_d) >= 0.2
        }

    def correlation_effect_size(self, correlation: float, n: int) -> Dict:
        """
        상관계수 기반 효과 크기 분석

        Args:
            correlation: 상관계수
            n: 표본 크기

        Returns:
            상관계수 효과 크기 결과 딕셔너리
        """
        if abs(correlation) > 1:
            raise ValueError("상관계수는 -1과 1 사이여야 합니다")

        # r²: 설명된 분산의 비율
        r_squared = correlation ** 2

        # Cohen's 기준으로 해석
        r_interpretation = self._interpret_correlation(abs(correlation))

        # 상관계수를 Cohen's d로 변환
        # d = 2r / sqrt(1 - r²)
        if abs(correlation) < 1:
            equivalent_cohens_d = (2 * correlation) / np.sqrt(1 - correlation**2)
        else:
            equivalent_cohens_d = np.inf if correlation > 0 else -np.inf

        return {
            'effect_size_type': 'Correlation Effect Size',
            'correlation': correlation,
            'r_squared': r_squared,
            'variance_explained_percent': r_squared * 100,
            'interpretation': r_interpretation,
            'equivalent_cohens_d': equivalent_cohens_d,
            'sample_size': n,
            'practical_significance': abs(correlation) >= 0.1
        }

    def eta_squared(self, groups: List[np.ndarray]) -> Dict:
        """
        Eta Squared (η²) 계산 - ANOVA 효과 크기

        Args:
            groups: 그룹별 데이터 리스트

        Returns:
            Eta squared 결과 딕셔너리
        """
        if len(groups) < 2:
            raise ValueError("최소 2개 그룹이 필요합니다")

        # 전체 데이터 결합
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        total_n = len(all_data)

        # 총 제곱합 (Total Sum of Squares)
        ss_total = np.sum((all_data - grand_mean) ** 2)

        # 집단 간 제곱합 (Between-group Sum of Squares)
        ss_between = 0
        for group in groups:
            group_mean = np.mean(group)
            group_n = len(group)
            ss_between += group_n * (group_mean - grand_mean) ** 2

        # 집단 내 제곱합 (Within-group Sum of Squares)
        ss_within = ss_total - ss_between

        # Eta Squared
        eta_squared = ss_between / ss_total if ss_total != 0 else 0

        # Partial Eta Squared (부분 에타 제곱)
        partial_eta_squared = ss_between / (ss_between + ss_within) if (ss_between + ss_within) != 0 else 0

        # Omega Squared (편향 보정된 효과 크기)
        k = len(groups)  # 그룹 수
        df_between = k - 1
        df_within = total_n - k
        ms_within = ss_within / df_within if df_within > 0 else 0

        omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within) if ss_total + ms_within != 0 else 0
        omega_squared = max(0, omega_squared)  # 음수 방지

        # 해석
        eta_interpretation = self._interpret_eta_squared(eta_squared)

        return {
            'effect_size_type': 'Eta Squared (ANOVA)',
            'eta_squared': eta_squared,
            'partial_eta_squared': partial_eta_squared,
            'omega_squared': omega_squared,
            'interpretation': eta_interpretation,
            'ss_total': ss_total,
            'ss_between': ss_between,
            'ss_within': ss_within,
            'n_groups': k,
            'total_n': total_n,
            'practical_significance': eta_squared >= 0.01
        }

    def financial_effect_sizes(self, baseline_metrics: Dict, improved_metrics: Dict) -> Dict:
        """
        금융 모델 특화 효과 크기 계산

        Args:
            baseline_metrics: 기준 모델 성능 지표
            improved_metrics: 개선 모델 성능 지표

        Returns:
            금융 특화 효과 크기 결과 딕셔너리
        """
        results = {}

        # 1. 샤프 비율 개선 효과
        if 'sharpe_ratio' in baseline_metrics and 'sharpe_ratio' in improved_metrics:
            sharpe_diff = improved_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
            sharpe_improvement_pct = (sharpe_diff / abs(baseline_metrics['sharpe_ratio'])) * 100 if baseline_metrics['sharpe_ratio'] != 0 else 0

            results['sharpe_ratio_improvement'] = {
                'absolute_difference': sharpe_diff,
                'relative_improvement_percent': sharpe_improvement_pct,
                'interpretation': self._interpret_sharpe_improvement(abs(sharpe_diff)),
                'practical_significance': abs(sharpe_diff) >= 0.1
            }

        # 2. MDD 감소 효과
        if 'max_drawdown' in baseline_metrics and 'max_drawdown' in improved_metrics:
            mdd_reduction = baseline_metrics['max_drawdown'] - improved_metrics['max_drawdown']
            mdd_reduction_pct = (mdd_reduction / baseline_metrics['max_drawdown']) * 100 if baseline_metrics['max_drawdown'] != 0 else 0

            results['mdd_reduction'] = {
                'absolute_reduction': mdd_reduction,
                'relative_reduction_percent': mdd_reduction_pct,
                'interpretation': self._interpret_mdd_reduction(mdd_reduction),
                'practical_significance': mdd_reduction >= 0.01
            }

        # 3. 정확도 개선 효과
        if 'accuracy' in baseline_metrics and 'accuracy' in improved_metrics:
            accuracy_improvement = improved_metrics['accuracy'] - baseline_metrics['accuracy']
            accuracy_improvement_pct = (accuracy_improvement / baseline_metrics['accuracy']) * 100 if baseline_metrics['accuracy'] != 0 else 0

            results['accuracy_improvement'] = {
                'absolute_improvement': accuracy_improvement,
                'relative_improvement_percent': accuracy_improvement_pct,
                'interpretation': self._interpret_accuracy_improvement(accuracy_improvement),
                'practical_significance': accuracy_improvement >= 0.01
            }

        # 4. 수익률 개선 효과
        if 'mean_return' in baseline_metrics and 'mean_return' in improved_metrics:
            return_improvement = improved_metrics['mean_return'] - baseline_metrics['mean_return']
            return_improvement_pct = (return_improvement / abs(baseline_metrics['mean_return'])) * 100 if baseline_metrics['mean_return'] != 0 else 0

            results['return_improvement'] = {
                'absolute_improvement': return_improvement,
                'relative_improvement_percent': return_improvement_pct,
                'annualized_improvement': return_improvement * 252,  # 일일 → 연간 환산
                'practical_significance': abs(return_improvement) >= 0.0001  # 0.01% 일일 수익률
            }

        # 5. 변동성 변화 효과
        if 'volatility' in baseline_metrics and 'volatility' in improved_metrics:
            volatility_change = improved_metrics['volatility'] - baseline_metrics['volatility']
            volatility_change_pct = (volatility_change / baseline_metrics['volatility']) * 100 if baseline_metrics['volatility'] != 0 else 0

            results['volatility_change'] = {
                'absolute_change': volatility_change,
                'relative_change_percent': volatility_change_pct,
                'interpretation': 'Volatility reduced' if volatility_change < 0 else 'Volatility increased',
                'practical_significance': abs(volatility_change) >= 0.001  # 0.1% 변동성 변화
            }

        return {
            'effect_size_type': 'Financial Model Comparison',
            'baseline_metrics': baseline_metrics,
            'improved_metrics': improved_metrics,
            'effect_sizes': results,
            'overall_improvement': self._assess_overall_improvement(results)
        }

    def model_performance_effect_size(self, y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict:
        """
        두 모델의 예측 성능 효과 크기 비교

        Args:
            y_true: 실제 값
            y_pred1: 모델 1 예측 값
            y_pred2: 모델 2 예측 값

        Returns:
            모델 성능 효과 크기 결과 딕셔너리
        """
        # 오차 계산
        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)

        # 쌍체 Cohen's d (오차 감소)
        error_reduction_cohens_d = self.cohens_d_paired(errors1, errors2)

        # 설명력 비교 (R²)
        def calculate_r2(y_t, y_p):
            ss_res = np.sum((y_t - y_p) ** 2)
            ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        r2_model1 = calculate_r2(y_true, y_pred1)
        r2_model2 = calculate_r2(y_true, y_pred2)
        r2_improvement = r2_model2 - r2_model1

        # 상관계수 비교
        corr1 = np.corrcoef(y_true, y_pred1)[0, 1] if len(y_true) > 1 else 0
        corr2 = np.corrcoef(y_true, y_pred2)[0, 1] if len(y_true) > 1 else 0

        return {
            'effect_size_type': 'Model Performance Comparison',
            'error_reduction_cohens_d': error_reduction_cohens_d['cohens_d'],
            'error_reduction_interpretation': error_reduction_cohens_d['interpretation'],
            'r2_model1': r2_model1,
            'r2_model2': r2_model2,
            'r2_improvement': r2_improvement,
            'correlation_model1': corr1,
            'correlation_model2': corr2,
            'correlation_improvement': corr2 - corr1,
            'mean_absolute_error_model1': np.mean(errors1),
            'mean_absolute_error_model2': np.mean(errors2),
            'mae_reduction_percent': ((np.mean(errors1) - np.mean(errors2)) / np.mean(errors1)) * 100 if np.mean(errors1) != 0 else 0,
            'sample_size': len(y_true),
            'practical_significance': abs(error_reduction_cohens_d['cohens_d']) >= 0.2
        }

    def _interpret_cohens_d(self, d: float) -> str:
        """Cohen's d 해석"""
        if d < 0.2:
            return "무시할 수 있는 효과"
        elif d < 0.5:
            return "작은 효과"
        elif d < 0.8:
            return "중간 효과"
        else:
            return "큰 효과"

    def _interpret_correlation(self, r: float) -> str:
        """상관계수 효과 크기 해석"""
        if r < 0.1:
            return "무시할 수 있는 효과"
        elif r < 0.3:
            return "작은 효과"
        elif r < 0.5:
            return "중간 효과"
        else:
            return "큰 효과"

    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Eta squared 해석"""
        if eta_sq < 0.01:
            return "무시할 수 있는 효과"
        elif eta_sq < 0.06:
            return "작은 효과"
        elif eta_sq < 0.14:
            return "중간 효과"
        else:
            return "큰 효과"

    def _interpret_sharpe_improvement(self, diff: float) -> str:
        """샤프 비율 개선 해석"""
        if diff < 0.1:
            return "무시할 수 있는 개선"
        elif diff < 0.3:
            return "작은 개선"
        elif diff < 0.5:
            return "중간 개선"
        else:
            return "큰 개선"

    def _interpret_mdd_reduction(self, reduction: float) -> str:
        """MDD 감소 해석"""
        if reduction < 0.01:
            return "무시할 수 있는 감소"
        elif reduction < 0.05:
            return "작은 감소"
        elif reduction < 0.10:
            return "중간 감소"
        else:
            return "큰 감소"

    def _interpret_accuracy_improvement(self, improvement: float) -> str:
        """정확도 개선 해석"""
        if improvement < 0.01:
            return "무시할 수 있는 개선"
        elif improvement < 0.02:
            return "작은 개선"
        elif improvement < 0.05:
            return "중간 개선"
        else:
            return "큰 개선"

    def _assess_overall_improvement(self, results: Dict) -> str:
        """전반적 개선 평가"""
        significant_improvements = 0
        total_metrics = 0

        for metric, result in results.items():
            total_metrics += 1
            if result.get('practical_significance', False):
                significant_improvements += 1

        if significant_improvements == 0:
            return "실용적 개선 없음"
        elif significant_improvements / total_metrics < 0.5:
            return "부분적 개선"
        elif significant_improvements / total_metrics < 0.8:
            return "상당한 개선"
        else:
            return "전면적 개선"

def main():
    """테스트 및 예제 실행"""
    print("📏 효과 크기 분석기 테스트")
    print("=" * 50)

    # 테스트 데이터 생성
    np.random.seed(42)

    # 모델 성능 비교 데이터
    model1_accuracy = np.random.normal(0.75, 0.05, 30)
    model2_accuracy = np.random.normal(0.78, 0.05, 30)

    # 예측 성능 데이터
    y_true = np.random.normal(0, 1, 100)
    y_pred1 = y_true + np.random.normal(0, 0.5, 100)
    y_pred2 = y_true + np.random.normal(0, 0.4, 100)

    analyzer = EffectSizeAnalyzer()

    # 1. 쌍체 Cohen's d
    print("\n1. 모델 정확도 개선의 Cohen's d")
    print("-" * 40)
    paired_d = analyzer.cohens_d_paired(model1_accuracy, model2_accuracy)
    print(f"Cohen's d: {paired_d['cohens_d']:.3f}")
    print(f"Hedges' g: {paired_d['hedges_g']:.3f}")
    print(f"해석: {paired_d['interpretation']}")
    print(f"실용적 유의성: {paired_d['practical_significance']}")

    # 2. 독립표본 Cohen's d
    print("\n2. 독립표본 Cohen's d")
    print("-" * 40)
    independent_d = analyzer.cohens_d_independent(model1_accuracy[:15], model2_accuracy[:15])
    print(f"Cohen's d: {independent_d['cohens_d']:.3f}")
    print(f"해석: {independent_d['interpretation']}")

    # 3. 상관계수 효과 크기
    print("\n3. 상관계수 효과 크기")
    print("-" * 40)
    corr_effect = analyzer.correlation_effect_size(0.65, n=50)
    print(f"상관계수: {corr_effect['correlation']:.3f}")
    print(f"설명된 분산: {corr_effect['variance_explained_percent']:.1f}%")
    print(f"해석: {corr_effect['interpretation']}")

    # 4. 모델 성능 효과 크기
    print("\n4. 모델 예측 성능 효과 크기")
    print("-" * 40)
    model_effect = analyzer.model_performance_effect_size(y_true, y_pred1, y_pred2)
    print(f"오차 감소 Cohen's d: {model_effect['error_reduction_cohens_d']:.3f}")
    print(f"R² 개선: {model_effect['r2_improvement']:.3f}")
    print(f"MAE 감소: {model_effect['mae_reduction_percent']:.1f}%")

    # 5. 금융 특화 효과 크기
    print("\n5. 금융 모델 효과 크기")
    print("-" * 40)
    baseline_metrics = {
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.15,
        'accuracy': 0.65,
        'mean_return': 0.0008
    }

    improved_metrics = {
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.12,
        'accuracy': 0.68,
        'mean_return': 0.0010
    }

    financial_effect = analyzer.financial_effect_sizes(baseline_metrics, improved_metrics)
    print(f"샤프 비율 개선: {financial_effect['effect_sizes']['sharpe_ratio_improvement']['absolute_difference']:.2f}")
    print(f"MDD 감소: {financial_effect['effect_sizes']['mdd_reduction']['absolute_reduction']:.3f}")
    print(f"전반적 개선: {financial_effect['overall_improvement']}")

if __name__ == "__main__":
    main()