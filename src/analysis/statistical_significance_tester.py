#!/usr/bin/env python3
"""
📊 통계적 유의성 검증 시스템
학술 논문 작성을 위한 엄격한 통계적 검증 도구

주요 기능:
- 모델 간 성능 차이 유의성 검정
- 신뢰구간 계산
- 효과 크기 측정
- 다중 비교 보정
- 검정력 분석
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class StatisticalSignificanceTester:
    """
    학술 논문을 위한 통계적 유의성 검증 클래스

    금융 머신러닝 모델의 성능 비교에 특화된 통계 검정 도구
    """

    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        초기화

        Args:
            alpha: 유의수준 (기본값: 0.05)
            power: 검정력 (기본값: 0.8)
        """
        self.alpha = alpha
        self.power = power
        self.results = {}

    def paired_ttest(self, model1_scores: np.ndarray, model2_scores: np.ndarray,
                    model1_name: str = "Model1", model2_name: str = "Model2") -> Dict:
        """
        쌍체 t-검정 (Paired t-test)

        동일한 데이터에서 두 모델의 성능을 비교
        가정: 정규분포, 쌍체 관측

        Args:
            model1_scores: 모델1의 성능 점수들
            model2_scores: 모델2의 성능 점수들
            model1_name: 모델1 이름
            model2_name: 모델2 이름

        Returns:
            검정 결과 딕셔너리
        """
        # 입력 검증
        if len(model1_scores) != len(model2_scores):
            raise ValueError("두 모델의 점수 개수가 일치하지 않습니다")

        if len(model1_scores) < 3:
            raise ValueError("유의한 검정을 위해 최소 3개 이상의 관측값이 필요합니다")

        # 정규성 검정 (Shapiro-Wilk)
        diff_scores = model1_scores - model2_scores
        shapiro_stat, shapiro_p = stats.shapiro(diff_scores)
        is_normal = shapiro_p > 0.05

        # t-검정 수행
        t_stat, p_value = ttest_rel(model1_scores, model2_scores)

        # 효과 크기 (Cohen's d for paired samples)
        mean_diff = np.mean(diff_scores)
        std_diff = np.std(diff_scores, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff != 0 else 0

        # 신뢰구간 계산 (95%)
        n = len(diff_scores)
        se_diff = std_diff / np.sqrt(n)
        t_critical = stats.t.ppf(1 - self.alpha/2, n-1)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        # 검정력 계산
        calculated_power = ttest_power(cohens_d, n, self.alpha, alternative='two-sided')

        # 실용적 유의성 판단
        # Cohen's d 기준: 0.2(작음), 0.5(중간), 0.8(큼)
        if abs(cohens_d) < 0.2:
            effect_size_interpretation = "무시할 수 있는 효과"
        elif abs(cohens_d) < 0.5:
            effect_size_interpretation = "작은 효과"
        elif abs(cohens_d) < 0.8:
            effect_size_interpretation = "중간 효과"
        else:
            effect_size_interpretation = "큰 효과"

        result = {
            'test_type': 'Paired t-test',
            'model1_name': model1_name,
            'model2_name': model2_name,
            'n_observations': n,
            'model1_mean': np.mean(model1_scores),
            'model2_mean': np.mean(model2_scores),
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'alpha': self.alpha,
            'cohens_d': cohens_d,
            'effect_size_interpretation': effect_size_interpretation,
            'confidence_interval_95': (ci_lower, ci_upper),
            'statistical_power': calculated_power,
            'normality_test': {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'is_normal': is_normal
            },
            'recommendation': self._get_ttest_recommendation(p_value, is_normal, calculated_power)
        }

        return result

    def wilcoxon_signed_rank_test(self, model1_scores: np.ndarray, model2_scores: np.ndarray,
                                 model1_name: str = "Model1", model2_name: str = "Model2") -> Dict:
        """
        윌콕슨 부호 순위 검정 (Wilcoxon Signed-Rank Test)

        비모수 검정으로 정규성 가정이 필요 없음
        쌍체 데이터의 중앙값 차이 검정

        Args:
            model1_scores: 모델1의 성능 점수들
            model2_scores: 모델2의 성능 점수들
            model1_name: 모델1 이름
            model2_name: 모델2 이름

        Returns:
            검정 결과 딕셔너리
        """
        # 입력 검증
        if len(model1_scores) != len(model2_scores):
            raise ValueError("두 모델의 점수 개수가 일치하지 않습니다")

        # 차이 계산
        diff_scores = model1_scores - model2_scores

        # 0인 차이 제거
        non_zero_diff = diff_scores[diff_scores != 0]

        if len(non_zero_diff) < 5:
            warnings.warn("유의한 윌콕슨 검정을 위해 최소 5개 이상의 non-zero 차이가 권장됩니다")

        # 윌콕슨 검정 수행
        if len(non_zero_diff) == 0:
            w_stat, p_value = 0, 1.0
        else:
            w_stat, p_value = wilcoxon(non_zero_diff, alternative='two-sided')

        # 효과 크기 (r = Z / sqrt(N))
        n = len(non_zero_diff)
        if n > 0:
            z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 0
            effect_size_r = abs(z_score) / np.sqrt(n)
        else:
            effect_size_r = 0

        # 효과 크기 해석 (Cohen's 기준)
        if effect_size_r < 0.1:
            effect_size_interpretation = "무시할 수 있는 효과"
        elif effect_size_r < 0.3:
            effect_size_interpretation = "작은 효과"
        elif effect_size_r < 0.5:
            effect_size_interpretation = "중간 효과"
        else:
            effect_size_interpretation = "큰 효과"

        result = {
            'test_type': 'Wilcoxon Signed-Rank Test',
            'model1_name': model1_name,
            'model2_name': model2_name,
            'n_observations': len(model1_scores),
            'n_non_zero_differences': n,
            'model1_median': np.median(model1_scores),
            'model2_median': np.median(model2_scores),
            'median_difference': np.median(diff_scores),
            'w_statistic': w_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'alpha': self.alpha,
            'effect_size_r': effect_size_r,
            'effect_size_interpretation': effect_size_interpretation,
            'zero_differences': len(diff_scores) - n,
            'recommendation': self._get_wilcoxon_recommendation(p_value, n)
        }

        return result

    def independent_samples_test(self, model1_scores: np.ndarray, model2_scores: np.ndarray,
                               model1_name: str = "Model1", model2_name: str = "Model2") -> Dict:
        """
        독립표본 검정 (t-test 또는 Mann-Whitney U test)

        서로 다른 데이터셋에서 두 모델의 성능을 비교
        정규성에 따라 적절한 검정 자동 선택

        Args:
            model1_scores: 모델1의 성능 점수들
            model2_scores: 모델2의 성능 점수들
            model1_name: 모델1 이름
            model2_name: 모델2 이름

        Returns:
            검정 결과 딕셔너리
        """
        # 정규성 검정
        if len(model1_scores) >= 3:
            _, p1 = stats.shapiro(model1_scores)
        else:
            p1 = 0

        if len(model2_scores) >= 3:
            _, p2 = stats.shapiro(model2_scores)
        else:
            p2 = 0

        is_normal = (p1 > 0.05) and (p2 > 0.05)

        if is_normal and len(model1_scores) >= 3 and len(model2_scores) >= 3:
            # 독립표본 t-검정
            t_stat, p_value = stats.ttest_ind(model1_scores, model2_scores)
            test_used = "Independent t-test"

            # Cohen's d 계산
            pooled_std = np.sqrt(((len(model1_scores)-1)*np.var(model1_scores, ddof=1) +
                                 (len(model2_scores)-1)*np.var(model2_scores, ddof=1)) /
                                (len(model1_scores) + len(model2_scores) - 2))
            cohens_d = (np.mean(model1_scores) - np.mean(model2_scores)) / pooled_std if pooled_std != 0 else 0

        else:
            # Mann-Whitney U 검정 (비모수)
            u_stat, p_value = mannwhitneyu(model1_scores, model2_scores, alternative='two-sided')
            test_used = "Mann-Whitney U test"
            t_stat = u_stat
            cohens_d = None

        # 효과 크기 해석
        if cohens_d is not None:
            if abs(cohens_d) < 0.2:
                effect_interpretation = "무시할 수 있는 효과"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "작은 효과"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "중간 효과"
            else:
                effect_interpretation = "큰 효과"
        else:
            effect_interpretation = "비모수 검정으로 Cohen's d 계산 불가"

        result = {
            'test_type': test_used,
            'model1_name': model1_name,
            'model2_name': model2_name,
            'n_observations_model1': len(model1_scores),
            'n_observations_model2': len(model2_scores),
            'model1_mean': np.mean(model1_scores),
            'model2_mean': np.mean(model2_scores),
            'model1_std': np.std(model1_scores, ddof=1),
            'model2_std': np.std(model2_scores, ddof=1),
            'test_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'alpha': self.alpha,
            'cohens_d': cohens_d,
            'effect_size_interpretation': effect_interpretation,
            'normality_model1_p': p1,
            'normality_model2_p': p2,
            'both_normal': is_normal
        }

        return result

    def multiple_model_comparison(self, model_scores: Dict[str, np.ndarray],
                                paired: bool = True) -> Dict:
        """
        다중 모델 비교 분석

        여러 모델 간의 성능을 쌍별 비교하고 다중 비교 보정 적용

        Args:
            model_scores: 모델명을 키로 하고 점수 배열을 값으로 하는 딕셔너리
            paired: 쌍체 데이터 여부 (기본값: True)

        Returns:
            전체 비교 결과 딕셔너리
        """
        model_names = list(model_scores.keys())
        n_models = len(model_names)
        n_comparisons = n_models * (n_models - 1) // 2

        if n_models < 2:
            raise ValueError("비교를 위해 최소 2개 모델이 필요합니다")

        # 모든 쌍별 비교 수행
        pairwise_results = []
        p_values = []
        comparison_names = []

        for i in range(n_models):
            for j in range(i+1, n_models):
                model1_name = model_names[i]
                model2_name = model_names[j]
                scores1 = model_scores[model1_name]
                scores2 = model_scores[model2_name]

                if paired:
                    # 쌍체 검정 (정규성에 따라 t-test 또는 Wilcoxon 선택)
                    diff_scores = scores1 - scores2
                    if len(diff_scores) >= 3:
                        _, shapiro_p = stats.shapiro(diff_scores)
                        is_normal = shapiro_p > 0.05
                    else:
                        is_normal = False

                    if is_normal:
                        result = self.paired_ttest(scores1, scores2, model1_name, model2_name)
                    else:
                        result = self.wilcoxon_signed_rank_test(scores1, scores2, model1_name, model2_name)
                else:
                    # 독립표본 검정
                    result = self.independent_samples_test(scores1, scores2, model1_name, model2_name)

                pairwise_results.append(result)
                p_values.append(result['p_value'])
                comparison_names.append(f"{model1_name} vs {model2_name}")

        # 다중 비교 보정
        # Bonferroni 보정
        bonferroni_corrected = [p * n_comparisons for p in p_values]
        bonferroni_significant = [p < self.alpha for p in bonferroni_corrected]

        # FDR (False Discovery Rate) 보정 - Benjamini-Hochberg
        reject_fdr, p_corrected_fdr, _, _ = multipletests(p_values, alpha=self.alpha, method='fdr_bh')

        # 전체 분석 결과
        result = {
            'n_models': n_models,
            'n_comparisons': n_comparisons,
            'model_names': model_names,
            'pairwise_results': pairwise_results,
            'multiple_comparison_correction': {
                'original_p_values': p_values,
                'comparison_names': comparison_names,
                'bonferroni': {
                    'corrected_p_values': bonferroni_corrected,
                    'significant': bonferroni_significant,
                    'n_significant': sum(bonferroni_significant)
                },
                'fdr_bh': {
                    'corrected_p_values': p_corrected_fdr.tolist(),
                    'significant': reject_fdr.tolist(),
                    'n_significant': sum(reject_fdr)
                }
            },
            'alpha': self.alpha,
            'recommendation': self._get_multiple_comparison_recommendation(n_comparisons, p_values)
        }

        return result

    def _get_ttest_recommendation(self, p_value: float, is_normal: bool, power: float) -> str:
        """t-검정 결과에 대한 권고사항 생성"""
        if not is_normal:
            return "정규성 가정이 위배되어 Wilcoxon signed-rank test 사용을 권장합니다."
        elif power < 0.8:
            return f"검정력이 낮습니다 (power={power:.3f}). 더 많은 데이터가 필요할 수 있습니다."
        elif p_value < self.alpha:
            return "통계적으로 유의한 차이가 있습니다. 효과 크기도 함께 고려하세요."
        else:
            return "통계적으로 유의한 차이가 없습니다."

    def _get_wilcoxon_recommendation(self, p_value: float, n_non_zero: int) -> str:
        """윌콕슨 검정 결과에 대한 권고사항 생성"""
        if n_non_zero < 5:
            return "non-zero 차이가 너무 적어 결과가 신뢰할 수 없을 수 있습니다."
        elif p_value < self.alpha:
            return "통계적으로 유의한 차이가 있습니다 (비모수 검정)."
        else:
            return "통계적으로 유의한 차이가 없습니다 (비모수 검정)."

    def _get_multiple_comparison_recommendation(self, n_comparisons: int, p_values: List[float]) -> str:
        """다중 비교 결과에 대한 권고사항 생성"""
        min_p = min(p_values)
        uncorrected_significant = sum([p < self.alpha for p in p_values])

        if n_comparisons > 10:
            return f"다중 비교가 많습니다 ({n_comparisons}개). FDR 보정 결과를 우선 고려하세요."
        elif uncorrected_significant == 0:
            return "어떤 모델 쌍에서도 유의한 차이를 발견하지 못했습니다."
        else:
            return f"보정 전 {uncorrected_significant}개 비교에서 유의한 차이 발견. 다중 비교 보정 결과를 확인하세요."

def main():
    """테스트 및 예제 실행"""
    print("📊 통계적 유의성 검증 시스템 테스트")
    print("=" * 50)

    # 테스트 데이터 생성
    np.random.seed(42)

    # 모델 성능 시뮬레이션 (정확도 점수)
    model_a_scores = np.random.normal(0.75, 0.05, 30)  # 평균 75%, 표준편차 5%
    model_b_scores = np.random.normal(0.73, 0.05, 30)  # 평균 73%, 표준편차 5%
    model_c_scores = np.random.normal(0.77, 0.04, 30)  # 평균 77%, 표준편차 4%

    tester = StatisticalSignificanceTester(alpha=0.05)

    # 1. 쌍체 t-검정
    print("\n1. 쌍체 t-검정 (Model A vs Model B)")
    print("-" * 40)
    ttest_result = tester.paired_ttest(model_a_scores, model_b_scores, "Model A", "Model B")
    print(f"p-value: {ttest_result['p_value']:.6f}")
    print(f"Cohen's d: {ttest_result['cohens_d']:.3f}")
    print(f"유의한 차이: {ttest_result['is_significant']}")
    print(f"권고사항: {ttest_result['recommendation']}")

    # 2. 윌콕슨 검정
    print("\n2. 윌콕슨 부호 순위 검정 (Model A vs Model C)")
    print("-" * 40)
    wilcoxon_result = tester.wilcoxon_signed_rank_test(model_a_scores, model_c_scores, "Model A", "Model C")
    print(f"p-value: {wilcoxon_result['p_value']:.6f}")
    print(f"Effect size r: {wilcoxon_result['effect_size_r']:.3f}")
    print(f"유의한 차이: {wilcoxon_result['is_significant']}")

    # 3. 다중 모델 비교
    print("\n3. 다중 모델 비교 (Bonferroni & FDR 보정)")
    print("-" * 40)
    model_scores = {
        'Model A': model_a_scores,
        'Model B': model_b_scores,
        'Model C': model_c_scores
    }

    multi_result = tester.multiple_model_comparison(model_scores, paired=True)
    print(f"총 비교 수: {multi_result['n_comparisons']}")

    bonf_results = multi_result['multiple_comparison_correction']['bonferroni']
    fdr_results = multi_result['multiple_comparison_correction']['fdr_bh']

    print(f"Bonferroni 보정 후 유의한 비교: {bonf_results['n_significant']}")
    print(f"FDR 보정 후 유의한 비교: {fdr_results['n_significant']}")

    print(f"\n권고사항: {multi_result['recommendation']}")

if __name__ == "__main__":
    main()