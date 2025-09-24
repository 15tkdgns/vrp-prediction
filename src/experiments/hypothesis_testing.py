#!/usr/bin/env python3
"""
🔬 가설 검증 시스템
학술 논문을 위한 체계적인 가설 설정 및 검증 프레임워크

주요 기능:
- 가설 설정 및 관리
- 통계적 검증 수행
- 효과 크기 분석
- 결과 해석 및 보고
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class HypothesisType(Enum):
    """가설 유형 열거형"""
    ONE_SAMPLE = "one_sample"
    TWO_SAMPLE_PAIRED = "two_sample_paired"
    TWO_SAMPLE_INDEPENDENT = "two_sample_independent"
    MULTI_SAMPLE = "multi_sample"
    CORRELATION = "correlation"
    PROPORTION = "proportion"

class AlternativeHypothesis(Enum):
    """대립가설 방향 열거형"""
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"

@dataclass
class Hypothesis:
    """
    가설 정의 클래스

    학술 연구를 위한 명확한 가설 구조화
    """
    id: str
    null_hypothesis: str
    alternative_hypothesis: str
    hypothesis_type: HypothesisType
    alternative_direction: AlternativeHypothesis
    significance_level: float = 0.05
    expected_effect_size: Optional[float] = None
    practical_significance_threshold: Optional[float] = None
    description: Optional[str] = None
    research_question: Optional[str] = None

@dataclass
class HypothesisTestResult:
    """
    가설 검증 결과 클래스
    """
    hypothesis_id: str
    test_statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    is_statistically_significant: bool
    is_practically_significant: bool
    statistical_power: Optional[float]
    sample_size: int
    test_method: str
    interpretation: str
    recommendation: str

class FinancialMLHypothesisFramework:
    """
    금융 머신러닝을 위한 가설 검증 프레임워크

    체계적인 실험 설계 및 가설 검증을 위한 통합 시스템
    """

    def __init__(self):
        """초기화"""
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.results: Dict[str, HypothesisTestResult] = {}

        # 금융 ML을 위한 사전 정의된 가설들
        self._initialize_financial_ml_hypotheses()

    def _initialize_financial_ml_hypotheses(self):
        """금융 ML을 위한 기본 가설들 초기화"""

        # H1: Purged CV가 일반 CV보다 현실적 성능 추정
        self.add_hypothesis(Hypothesis(
            id="H1_purged_cv_effectiveness",
            null_hypothesis="Purged Cross-Validation과 일반 Cross-Validation의 성능 추정에 차이가 없다",
            alternative_hypothesis="Purged Cross-Validation이 일반 Cross-Validation보다 현실적인 성능을 추정한다",
            hypothesis_type=HypothesisType.TWO_SAMPLE_PAIRED,
            alternative_direction=AlternativeHypothesis.LESS,  # Purged CV가 더 낮은(현실적) 성능
            significance_level=0.05,
            expected_effect_size=0.3,
            practical_significance_threshold=0.05,  # 5% 성능 차이
            research_question="데이터 유출 방지가 성능 추정의 현실성에 미치는 영향은?",
            description="Purged CV가 과적합을 방지하여 더 현실적인 성능 추정을 제공하는지 검증"
        ))

        # H2: 앙상블 방법이 단일 모델보다 안정적
        self.add_hypothesis(Hypothesis(
            id="H2_ensemble_stability",
            null_hypothesis="앙상블 모델과 단일 모델의 성능 변동성에 차이가 없다",
            alternative_hypothesis="앙상블 모델이 단일 모델보다 낮은 성능 변동성을 보인다",
            hypothesis_type=HypothesisType.TWO_SAMPLE_PAIRED,
            alternative_direction=AlternativeHypothesis.LESS,  # 앙상블이 더 낮은 변동성
            significance_level=0.05,
            expected_effect_size=0.4,
            practical_significance_threshold=0.1,  # 10% 변동성 감소
            research_question="앙상블 방법이 예측 안정성 향상에 기여하는가?",
            description="다중 모델 앙상블이 단일 모델 대비 성능 안정성을 개선하는지 검증"
        ))

        # H3: 시간 인식 블렌딩이 정적 가중치보다 우수
        self.add_hypothesis(Hypothesis(
            id="H3_time_aware_superiority",
            null_hypothesis="시간 인식 블렌딩과 정적 가중치 앙상블의 성능에 차이가 없다",
            alternative_hypothesis="시간 인식 블렌딩이 정적 가중치 앙상블보다 우수한 성능을 보인다",
            hypothesis_type=HypothesisType.TWO_SAMPLE_PAIRED,
            alternative_direction=AlternativeHypothesis.GREATER,  # 시간 인식이 더 좋은 성능
            significance_level=0.05,
            expected_effect_size=0.2,
            practical_significance_threshold=0.02,  # 2% 성능 개선
            research_question="동적 가중치 조정이 정적 앙상블 대비 성능 향상을 가져오는가?",
            description="시간에 따른 가중치 적응이 고정 가중치 대비 성능 개선에 기여하는지 검증"
        ))

        # H4: 특징 개수 증가가 성능 향상으로 이어지지 않음
        self.add_hypothesis(Hypothesis(
            id="H4_feature_curse_dimensionality",
            null_hypothesis="특징 개수와 모델 성능 간에 상관관계가 없다",
            alternative_hypothesis="특징 개수 증가가 일정 임계값 이후 성능 저하를 야기한다",
            hypothesis_type=HypothesisType.CORRELATION,
            alternative_direction=AlternativeHypothesis.LESS,  # 음의 상관관계
            significance_level=0.05,
            expected_effect_size=-0.3,
            practical_significance_threshold=0.1,
            research_question="차원의 저주가 금융 시계열 예측에서 관찰되는가?",
            description="과도한 특징 사용이 모델 성능에 미치는 부정적 영향 검증"
        ))

        # H5: 데이터 유출 제거가 현실적 성능으로 이어짐
        self.add_hypothesis(Hypothesis(
            id="H5_data_leakage_impact",
            null_hypothesis="데이터 유출 제거 전후의 성능 지표에 차이가 없다",
            alternative_hypothesis="데이터 유출 제거가 현실적(낮은) 성능 지표로 이어진다",
            hypothesis_type=HypothesisType.TWO_SAMPLE_PAIRED,
            alternative_direction=AlternativeHypothesis.LESS,  # 유출 제거 후 더 낮은(현실적) 성능
            significance_level=0.05,
            expected_effect_size=0.8,  # 큰 효과 예상
            practical_significance_threshold=0.2,  # 20% 성능 차이
            research_question="데이터 유출이 성능 과대평가에 미치는 영향의 크기는?",
            description="Look-ahead bias 제거가 모델 성능 평가의 현실성에 미치는 영향 정량화"
        ))

    def add_hypothesis(self, hypothesis: Hypothesis):
        """가설 추가"""
        self.hypotheses[hypothesis.id] = hypothesis

    def test_purged_cv_effectiveness(self, purged_cv_scores: np.ndarray,
                                   regular_cv_scores: np.ndarray) -> HypothesisTestResult:
        """
        H1: Purged CV 효과성 검증

        Args:
            purged_cv_scores: Purged CV 성능 점수들
            regular_cv_scores: 일반 CV 성능 점수들

        Returns:
            가설 검증 결과
        """
        hypothesis = self.hypotheses["H1_purged_cv_effectiveness"]

        # 쌍체 t-검정 (Purged CV가 더 낮은 점수를 가져야 함 - 현실적)
        t_stat, p_value = stats.ttest_rel(regular_cv_scores, purged_cv_scores)

        # 효과 크기 (Cohen's d)
        diff = regular_cv_scores - purged_cv_scores
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0

        # 신뢰구간 (차이에 대한)
        n = len(diff)
        se = stats.sem(diff)
        ci = stats.t.interval(0.95, n-1, loc=np.mean(diff), scale=se)

        # 실용적 유의성
        practical_significant = abs(np.mean(diff)) >= hypothesis.practical_significance_threshold

        # 해석
        if p_value < hypothesis.significance_level and t_stat > 0:
            interpretation = "Purged CV가 통계적으로 유의하게 더 낮은(현실적) 성능을 보고함"
            recommendation = "Purged CV 사용을 강력히 권장. 일반 CV는 과적합된 성능 추정"
        else:
            interpretation = "Purged CV와 일반 CV 간 유의한 차이 없음"
            recommendation = "추가 데이터로 재검증 필요"

        result = HypothesisTestResult(
            hypothesis_id=hypothesis.id,
            test_statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            is_statistically_significant=p_value < hypothesis.significance_level,
            is_practically_significant=practical_significant,
            statistical_power=None,  # 추후 계산 가능
            sample_size=len(purged_cv_scores),
            test_method="Paired t-test",
            interpretation=interpretation,
            recommendation=recommendation
        )

        self.results[hypothesis.id] = result
        return result

    def test_ensemble_stability(self, ensemble_scores: np.ndarray,
                              single_model_scores: np.ndarray) -> HypothesisTestResult:
        """
        H2: 앙상블 안정성 검증

        Args:
            ensemble_scores: 앙상블 모델의 성능 점수들
            single_model_scores: 단일 모델의 성능 점수들

        Returns:
            가설 검증 결과
        """
        hypothesis = self.hypotheses["H2_ensemble_stability"]

        # 분산 비교 (F-test)
        ensemble_var = np.var(ensemble_scores, ddof=1)
        single_var = np.var(single_model_scores, ddof=1)

        f_stat = single_var / ensemble_var if ensemble_var > 0 else np.inf
        df1, df2 = len(single_model_scores) - 1, len(ensemble_scores) - 1
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        # 효과 크기 (변동성 감소 비율)
        effect_size = (single_var - ensemble_var) / single_var if single_var > 0 else 0

        # 신뢰구간 (분산 비율에 대한)
        ci_lower = f_stat / stats.f.ppf(0.975, df1, df2)
        ci_upper = f_stat / stats.f.ppf(0.025, df1, df2)

        # 실용적 유의성
        practical_significant = effect_size >= hypothesis.practical_significance_threshold

        # 해석
        if p_value < hypothesis.significance_level and f_stat > 1:
            interpretation = f"앙상블이 단일 모델보다 {effect_size*100:.1f}% 더 안정적"
            recommendation = "앙상블 방법 사용 권장. 예측 안정성 크게 개선"
        else:
            interpretation = "앙상블과 단일 모델 간 안정성 차이 없음"
            recommendation = "앙상블의 안정성 이점 제한적"

        result = HypothesisTestResult(
            hypothesis_id=hypothesis.id,
            test_statistic=f_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_statistically_significant=p_value < hypothesis.significance_level,
            is_practically_significant=practical_significant,
            statistical_power=None,
            sample_size=min(len(ensemble_scores), len(single_model_scores)),
            test_method="F-test for variance equality",
            interpretation=interpretation,
            recommendation=recommendation
        )

        self.results[hypothesis.id] = result
        return result

    def test_time_aware_superiority(self, time_aware_scores: np.ndarray,
                                  static_weight_scores: np.ndarray) -> HypothesisTestResult:
        """
        H3: 시간 인식 블렌딩 우수성 검증

        Args:
            time_aware_scores: 시간 인식 블렌딩 성능 점수들
            static_weight_scores: 정적 가중치 성능 점수들

        Returns:
            가설 검증 결과
        """
        hypothesis = self.hypotheses["H3_time_aware_superiority"]

        # 일측 쌍체 t-검정 (시간 인식이 더 좋아야 함)
        t_stat, p_value_two_sided = stats.ttest_rel(time_aware_scores, static_weight_scores)
        p_value = p_value_two_sided / 2 if t_stat > 0 else 1 - p_value_two_sided / 2

        # 효과 크기
        diff = time_aware_scores - static_weight_scores
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0

        # 신뢰구간
        n = len(diff)
        se = stats.sem(diff)
        ci = stats.t.interval(0.95, n-1, loc=np.mean(diff), scale=se)

        # 실용적 유의성
        practical_significant = np.mean(diff) >= hypothesis.practical_significance_threshold

        # 해석
        if p_value < hypothesis.significance_level and t_stat > 0:
            improvement = np.mean(diff) * 100
            interpretation = f"시간 인식 블렌딩이 {improvement:.2f}% 성능 향상"
            recommendation = "시간 인식 블렌딩 사용 권장. 동적 가중치의 이점 확인"
        else:
            interpretation = "시간 인식 블렌딩의 유의한 성능 향상 없음"
            recommendation = "정적 가중치도 충분히 효과적. 복잡성 대비 이익 제한적"

        result = HypothesisTestResult(
            hypothesis_id=hypothesis.id,
            test_statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            is_statistically_significant=p_value < hypothesis.significance_level,
            is_practically_significant=practical_significant,
            statistical_power=None,
            sample_size=len(time_aware_scores),
            test_method="One-sided paired t-test",
            interpretation=interpretation,
            recommendation=recommendation
        )

        self.results[hypothesis.id] = result
        return result

    def test_feature_curse_dimensionality(self, feature_counts: np.ndarray,
                                        performance_scores: np.ndarray) -> HypothesisTestResult:
        """
        H4: 차원의 저주 검증

        Args:
            feature_counts: 특징 개수 배열
            performance_scores: 해당 성능 점수 배열

        Returns:
            가설 검증 결과
        """
        hypothesis = self.hypotheses["H4_feature_curse_dimensionality"]

        # 상관분석
        correlation, p_value = stats.pearsonr(feature_counts, performance_scores)

        # 효과 크기 (상관계수 자체가 효과 크기)
        effect_size = correlation

        # 신뢰구간 (Fisher's z-transformation)
        n = len(feature_counts)
        z_r = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se_z = 1 / np.sqrt(n - 3)
        z_critical = stats.norm.ppf(0.975)
        z_lower = z_r - z_critical * se_z
        z_upper = z_r + z_critical * se_z

        # 역변환
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        # 실용적 유의성
        practical_significant = abs(correlation) >= hypothesis.practical_significance_threshold

        # 해석
        if p_value < hypothesis.significance_level and correlation < 0:
            interpretation = f"특징 증가와 성능 간 유의한 음의 상관관계 (r={correlation:.3f})"
            recommendation = "특징 선택 중요. 과도한 특징은 성능 저하 야기"
        elif p_value < hypothesis.significance_level and correlation > 0:
            interpretation = f"특징 증가가 성능 향상에 기여 (r={correlation:.3f})"
            recommendation = "더 많은 특징 사용 고려 가능"
        else:
            interpretation = "특징 개수와 성능 간 유의한 관계 없음"
            recommendation = "특징 개수보다 특징 품질에 집중"

        result = HypothesisTestResult(
            hypothesis_id=hypothesis.id,
            test_statistic=correlation,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_statistically_significant=p_value < hypothesis.significance_level,
            is_practically_significant=practical_significant,
            statistical_power=None,
            sample_size=n,
            test_method="Pearson correlation test",
            interpretation=interpretation,
            recommendation=recommendation
        )

        self.results[hypothesis.id] = result
        return result

    def test_data_leakage_impact(self, pre_leakage_scores: np.ndarray,
                               post_leakage_scores: np.ndarray) -> HypothesisTestResult:
        """
        H5: 데이터 유출 영향 검증

        Args:
            pre_leakage_scores: 데이터 유출 제거 전 성능 점수들
            post_leakage_scores: 데이터 유출 제거 후 성능 점수들

        Returns:
            가설 검증 결과
        """
        hypothesis = self.hypotheses["H5_data_leakage_impact"]

        # 쌍체 t-검정 (유출 제거 후 더 낮은 성능 예상)
        t_stat, p_value = stats.ttest_rel(pre_leakage_scores, post_leakage_scores)

        # 효과 크기
        diff = pre_leakage_scores - post_leakage_scores
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0

        # 신뢰구간
        n = len(diff)
        se = stats.sem(diff)
        ci = stats.t.interval(0.95, n-1, loc=np.mean(diff), scale=se)

        # 실용적 유의성
        practical_significant = abs(np.mean(diff)) >= hypothesis.practical_significance_threshold

        # 해석
        if p_value < hypothesis.significance_level and t_stat > 0:
            overestimation = np.mean(diff) * 100
            interpretation = f"데이터 유출이 성능을 {overestimation:.1f}% 과대평가"
            recommendation = "데이터 유출 제거 필수. 현재 성능이 더 현실적"
        else:
            interpretation = "데이터 유출의 유의한 영향 발견되지 않음"
            recommendation = "현재 파이프라인이 이미 적절히 설계됨"

        result = HypothesisTestResult(
            hypothesis_id=hypothesis.id,
            test_statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            is_statistically_significant=p_value < hypothesis.significance_level,
            is_practically_significant=practical_significant,
            statistical_power=None,
            sample_size=len(pre_leakage_scores),
            test_method="Paired t-test",
            interpretation=interpretation,
            recommendation=recommendation
        )

        self.results[hypothesis.id] = result
        return result

    def generate_hypothesis_report(self) -> str:
        """모든 가설 검증 결과 종합 보고서 생성"""
        if not self.results:
            return "가설 검증 결과가 없습니다. 먼저 가설 검증을 수행하세요."

        report = ["🔬 가설 검증 종합 보고서", "=" * 60, ""]

        # 가설별 결과 요약
        for hypothesis_id, result in self.results.items():
            hypothesis = self.hypotheses[hypothesis_id]

            report.append(f"## {hypothesis_id}: {hypothesis.research_question}")
            report.append("-" * 50)
            report.append(f"**귀무가설**: {hypothesis.null_hypothesis}")
            report.append(f"**대립가설**: {hypothesis.alternative_hypothesis}")
            report.append("")

            # 검증 결과
            report.append("### 검증 결과:")
            report.append(f"- 검정 통계량: {result.test_statistic:.4f}")
            report.append(f"- p-value: {result.p_value:.6f}")
            report.append(f"- 효과 크기: {result.effect_size:.4f}" if result.effect_size is not None else "- 효과 크기: N/A")

            if result.confidence_interval:
                report.append(f"- 95% 신뢰구간: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")

            report.append(f"- 표본 크기: {result.sample_size}")
            report.append(f"- 검정 방법: {result.test_method}")
            report.append("")

            # 유의성 평가
            stat_sig = "✅ 통계적으로 유의" if result.is_statistically_significant else "❌ 통계적으로 유의하지 않음"
            prac_sig = "✅ 실용적으로 유의" if result.is_practically_significant else "❌ 실용적으로 유의하지 않음"

            report.append("### 유의성 평가:")
            report.append(f"- {stat_sig} (α = {hypothesis.significance_level})")
            report.append(f"- {prac_sig}")
            report.append("")

            # 해석 및 권고
            report.append("### 해석:")
            report.append(f"{result.interpretation}")
            report.append("")
            report.append("### 권고사항:")
            report.append(f"{result.recommendation}")
            report.append("")
            report.append("=" * 60)
            report.append("")

        # 전체 결론
        report.append("## 🎯 전체 결론")
        report.append("-" * 30)

        significant_hypotheses = [k for k, v in self.results.items() if v.is_statistically_significant]
        practical_hypotheses = [k for k, v in self.results.items() if v.is_practically_significant]

        report.append(f"- 총 가설 수: {len(self.results)}")
        report.append(f"- 통계적으로 유의한 가설: {len(significant_hypotheses)}개")
        report.append(f"- 실용적으로 유의한 가설: {len(practical_hypotheses)}개")
        report.append("")

        # 핵심 발견사항
        if significant_hypotheses:
            report.append("### 핵심 발견사항:")
            for hyp_id in significant_hypotheses:
                result = self.results[hyp_id]
                hypothesis = self.hypotheses[hyp_id]
                report.append(f"- **{hyp_id}**: {result.interpretation}")

        report.append("")
        report.append("### 방법론적 함의:")
        report.append("이 결과들은 금융 머신러닝 연구에서 다음과 같은 방법론적 지침을 제공합니다:")

        if "H1_purged_cv_effectiveness" in significant_hypotheses:
            report.append("- Purged Cross-Validation 사용 필수")

        if "H2_ensemble_stability" in significant_hypotheses:
            report.append("- 앙상블 방법을 통한 안정성 확보 권장")

        if "H5_data_leakage_impact" in significant_hypotheses:
            report.append("- 데이터 유출 방지가 현실적 성능 평가의 핵심")

        return "\n".join(report)

def main():
    """테스트 및 예제 실행"""
    print("🔬 가설 검증 시스템 테스트")
    print("=" * 60)

    # 가설 검증 프레임워크 초기화
    framework = FinancialMLHypothesisFramework()

    # 시뮬레이션 데이터 생성
    np.random.seed(42)

    # H1: Purged CV vs Regular CV
    regular_cv = np.random.normal(0.85, 0.05, 20)  # 과대평가된 성능
    purged_cv = np.random.normal(0.75, 0.04, 20)   # 현실적 성능

    print("H1: Purged CV 효과성 검증")
    h1_result = framework.test_purged_cv_effectiveness(purged_cv, regular_cv)
    print(f"결과: {h1_result.interpretation}")
    print(f"권고: {h1_result.recommendation}")
    print()

    # H2: Ensemble vs Single Model
    ensemble_scores = np.random.normal(0.75, 0.02, 25)    # 낮은 변동성
    single_scores = np.random.normal(0.75, 0.06, 25)      # 높은 변동성

    print("H2: 앙상블 안정성 검증")
    h2_result = framework.test_ensemble_stability(ensemble_scores, single_scores)
    print(f"결과: {h2_result.interpretation}")
    print(f"권고: {h2_result.recommendation}")
    print()

    # H3: Time-aware vs Static weights
    time_aware = np.random.normal(0.77, 0.03, 20)
    static_weights = np.random.normal(0.75, 0.03, 20)

    print("H3: 시간 인식 블렌딩 우수성 검증")
    h3_result = framework.test_time_aware_superiority(time_aware, static_weights)
    print(f"결과: {h3_result.interpretation}")
    print(f"권고: {h3_result.recommendation}")
    print()

    # H4: Feature count vs Performance
    feature_counts = np.array([5, 10, 15, 20, 25, 30, 35, 40])
    # 차원의 저주 시뮬레이션 (20개 이후 성능 저하)
    performance = 0.8 - 0.005 * np.maximum(0, feature_counts - 20) + np.random.normal(0, 0.02, len(feature_counts))

    print("H4: 차원의 저주 검증")
    h4_result = framework.test_feature_curse_dimensionality(feature_counts, performance)
    print(f"결과: {h4_result.interpretation}")
    print(f"권고: {h4_result.recommendation}")
    print()

    # H5: Data leakage impact
    pre_leakage = np.random.normal(0.95, 0.02, 15)   # 과대평가
    post_leakage = np.random.normal(0.75, 0.03, 15)  # 현실적

    print("H5: 데이터 유출 영향 검증")
    h5_result = framework.test_data_leakage_impact(pre_leakage, post_leakage)
    print(f"결과: {h5_result.interpretation}")
    print(f"권고: {h5_result.recommendation}")
    print()

    # 종합 보고서 생성
    print("📋 종합 보고서:")
    print("=" * 60)
    report = framework.generate_hypothesis_report()
    print(report)

if __name__ == "__main__":
    main()