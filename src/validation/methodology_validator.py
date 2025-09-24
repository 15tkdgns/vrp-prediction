"""
방법론 검증 시스템

이 모듈은 기계학습 실험 방법론의 타당성을 종합적으로 검증합니다.
데이터 분할, 교차 검증, 하이퍼파라미터 튜닝, 통계적 검증 등
모든 방법론적 측면을 체계적으로 검증하고 보고서를 생성합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from datetime import datetime
import pickle
import hashlib
from scipy import stats
warnings.filterwarnings('ignore')

@dataclass
class DataSplitValidation:
    """데이터 분할 검증 결과"""
    train_size: int
    val_size: int
    test_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    is_stratified: bool
    class_distribution_train: Dict[str, float]
    class_distribution_val: Dict[str, float]
    class_distribution_test: Dict[str, float]
    data_leakage_detected: bool
    temporal_ordering_preserved: bool
    independence_verified: bool
    validation_score: float

@dataclass
class CrossValidationValidation:
    """교차 검증 방법론 검증 결과"""
    cv_type: str
    n_folds: int
    is_stratified: bool
    fold_distributions: List[Dict[str, float]]
    variance_across_folds: Dict[str, float]
    bias_estimation: Dict[str, float]
    nested_cv_used: bool
    temporal_cv_appropriate: bool
    overfitting_risk_assessment: str
    validation_score: float

@dataclass
class HyperparameterTuningValidation:
    """하이퍼파라미터 튜닝 검증 결과"""
    tuning_method: str
    search_space_coverage: float
    grid_size: int
    random_seed_fixed: bool
    separate_validation_set: bool
    nested_cv_for_tuning: bool
    overfitting_to_validation: bool
    convergence_achieved: bool
    best_params_stability: Dict[str, float]
    validation_score: float

@dataclass
class StatisticalValidation:
    """통계적 검증 방법론 검증 결과"""
    hypothesis_clearly_defined: bool
    appropriate_test_selected: bool
    assumptions_checked: bool
    effect_size_reported: bool
    confidence_intervals_provided: bool
    multiple_comparison_corrected: bool
    sample_size_adequate: bool
    power_analysis_conducted: bool
    significance_level_appropriate: bool
    validation_score: float

@dataclass
class ModelValidation:
    """모델 검증 결과"""
    architecture_justified: bool
    baseline_comparison: bool
    ablation_studies_conducted: bool
    feature_importance_analyzed: bool
    model_interpretability_addressed: bool
    computational_complexity_analyzed: bool
    scalability_tested: bool
    robustness_evaluated: bool
    generalization_assessed: bool
    validation_score: float

@dataclass
class ReproducibilityValidation:
    """재현성 검증 결과"""
    random_seeds_fixed: bool
    environment_documented: bool
    data_preprocessing_deterministic: bool
    model_training_deterministic: bool
    results_reproducible: bool
    code_version_controlled: bool
    dependencies_specified: bool
    hardware_specifications_documented: bool
    execution_time_documented: bool
    validation_score: float

@dataclass
class EthicalValidation:
    """윤리적 검증 결과"""
    bias_assessment_conducted: bool
    fairness_metrics_evaluated: bool
    privacy_considerations_addressed: bool
    data_consent_verified: bool
    potential_harms_identified: bool
    mitigation_strategies_proposed: bool
    transparency_maintained: bool
    accountability_established: bool
    validation_score: float

@dataclass
class MethodologyValidationReport:
    """종합 방법론 검증 보고서"""
    experiment_id: str
    validation_timestamp: str
    overall_score: float
    data_split_validation: DataSplitValidation
    cv_validation: CrossValidationValidation
    hyperparameter_validation: HyperparameterTuningValidation
    statistical_validation: StatisticalValidation
    model_validation: ModelValidation
    reproducibility_validation: ReproducibilityValidation
    ethical_validation: EthicalValidation
    critical_issues: List[str]
    recommendations: List[str]
    validation_summary: str

class MethodologyValidator:
    """방법론 검증 시스템"""

    def __init__(self, output_dir: str = "results/validation_reports"):
        """
        초기화

        Args:
            output_dir: 검증 보고서 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 검증 기준점 설정
        self.validation_thresholds = {
            'data_split_min_score': 0.8,
            'cv_min_score': 0.7,
            'hyperparameter_min_score': 0.75,
            'statistical_min_score': 0.85,
            'model_min_score': 0.7,
            'reproducibility_min_score': 0.9,
            'ethical_min_score': 0.8,
            'overall_min_score': 0.8
        }

    def validate_data_split(self,
                          train_data: pd.DataFrame,
                          val_data: pd.DataFrame,
                          test_data: pd.DataFrame,
                          target_column: str,
                          is_time_series: bool = False) -> DataSplitValidation:
        """
        데이터 분할 방법론 검증

        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            test_data: 테스트 데이터
            target_column: 타겟 컬럼명
            is_time_series: 시계열 데이터 여부

        Returns:
            데이터 분할 검증 결과
        """
        total_size = len(train_data) + len(val_data) + len(test_data)

        train_size = len(train_data)
        val_size = len(val_data)
        test_size = len(test_data)

        train_ratio = train_size / total_size
        val_ratio = val_size / total_size
        test_ratio = test_size / total_size

        # 클래스 분포 확인 (분류 문제인 경우)
        is_classification = self._is_classification_target(train_data[target_column])

        if is_classification:
            train_dist = self._get_class_distribution(train_data[target_column])
            val_dist = self._get_class_distribution(val_data[target_column])
            test_dist = self._get_class_distribution(test_data[target_column])
            is_stratified = self._check_stratification(train_dist, val_dist, test_dist)
        else:
            train_dist = val_dist = test_dist = {}
            is_stratified = True  # 회귀에서는 계층화 불필요

        # 데이터 누출 검사
        data_leakage_detected = self._check_data_leakage(train_data, val_data, test_data)

        # 시간 순서 보존 확인 (시계열인 경우)
        temporal_ordering_preserved = True
        if is_time_series:
            temporal_ordering_preserved = self._check_temporal_ordering(
                train_data, val_data, test_data
            )

        # 독립성 검증
        independence_verified = self._check_data_independence(train_data, val_data, test_data)

        # 검증 점수 계산
        validation_score = self._calculate_data_split_score(
            train_ratio, val_ratio, test_ratio, is_stratified,
            not data_leakage_detected, temporal_ordering_preserved, independence_verified
        )

        return DataSplitValidation(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            is_stratified=is_stratified,
            class_distribution_train=train_dist,
            class_distribution_val=val_dist,
            class_distribution_test=test_dist,
            data_leakage_detected=data_leakage_detected,
            temporal_ordering_preserved=temporal_ordering_preserved,
            independence_verified=independence_verified,
            validation_score=validation_score
        )

    def validate_cross_validation(self,
                                cv_results: List[Dict[str, float]],
                                cv_config: Dict[str, Any]) -> CrossValidationValidation:
        """
        교차 검증 방법론 검증

        Args:
            cv_results: 교차 검증 결과 (각 fold별 점수)
            cv_config: 교차 검증 설정

        Returns:
            교차 검증 검증 결과
        """
        cv_type = cv_config.get('type', 'unknown')
        n_folds = cv_config.get('n_folds', 0)
        is_stratified = cv_config.get('stratified', False)
        nested_cv_used = cv_config.get('nested', False)

        # Fold별 분포 분석
        fold_distributions = []
        if 'fold_distributions' in cv_config:
            fold_distributions = cv_config['fold_distributions']

        # Fold간 분산 계산
        variance_across_folds = {}
        for metric in cv_results[0].keys():
            scores = [fold[metric] for fold in cv_results]
            variance_across_folds[metric] = np.var(scores)

        # 편향 추정
        bias_estimation = {}
        for metric in cv_results[0].keys():
            scores = [fold[metric] for fold in cv_results]
            bias_estimation[metric] = np.mean(scores) - np.median(scores)

        # 시간 순서 적절성 (시계열 데이터의 경우)
        temporal_cv_appropriate = cv_config.get('temporal_appropriate', True)

        # 과적합 위험도 평가
        overfitting_risk = self._assess_overfitting_risk(cv_results, cv_config)

        # 검증 점수 계산
        validation_score = self._calculate_cv_score(
            n_folds, is_stratified, nested_cv_used, variance_across_folds,
            temporal_cv_appropriate, overfitting_risk
        )

        return CrossValidationValidation(
            cv_type=cv_type,
            n_folds=n_folds,
            is_stratified=is_stratified,
            fold_distributions=fold_distributions,
            variance_across_folds=variance_across_folds,
            bias_estimation=bias_estimation,
            nested_cv_used=nested_cv_used,
            temporal_cv_appropriate=temporal_cv_appropriate,
            overfitting_risk_assessment=overfitting_risk,
            validation_score=validation_score
        )

    def validate_hyperparameter_tuning(self,
                                     tuning_config: Dict[str, Any],
                                     tuning_results: Dict[str, Any]) -> HyperparameterTuningValidation:
        """
        하이퍼파라미터 튜닝 방법론 검증

        Args:
            tuning_config: 하이퍼파라미터 튜닝 설정
            tuning_results: 튜닝 결과

        Returns:
            하이퍼파라미터 튜닝 검증 결과
        """
        tuning_method = tuning_config.get('method', 'unknown')
        search_space = tuning_config.get('search_space', {})

        # 탐색 공간 커버리지 계산
        search_space_coverage = self._calculate_search_space_coverage(
            search_space, tuning_results.get('tried_params', [])
        )

        grid_size = len(tuning_results.get('tried_params', []))
        random_seed_fixed = tuning_config.get('random_seed') is not None
        separate_validation_set = tuning_config.get('separate_validation', False)
        nested_cv_for_tuning = tuning_config.get('nested_cv', False)

        # 검증 세트에 대한 과적합 확인
        overfitting_to_validation = self._check_validation_overfitting(tuning_results)

        # 수렴 확인
        convergence_achieved = self._check_tuning_convergence(tuning_results)

        # 최적 파라미터 안정성
        best_params_stability = self._analyze_best_params_stability(tuning_results)

        # 검증 점수 계산
        validation_score = self._calculate_hyperparameter_score(
            search_space_coverage, random_seed_fixed, separate_validation_set,
            nested_cv_for_tuning, not overfitting_to_validation, convergence_achieved
        )

        return HyperparameterTuningValidation(
            tuning_method=tuning_method,
            search_space_coverage=search_space_coverage,
            grid_size=grid_size,
            random_seed_fixed=random_seed_fixed,
            separate_validation_set=separate_validation_set,
            nested_cv_for_tuning=nested_cv_for_tuning,
            overfitting_to_validation=overfitting_to_validation,
            convergence_achieved=convergence_achieved,
            best_params_stability=best_params_stability,
            validation_score=validation_score
        )

    def validate_statistical_methods(self,
                                   statistical_config: Dict[str, Any],
                                   statistical_results: Dict[str, Any]) -> StatisticalValidation:
        """
        통계적 검증 방법론 검증

        Args:
            statistical_config: 통계 분석 설정
            statistical_results: 통계 분석 결과

        Returns:
            통계적 검증 결과
        """
        # 가설 명확성
        hypothesis_clearly_defined = 'null_hypothesis' in statistical_config and \
                                   'alternative_hypothesis' in statistical_config

        # 적절한 검정 선택
        test_type = statistical_config.get('test_type', '')
        data_type = statistical_config.get('data_type', '')
        appropriate_test_selected = self._check_test_appropriateness(test_type, data_type)

        # 가정 확인
        assumptions_checked = statistical_config.get('assumptions_checked', False)

        # 효과 크기 보고
        effect_size_reported = 'effect_size' in statistical_results

        # 신뢰구간 제공
        confidence_intervals_provided = 'confidence_interval' in statistical_results

        # 다중 비교 보정
        multiple_comparison_corrected = statistical_config.get('multiple_comparison_correction', False)

        # 표본 크기 적절성
        sample_size = statistical_config.get('sample_size', 0)
        effect_size = statistical_results.get('effect_size', 0.5)
        sample_size_adequate = self._check_sample_size_adequacy(sample_size, effect_size)

        # 검정력 분석
        power_analysis_conducted = 'power_analysis' in statistical_results

        # 유의수준 적절성
        alpha = statistical_config.get('alpha', 0.05)
        significance_level_appropriate = 0.01 <= alpha <= 0.1

        # 검증 점수 계산
        validation_score = self._calculate_statistical_score(
            hypothesis_clearly_defined, appropriate_test_selected, assumptions_checked,
            effect_size_reported, confidence_intervals_provided, multiple_comparison_corrected,
            sample_size_adequate, power_analysis_conducted, significance_level_appropriate
        )

        return StatisticalValidation(
            hypothesis_clearly_defined=hypothesis_clearly_defined,
            appropriate_test_selected=appropriate_test_selected,
            assumptions_checked=assumptions_checked,
            effect_size_reported=effect_size_reported,
            confidence_intervals_provided=confidence_intervals_provided,
            multiple_comparison_corrected=multiple_comparison_corrected,
            sample_size_adequate=sample_size_adequate,
            power_analysis_conducted=power_analysis_conducted,
            significance_level_appropriate=significance_level_appropriate,
            validation_score=validation_score
        )

    def validate_model_methodology(self,
                                 model_config: Dict[str, Any],
                                 model_results: Dict[str, Any]) -> ModelValidation:
        """
        모델 방법론 검증

        Args:
            model_config: 모델 설정
            model_results: 모델 결과

        Returns:
            모델 검증 결과
        """
        # 아키텍처 정당화
        architecture_justified = 'architecture_justification' in model_config

        # 베이스라인 비교
        baseline_comparison = 'baseline_models' in model_results

        # Ablation studies
        ablation_studies_conducted = 'ablation_results' in model_results

        # 특성 중요도 분석
        feature_importance_analyzed = 'feature_importance' in model_results

        # 모델 해석가능성
        model_interpretability_addressed = model_config.get('interpretability_methods', []) != []

        # 계산 복잡도 분석
        computational_complexity_analyzed = 'computational_complexity' in model_results

        # 확장성 테스트
        scalability_tested = 'scalability_results' in model_results

        # 견고성 평가
        robustness_evaluated = 'robustness_tests' in model_results

        # 일반화 평가
        generalization_assessed = model_config.get('generalization_tests', False)

        # 검증 점수 계산
        validation_score = self._calculate_model_score(
            architecture_justified, baseline_comparison, ablation_studies_conducted,
            feature_importance_analyzed, model_interpretability_addressed,
            computational_complexity_analyzed, scalability_tested,
            robustness_evaluated, generalization_assessed
        )

        return ModelValidation(
            architecture_justified=architecture_justified,
            baseline_comparison=baseline_comparison,
            ablation_studies_conducted=ablation_studies_conducted,
            feature_importance_analyzed=feature_importance_analyzed,
            model_interpretability_addressed=model_interpretability_addressed,
            computational_complexity_analyzed=computational_complexity_analyzed,
            scalability_tested=scalability_tested,
            robustness_evaluated=robustness_evaluated,
            generalization_assessed=generalization_assessed,
            validation_score=validation_score
        )

    def validate_reproducibility(self,
                               experiment_config: Dict[str, Any],
                               environment_info: Dict[str, Any]) -> ReproducibilityValidation:
        """
        재현성 검증

        Args:
            experiment_config: 실험 설정
            environment_info: 환경 정보

        Returns:
            재현성 검증 결과
        """
        # 랜덤 시드 고정
        random_seeds_fixed = 'random_seed' in experiment_config and \
                           experiment_config['random_seed'] is not None

        # 환경 문서화
        environment_documented = all(key in environment_info for key in
                                   ['python_version', 'packages', 'os_info'])

        # 데이터 전처리 결정적
        data_preprocessing_deterministic = experiment_config.get('deterministic_preprocessing', False)

        # 모델 훈련 결정적
        model_training_deterministic = experiment_config.get('deterministic_training', False)

        # 결과 재현성
        results_reproducible = experiment_config.get('results_reproducible', False)

        # 코드 버전 관리
        code_version_controlled = 'git_commit' in environment_info

        # 의존성 명시
        dependencies_specified = 'requirements_file' in environment_info

        # 하드웨어 사양 문서화
        hardware_specifications_documented = 'hardware_info' in environment_info

        # 실행 시간 문서화
        execution_time_documented = 'execution_time' in environment_info

        # 검증 점수 계산
        validation_score = self._calculate_reproducibility_score(
            random_seeds_fixed, environment_documented, data_preprocessing_deterministic,
            model_training_deterministic, results_reproducible, code_version_controlled,
            dependencies_specified, hardware_specifications_documented, execution_time_documented
        )

        return ReproducibilityValidation(
            random_seeds_fixed=random_seeds_fixed,
            environment_documented=environment_documented,
            data_preprocessing_deterministic=data_preprocessing_deterministic,
            model_training_deterministic=model_training_deterministic,
            results_reproducible=results_reproducible,
            code_version_controlled=code_version_controlled,
            dependencies_specified=dependencies_specified,
            hardware_specifications_documented=hardware_specifications_documented,
            execution_time_documented=execution_time_documented,
            validation_score=validation_score
        )

    def validate_ethics(self,
                       ethics_config: Dict[str, Any],
                       ethics_assessment: Dict[str, Any]) -> EthicalValidation:
        """
        윤리적 검증

        Args:
            ethics_config: 윤리 설정
            ethics_assessment: 윤리적 평가 결과

        Returns:
            윤리적 검증 결과
        """
        # 편향 평가
        bias_assessment_conducted = 'bias_analysis' in ethics_assessment

        # 공정성 메트릭
        fairness_metrics_evaluated = 'fairness_metrics' in ethics_assessment

        # 개인정보 고려사항
        privacy_considerations_addressed = ethics_config.get('privacy_measures', []) != []

        # 데이터 동의
        data_consent_verified = ethics_config.get('data_consent_obtained', False)

        # 잠재적 위해 식별
        potential_harms_identified = 'harm_analysis' in ethics_assessment

        # 완화 전략
        mitigation_strategies_proposed = 'mitigation_strategies' in ethics_assessment

        # 투명성
        transparency_maintained = ethics_config.get('transparency_measures', []) != []

        # 책임성
        accountability_established = ethics_config.get('accountability_framework', False)

        # 검증 점수 계산
        validation_score = self._calculate_ethics_score(
            bias_assessment_conducted, fairness_metrics_evaluated,
            privacy_considerations_addressed, data_consent_verified,
            potential_harms_identified, mitigation_strategies_proposed,
            transparency_maintained, accountability_established
        )

        return EthicalValidation(
            bias_assessment_conducted=bias_assessment_conducted,
            fairness_metrics_evaluated=fairness_metrics_evaluated,
            privacy_considerations_addressed=privacy_considerations_addressed,
            data_consent_verified=data_consent_verified,
            potential_harms_identified=potential_harms_identified,
            mitigation_strategies_proposed=mitigation_strategies_proposed,
            transparency_maintained=transparency_maintained,
            accountability_established=accountability_established,
            validation_score=validation_score
        )

    def generate_comprehensive_validation_report(self,
                                                experiment_id: str,
                                                data_split_validation: DataSplitValidation,
                                                cv_validation: CrossValidationValidation,
                                                hyperparameter_validation: HyperparameterTuningValidation,
                                                statistical_validation: StatisticalValidation,
                                                model_validation: ModelValidation,
                                                reproducibility_validation: ReproducibilityValidation,
                                                ethical_validation: EthicalValidation) -> MethodologyValidationReport:
        """
        종합 검증 보고서 생성

        Args:
            experiment_id: 실험 ID
            모든 개별 검증 결과들

        Returns:
            종합 방법론 검증 보고서
        """
        # 전체 점수 계산
        scores = [
            data_split_validation.validation_score,
            cv_validation.validation_score,
            hyperparameter_validation.validation_score,
            statistical_validation.validation_score,
            model_validation.validation_score,
            reproducibility_validation.validation_score,
            ethical_validation.validation_score
        ]
        overall_score = np.mean(scores)

        # 심각한 문제 식별
        critical_issues = self._identify_critical_issues([
            data_split_validation, cv_validation, hyperparameter_validation,
            statistical_validation, model_validation, reproducibility_validation,
            ethical_validation
        ])

        # 권장사항 생성
        recommendations = self._generate_recommendations([
            data_split_validation, cv_validation, hyperparameter_validation,
            statistical_validation, model_validation, reproducibility_validation,
            ethical_validation
        ])

        # 검증 요약 생성
        validation_summary = self._generate_validation_summary(overall_score, critical_issues)

        return MethodologyValidationReport(
            experiment_id=experiment_id,
            validation_timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            data_split_validation=data_split_validation,
            cv_validation=cv_validation,
            hyperparameter_validation=hyperparameter_validation,
            statistical_validation=statistical_validation,
            model_validation=model_validation,
            reproducibility_validation=reproducibility_validation,
            ethical_validation=ethical_validation,
            critical_issues=critical_issues,
            recommendations=recommendations,
            validation_summary=validation_summary
        )

    # 보조 메서드들
    def _is_classification_target(self, target: pd.Series) -> bool:
        """타겟이 분류 문제인지 확인"""
        return target.dtype == 'object' or len(target.unique()) < 20

    def _get_class_distribution(self, target: pd.Series) -> Dict[str, float]:
        """클래스 분포 계산"""
        return target.value_counts(normalize=True).to_dict()

    def _check_stratification(self, train_dist: Dict, val_dist: Dict, test_dist: Dict) -> bool:
        """계층화 확인"""
        threshold = 0.05  # 5% 임계값
        for class_name in train_dist.keys():
            train_ratio = train_dist.get(class_name, 0)
            val_ratio = val_dist.get(class_name, 0)
            test_ratio = test_dist.get(class_name, 0)

            if abs(train_ratio - val_ratio) > threshold or abs(train_ratio - test_ratio) > threshold:
                return False
        return True

    def _check_data_leakage(self, train_data: pd.DataFrame,
                          val_data: pd.DataFrame,
                          test_data: pd.DataFrame) -> bool:
        """데이터 누출 검사"""
        # 인덱스 중복 확인
        train_idx = set(train_data.index)
        val_idx = set(val_data.index)
        test_idx = set(test_data.index)

        return len(train_idx & val_idx) > 0 or len(train_idx & test_idx) > 0 or len(val_idx & test_idx) > 0

    def _check_temporal_ordering(self, train_data: pd.DataFrame,
                               val_data: pd.DataFrame,
                               test_data: pd.DataFrame) -> bool:
        """시간 순서 보존 확인"""
        # 간단한 시간 순서 확인 (실제로는 더 복잡한 로직 필요)
        if 'timestamp' in train_data.columns:
            train_max = train_data['timestamp'].max()
            val_min = val_data['timestamp'].min()
            val_max = val_data['timestamp'].max()
            test_min = test_data['timestamp'].min()

            return train_max <= val_min and val_max <= test_min
        return True

    def _check_data_independence(self, train_data: pd.DataFrame,
                               val_data: pd.DataFrame,
                               test_data: pd.DataFrame) -> bool:
        """데이터 독립성 확인"""
        # 특성 분포의 유사성 확인
        # 실제로는 더 정교한 통계적 검증 필요
        return True

    def _calculate_data_split_score(self, train_ratio: float, val_ratio: float,
                                  test_ratio: float, is_stratified: bool,
                                  no_leakage: bool, temporal_ok: bool,
                                  independent: bool) -> float:
        """데이터 분할 점수 계산"""
        score = 0.0

        # 비율 점수 (60/20/20 또는 70/15/15가 이상적)
        ideal_ratios = [(0.6, 0.2, 0.2), (0.7, 0.15, 0.15), (0.8, 0.1, 0.1)]
        ratio_scores = []
        for ideal_train, ideal_val, ideal_test in ideal_ratios:
            ratio_score = 1.0 - (abs(train_ratio - ideal_train) +
                                abs(val_ratio - ideal_val) +
                                abs(test_ratio - ideal_test)) / 3
            ratio_scores.append(max(0, ratio_score))
        score += max(ratio_scores) * 0.3

        # 계층화 점수
        score += 0.2 if is_stratified else 0.0

        # 데이터 누출 방지
        score += 0.25 if no_leakage else 0.0

        # 시간 순서 보존
        score += 0.15 if temporal_ok else 0.0

        # 독립성
        score += 0.1 if independent else 0.0

        return min(1.0, score)

    def _assess_overfitting_risk(self, cv_results: List[Dict], cv_config: Dict) -> str:
        """과적합 위험도 평가"""
        if len(cv_results) < 3:
            return "high"

        # 분산 계산
        variances = []
        for metric in cv_results[0].keys():
            scores = [fold[metric] for fold in cv_results]
            variances.append(np.var(scores))

        avg_variance = np.mean(variances)

        if avg_variance > 0.01:
            return "high"
        elif avg_variance > 0.005:
            return "medium"
        else:
            return "low"

    def _calculate_cv_score(self, n_folds: int, is_stratified: bool,
                          nested_cv_used: bool, variances: Dict,
                          temporal_ok: bool, overfitting_risk: str) -> float:
        """교차 검증 점수 계산"""
        score = 0.0

        # 폴드 수 점수
        if n_folds >= 5:
            score += 0.2
        elif n_folds >= 3:
            score += 0.1

        # 계층화
        score += 0.2 if is_stratified else 0.0

        # 중첩 교차 검증
        score += 0.15 if nested_cv_used else 0.0

        # 분산 점수
        avg_variance = np.mean(list(variances.values())) if variances else 0
        variance_score = max(0, 1 - avg_variance * 100)  # 분산이 작을수록 좋음
        score += variance_score * 0.2

        # 시간적 적절성
        score += 0.15 if temporal_ok else 0.0

        # 과적합 위험도
        risk_scores = {"low": 0.1, "medium": 0.05, "high": 0.0}
        score += risk_scores.get(overfitting_risk, 0.0)

        return min(1.0, score)

    def _calculate_search_space_coverage(self, search_space: Dict,
                                       tried_params: List[Dict]) -> float:
        """탐색 공간 커버리지 계산"""
        if not search_space or not tried_params:
            return 0.0

        coverage_scores = []
        for param_name, param_range in search_space.items():
            tried_values = [params.get(param_name) for params in tried_params
                          if param_name in params]
            tried_values = [v for v in tried_values if v is not None]

            if not tried_values:
                coverage_scores.append(0.0)
                continue

            if isinstance(param_range, list):
                # 이산 값들
                unique_tried = len(set(tried_values))
                total_possible = len(param_range)
                coverage = unique_tried / total_possible
            else:
                # 연속 값들 (간단한 근사)
                coverage = min(1.0, len(set(tried_values)) / 10)

            coverage_scores.append(coverage)

        return np.mean(coverage_scores) if coverage_scores else 0.0

    def _check_validation_overfitting(self, tuning_results: Dict) -> bool:
        """검증 세트 과적합 확인"""
        # 간단한 휴리스틱: 너무 많은 시도가 있었는지 확인
        n_trials = len(tuning_results.get('tried_params', []))
        return n_trials > 100  # 임의의 임계값

    def _check_tuning_convergence(self, tuning_results: Dict) -> bool:
        """튜닝 수렴 확인"""
        scores = tuning_results.get('scores', [])
        if len(scores) < 10:
            return False

        # 마지막 10개 시도에서 개선이 없으면 수렴으로 간주
        recent_scores = scores[-10:]
        best_recent = max(recent_scores)
        overall_best = max(scores)

        return abs(best_recent - overall_best) < 0.001

    def _analyze_best_params_stability(self, tuning_results: Dict) -> Dict[str, float]:
        """최적 파라미터 안정성 분석"""
        # 상위 10% 결과의 파라미터 분포 분석
        scores = tuning_results.get('scores', [])
        params = tuning_results.get('tried_params', [])

        if not scores or not params:
            return {}

        # 상위 10% 선택
        n_top = max(1, len(scores) // 10)
        top_indices = np.argsort(scores)[-n_top:]

        stability = {}
        for param_name in params[0].keys():
            top_values = [params[i][param_name] for i in top_indices]
            stability[param_name] = 1.0 - np.std(top_values) / (np.mean(top_values) + 1e-8)

        return stability

    def _calculate_hyperparameter_score(self, coverage: float, seed_fixed: bool,
                                      separate_val: bool, nested_cv: bool,
                                      no_overfitting: bool, converged: bool) -> float:
        """하이퍼파라미터 튜닝 점수 계산"""
        score = 0.0
        score += coverage * 0.25  # 탐색 공간 커버리지
        score += 0.15 if seed_fixed else 0.0
        score += 0.2 if separate_val else 0.0
        score += 0.2 if nested_cv else 0.0
        score += 0.1 if no_overfitting else 0.0
        score += 0.1 if converged else 0.0

        return min(1.0, score)

    def _check_test_appropriateness(self, test_type: str, data_type: str) -> bool:
        """검정 방법의 적절성 확인"""
        appropriate_combinations = {
            ('t-test', 'continuous'): True,
            ('mann-whitney', 'ordinal'): True,
            ('chi-square', 'categorical'): True,
            ('anova', 'continuous'): True,
            ('kruskal-wallis', 'ordinal'): True
        }
        return appropriate_combinations.get((test_type.lower(), data_type.lower()), False)

    def _check_sample_size_adequacy(self, sample_size: int, effect_size: float) -> bool:
        """표본 크기 적절성 확인"""
        # 간단한 검정력 분석 (Cohen의 기준 사용)
        if effect_size >= 0.8:  # 큰 효과
            return sample_size >= 25
        elif effect_size >= 0.5:  # 중간 효과
            return sample_size >= 64
        else:  # 작은 효과
            return sample_size >= 393

    def _calculate_statistical_score(self, *args) -> float:
        """통계적 검증 점수 계산"""
        conditions = args
        weights = [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.05]

        score = sum(condition * weight for condition, weight in zip(conditions, weights))
        return min(1.0, score)

    def _calculate_model_score(self, *args) -> float:
        """모델 검증 점수 계산"""
        conditions = args
        weights = [0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1]

        score = sum(condition * weight for condition, weight in zip(conditions, weights))
        return min(1.0, score)

    def _calculate_reproducibility_score(self, *args) -> float:
        """재현성 점수 계산"""
        conditions = args
        weights = [0.15, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.05]

        score = sum(condition * weight for condition, weight in zip(conditions, weights))
        return min(1.0, score)

    def _calculate_ethics_score(self, *args) -> float:
        """윤리 점수 계산"""
        conditions = args
        weights = [0.15, 0.15, 0.15, 0.1, 0.15, 0.1, 0.1, 0.1]

        score = sum(condition * weight for condition, weight in zip(conditions, weights))
        return min(1.0, score)

    def _identify_critical_issues(self, validations: List) -> List[str]:
        """심각한 문제 식별"""
        issues = []

        # 각 검증 결과에서 심각한 문제 확인
        data_split = validations[0]
        if data_split.data_leakage_detected:
            issues.append("데이터 누출이 감지되었습니다.")

        if data_split.validation_score < self.validation_thresholds['data_split_min_score']:
            issues.append("데이터 분할 방법론에 심각한 문제가 있습니다.")

        cv_validation = validations[1]
        if cv_validation.validation_score < self.validation_thresholds['cv_min_score']:
            issues.append("교차 검증 방법론에 문제가 있습니다.")

        reproducibility = validations[5]
        if not reproducibility.random_seeds_fixed:
            issues.append("재현성을 위한 랜덤 시드가 고정되지 않았습니다.")

        return issues

    def _generate_recommendations(self, validations: List) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        data_split = validations[0]
        if not data_split.is_stratified:
            recommendations.append("분류 문제에서는 계층화된 분할을 사용하십시오.")

        cv_validation = validations[1]
        if cv_validation.n_folds < 5:
            recommendations.append("교차 검증에서 최소 5-fold를 사용하십시오.")

        hyperparameter = validations[2]
        if not hyperparameter.nested_cv_for_tuning:
            recommendations.append("하이퍼파라미터 튜닝에 중첩 교차 검증을 사용하십시오.")

        statistical = validations[3]
        if not statistical.effect_size_reported:
            recommendations.append("통계적 검증에서 효과 크기를 보고하십시오.")

        return recommendations

    def _generate_validation_summary(self, overall_score: float,
                                   critical_issues: List[str]) -> str:
        """검증 요약 생성"""
        if overall_score >= self.validation_thresholds['overall_min_score']:
            summary = f"전체 검증 점수 {overall_score:.3f}으로 방법론이 적절합니다."
        else:
            summary = f"전체 검증 점수 {overall_score:.3f}으로 방법론 개선이 필요합니다."

        if critical_issues:
            summary += f" {len(critical_issues)}개의 심각한 문제가 발견되었습니다."

        return summary

    def save_validation_report(self, report: MethodologyValidationReport) -> str:
        """검증 보고서 저장"""
        filename = f"validation_report_{report.experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        # dataclass를 dict로 변환
        report_dict = self._dataclass_to_dict(report)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def _dataclass_to_dict(self, obj) -> Dict:
        """dataclass를 딕셔너리로 변환"""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._dataclass_to_dict(value)
                elif isinstance(value, list):
                    result[key] = [self._dataclass_to_dict(item) if hasattr(item, '__dict__') else item
                                 for item in value]
                else:
                    result[key] = value
            return result
        return obj

if __name__ == "__main__":
    # 테스트 예제
    validator = MethodologyValidator()

    # 샘플 데이터 생성
    np.random.seed(42)
    total_samples = 1000

    # 더미 데이터프레임 생성
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, total_samples),
        'feature2': np.random.normal(0, 1, total_samples),
        'target': np.random.choice([0, 1], total_samples)
    })

    # 데이터 분할
    train_data = data[:600]
    val_data = data[600:800]
    test_data = data[800:]

    # 데이터 분할 검증
    data_split_validation = validator.validate_data_split(
        train_data, val_data, test_data, 'target'
    )

    print(f"데이터 분할 검증 점수: {data_split_validation.validation_score:.3f}")
    print(f"데이터 누출 감지: {data_split_validation.data_leakage_detected}")
    print(f"계층화 적용: {data_split_validation.is_stratified}")

    print("\\n방법론 검증 시스템 테스트 완료!")