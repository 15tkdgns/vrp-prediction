#!/usr/bin/env python3
"""
통합 검증 시스템 - 14개 분산 검증 모듈의 핵심 기능 통합
데이터 누출 방지, 통계적 검증, 성능 검증을 단일 시스템으로 통합
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy import stats
import warnings
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """검증 설정 클래스"""
    method: str = "time_series_split"  # time_series_split, purged_time_series_split
    n_splits: int = 5
    test_size: float = 0.2
    purged_window: int = 5
    min_samples_per_split: int = 100
    enable_leak_detection: bool = True
    significance_level: float = 0.05
    strict_mode: bool = True


@dataclass
class ValidationResult:
    """검증 결과 클래스"""
    method: str
    scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    leak_detection: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    passed: bool
    timestamp: str
    details: Dict[str, Any]


class BaseValidator(ABC):
    """검증기 기본 추상 클래스"""

    def __init__(self, config: ValidationConfig):
        self.config = config

    @abstractmethod
    def validate(self, X: np.ndarray, y: np.ndarray, model: Any) -> ValidationResult:
        """검증 수행"""
        pass


class DataLeakageDetector:
    """데이터 누출 탐지기 - 여러 검증 모듈의 기능 통합"""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def detect_temporal_leakage(self, X: pd.DataFrame, y: pd.Series, date_column: str = None) -> Dict[str, Any]:
        """시간적 데이터 누출 탐지"""
        issues = []

        # 1. 날짜 순서 검증
        if date_column and date_column in X.columns:
            dates = pd.to_datetime(X[date_column])
            if not dates.is_monotonic_increasing:
                issues.append("날짜 순서가 올바르지 않습니다.")

        # 2. 미래 정보 탐지
        future_info_features = []
        for col in X.columns:
            if any(keyword in col.lower() for keyword in ['future', 'next', 'tomorrow', 'ahead']):
                future_info_features.append(col)

        if future_info_features:
            issues.append(f"미래 정보가 포함된 특성: {future_info_features}")

        # 3. 타겟 누출 탐지 (높은 상관관계)
        high_correlation_features = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if col != date_column:
                correlation = np.corrcoef(X[col].fillna(0), y)[0, 1]
                if abs(correlation) > 0.95:  # 매우 높은 상관관계
                    high_correlation_features.append((col, correlation))

        if high_correlation_features:
            issues.append(f"의심스러운 높은 상관관계: {high_correlation_features}")

        return {
            'temporal_issues': issues,
            'future_info_features': future_info_features,
            'high_correlation_features': high_correlation_features,
            'leak_detected': len(issues) > 0
        }

    def detect_feature_leakage(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """특성 기반 데이터 누출 탐지"""
        issues = []

        # 1. 완벽한 예측자 탐지
        perfect_predictors = []
        for col in X.select_dtypes(include=[np.number]).columns:
            unique_values = X[col].nunique()
            target_unique = y.nunique()

            # 타겟과 1대1 대응되는 특성
            if unique_values == len(y) or unique_values == target_unique:
                perfect_predictors.append(col)

        if perfect_predictors:
            issues.append(f"완벽한 예측자 의심: {perfect_predictors}")

        # 2. 상수 특성 탐지
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            issues.append(f"상수 특성: {constant_features}")

        # 3. 중복 특성 탐지
        duplicate_features = []
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if np.corrcoef(X[col1].fillna(0), X[col2].fillna(0))[0, 1] > 0.999:
                    duplicate_features.append((col1, col2))

        if duplicate_features:
            issues.append(f"중복 특성: {duplicate_features}")

        return {
            'feature_issues': issues,
            'perfect_predictors': perfect_predictors,
            'constant_features': constant_features,
            'duplicate_features': duplicate_features,
            'leak_detected': len(issues) > 0
        }

    def comprehensive_leak_detection(self, X: pd.DataFrame, y: pd.Series, date_column: str = None) -> Dict[str, Any]:
        """종합적인 데이터 누출 탐지"""
        temporal_results = self.detect_temporal_leakage(X, y, date_column)
        feature_results = self.detect_feature_leakage(X, y)

        overall_leak_detected = temporal_results['leak_detected'] or feature_results['leak_detected']

        return {
            'temporal_leakage': temporal_results,
            'feature_leakage': feature_results,
            'overall_leak_detected': overall_leak_detected,
            'recommendations': self._generate_recommendations(temporal_results, feature_results)
        }

    def _generate_recommendations(self, temporal_results: Dict, feature_results: Dict) -> List[str]:
        """수정 권장사항 생성"""
        recommendations = []

        if temporal_results['future_info_features']:
            recommendations.append("미래 정보가 포함된 특성들을 제거하세요.")

        if temporal_results['high_correlation_features']:
            recommendations.append("타겟과 매우 높은 상관관계를 가진 특성들을 검토하세요.")

        if feature_results['perfect_predictors']:
            recommendations.append("완벽한 예측자로 의심되는 특성들을 제거하세요.")

        if feature_results['constant_features']:
            recommendations.append("정보가 없는 상수 특성들을 제거하세요.")

        if feature_results['duplicate_features']:
            recommendations.append("중복된 특성들 중 하나씩 제거하세요.")

        return recommendations


class StatisticalValidator:
    """통계적 검증기"""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def test_statistical_significance(self, scores1: List[float], scores2: List[float] = None) -> Dict[str, Any]:
        """통계적 유의성 검증"""
        results = {}

        # 1. 단일 모델 성능 검증 (t-test against random baseline)
        random_baseline = 0.5  # 이진 분류 기준
        t_stat, p_value = stats.ttest_1samp(scores1, random_baseline)

        results['single_model_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.config.significance_level,
            'mean_score': np.mean(scores1),
            'baseline': random_baseline
        }

        # 2. 두 모델 비교 (paired t-test)
        if scores2 is not None:
            t_stat_paired, p_value_paired = stats.ttest_rel(scores1, scores2)

            results['paired_model_test'] = {
                't_statistic': t_stat_paired,
                'p_value': p_value_paired,
                'significant': p_value_paired < self.config.significance_level,
                'mean_diff': np.mean(scores1) - np.mean(scores2),
                'model1_better': np.mean(scores1) > np.mean(scores2)
            }

        # 3. 정규성 검정
        shapiro_stat, shapiro_p = stats.shapiro(scores1)
        results['normality_test'] = {
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'normal_distribution': shapiro_p > self.config.significance_level
        }

        # 4. 분산 동질성 검정
        if scores2 is not None:
            levene_stat, levene_p = stats.levene(scores1, scores2)
            results['variance_test'] = {
                'levene_statistic': levene_stat,
                'levene_p_value': levene_p,
                'equal_variances': levene_p > self.config.significance_level
            }

        return results

    def test_overfitting(self, train_scores: List[float], val_scores: List[float]) -> Dict[str, Any]:
        """과적합 검증"""
        train_mean = np.mean(train_scores)
        val_mean = np.mean(val_scores)
        overfitting_ratio = (train_mean - val_mean) / train_mean if train_mean > 0 else 0

        # 임계값 (일반적으로 10% 차이는 허용)
        overfitting_threshold = 0.1

        return {
            'train_mean': train_mean,
            'validation_mean': val_mean,
            'overfitting_ratio': overfitting_ratio,
            'overfitting_detected': overfitting_ratio > overfitting_threshold,
            'threshold': overfitting_threshold,
            'severity': 'high' if overfitting_ratio > 0.2 else 'medium' if overfitting_ratio > 0.1 else 'low'
        }


class PurgedTimeSeriesSplit:
    """데이터 누출 방지를 위한 정제된 시계열 분할"""

    def __init__(self, n_splits: int = 5, test_size: float = 0.2, purge_window: int = 5):
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_window = purge_window

    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None):
        """정제된 시계열 분할"""
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)

        for i in range(self.n_splits):
            # 테스트 세트 시작점 계산
            test_start = n_samples - test_size_samples - (self.n_splits - 1 - i) * test_size_samples // self.n_splits

            # 정제 구간 적용
            train_end = max(0, test_start - self.purge_window)
            test_end = min(n_samples, test_start + test_size_samples)

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            # 최소 샘플 수 확인
            if len(train_indices) > 50 and len(test_indices) > 10:
                yield train_indices, test_indices


class UnifiedCrossValidator(BaseValidator):
    """통합 교차 검증기"""

    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.leak_detector = DataLeakageDetector(config)
        self.stat_validator = StatisticalValidator(config)

    def validate(self, X: np.ndarray, y: np.ndarray, model: Any, X_df: pd.DataFrame = None) -> ValidationResult:
        """통합 검증 수행"""
        # 1. 데이터 누출 탐지
        leak_results = {}
        if self.config.enable_leak_detection and X_df is not None:
            leak_results = self.leak_detector.comprehensive_leak_detection(X_df, pd.Series(y))

        # 2. 교차 검증 수행
        cv_results = self._perform_cross_validation(X, y, model)

        # 3. 통계적 검증
        stat_results = self.stat_validator.test_statistical_significance(cv_results['accuracy'])

        # 4. 과적합 검증
        if 'train_accuracy' in cv_results:
            overfitting_results = self.stat_validator.test_overfitting(
                cv_results['train_accuracy'], cv_results['accuracy']
            )
        else:
            overfitting_results = {}

        # 5. 전체 검증 통과 여부 결정
        passed = self._determine_validation_pass(leak_results, cv_results, stat_results, overfitting_results)

        return ValidationResult(
            method=self.config.method,
            scores=cv_results,
            mean_scores={k: np.mean(v) for k, v in cv_results.items()},
            std_scores={k: np.std(v) for k, v in cv_results.items()},
            leak_detection=leak_results,
            statistical_tests=stat_results,
            passed=passed,
            timestamp=datetime.now().isoformat(),
            details={
                'overfitting': overfitting_results,
                'config': {
                    'method': self.config.method,
                    'n_splits': self.config.n_splits,
                    'test_size': self.config.test_size,
                    'purged_window': self.config.purged_window
                }
            }
        )

    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, model: Any) -> Dict[str, List[float]]:
        """교차 검증 수행"""
        # 분할기 선택
        if self.config.method == "purged_time_series_split":
            splitter = PurgedTimeSeriesSplit(
                n_splits=self.config.n_splits,
                test_size=self.config.test_size,
                purge_window=self.config.purged_window
            )
        else:
            splitter = TimeSeriesSplit(n_splits=self.config.n_splits, test_size=None)

        # 메트릭 저장
        scores = {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'roc_auc': [],
            'train_accuracy': []  # 과적합 검증용
        }

        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 모델 훈련
            model_copy = self._clone_model(model)
            model_copy.fit(X_train, y_train)

            # 예측 및 평가
            y_pred = model_copy.predict(X_test)
            y_pred_proba = model_copy.predict_proba(X_test)[:, 1] if hasattr(model_copy, 'predict_proba') else y_pred

            # 훈련 성능 (과적합 검증용)
            y_train_pred = model_copy.predict(X_train)
            scores['train_accuracy'].append(accuracy_score(y_train, y_train_pred))

            # 테스트 성능
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))

            try:
                scores['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
            except:
                scores['roc_auc'].append(0.5)  # 기본값

        return scores

    def _clone_model(self, model: Any) -> Any:
        """모델 복사 (sklearn clone 기능 활용)"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # sklearn clone이 실패하면 원본 반환
            return model

    def _determine_validation_pass(self, leak_results: Dict, cv_results: Dict, stat_results: Dict, overfitting_results: Dict) -> bool:
        """검증 통과 여부 결정"""
        if self.config.strict_mode:
            # 엄격 모드: 모든 조건 만족
            conditions = [
                not leak_results.get('overall_leak_detected', False),  # 데이터 누출 없음
                stat_results.get('single_model_test', {}).get('significant', False),  # 통계적 유의성
                not overfitting_results.get('overfitting_detected', False),  # 과적합 없음
                np.mean(cv_results.get('accuracy', [0])) > 0.45  # 최소 성능
            ]
            return all(conditions)
        else:
            # 관대한 모드: 핵심 조건만 확인
            return (
                not leak_results.get('overall_leak_detected', False) and
                np.mean(cv_results.get('accuracy', [0])) > 0.40
            )


class ValidationReportGenerator:
    """검증 보고서 생성기"""

    def __init__(self):
        pass

    def generate_report(self, validation_result: ValidationResult, save_path: str = None) -> Dict[str, Any]:
        """검증 보고서 생성"""
        report = {
            'validation_summary': {
                'method': validation_result.method,
                'timestamp': validation_result.timestamp,
                'passed': validation_result.passed,
                'overall_status': 'PASS' if validation_result.passed else 'FAIL'
            },
            'performance_metrics': {
                'mean_accuracy': validation_result.mean_scores.get('accuracy', 0),
                'mean_f1_score': validation_result.mean_scores.get('f1', 0),
                'std_accuracy': validation_result.std_scores.get('accuracy', 0),
                'std_f1_score': validation_result.std_scores.get('f1', 0)
            },
            'data_leakage_check': validation_result.leak_detection,
            'statistical_tests': validation_result.statistical_tests,
            'recommendations': self._generate_recommendations(validation_result)
        }

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        return report

    def _generate_recommendations(self, validation_result: ValidationResult) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        if not validation_result.passed:
            recommendations.append("검증에 실패했습니다. 아래 사항들을 검토하세요.")

        if validation_result.leak_detection.get('overall_leak_detected', False):
            recommendations.extend(validation_result.leak_detection.get('recommendations', []))

        if validation_result.mean_scores.get('accuracy', 0) < 0.5:
            recommendations.append("모델 성능이 낮습니다. 특성 엔지니어링이나 모델 하이퍼파라미터를 조정하세요.")

        overfitting = validation_result.details.get('overfitting', {})
        if overfitting.get('overfitting_detected', False):
            recommendations.append("과적합이 감지되었습니다. 정규화나 교차 검증을 강화하세요.")

        return recommendations


# 팩토리 함수
def create_validator(config: ValidationConfig = None) -> UnifiedCrossValidator:
    """검증기 생성"""
    if config is None:
        config = ValidationConfig()
    return UnifiedCrossValidator(config)


if __name__ == "__main__":
    # 사용 예시
    config = ValidationConfig(
        method="purged_time_series_split",
        n_splits=5,
        enable_leak_detection=True,
        strict_mode=True
    )

    validator = create_validator(config)
    print("✅ 통합 검증 시스템 초기화 완료")
    print(f"검증 방법: {config.method}")
    print(f"데이터 누출 탐지: {config.enable_leak_detection}")
    print(f"엄격 모드: {config.strict_mode}")