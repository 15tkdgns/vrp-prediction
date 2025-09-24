#!/usr/bin/env python3
"""
PurgedKFold 및 중첩 교차 검증 시스템
금융 시계열 데이터 유출 방지를 위한 고급 검증 시스템
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator, Union
from dataclasses import dataclass
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, f1_score
import warnings
import logging
from datetime import datetime, timedelta
import itertools

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class PurgedFoldConfig:
    """PurgedKFold 설정 클래스"""
    n_splits: int = 5
    purge_size: int = 5  # 훈련-검증 간 제거할 샘플 수
    embargo_size: int = 5  # 검증 후 추가 제거할 샘플 수
    min_samples_per_fold: int = 50  # 폴드당 최소 샘플 수


class PurgedKFold:
    """
    금융 시계열용 정제된 K-Fold 교차 검증
    데이터 유출 방지를 위한 Purging과 Embargo 적용
    """

    def __init__(self, config: PurgedFoldConfig = None):
        """
        Args:
            config: PurgedKFold 설정
        """
        self.config = config or PurgedFoldConfig()

    def split(self, X: np.ndarray, y: np.ndarray = None,
              groups: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Purged K-Fold 분할 수행

        Args:
            X: 특성 데이터
            y: 타겟 데이터
            groups: 그룹 정보 (시계열에서는 시간 정보)

        Yields:
            (train_indices, test_indices): 훈련/검증 인덱스 쌍
        """
        n_samples = len(X)
        fold_size = n_samples // self.config.n_splits

        for i in range(self.config.n_splits):
            # 검증 세트 인덱스 계산
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)

            if i == self.config.n_splits - 1:  # 마지막 폴드
                test_end = n_samples

            test_indices = np.arange(test_start, test_end)

            # 훈련 세트 인덱스 계산 (Purging 적용)
            # 검증 세트 이전 데이터에서 purge_size만큼 제거
            train_end = max(0, test_start - self.config.purge_size)

            # 검증 세트 이후 데이터에서 embargo_size만큼 제거
            train_start_after = min(n_samples, test_end + self.config.embargo_size)

            # 전체 훈련 인덱스 생성
            train_indices_before = np.arange(0, train_end)
            train_indices_after = np.arange(train_start_after, n_samples)
            train_indices = np.concatenate([train_indices_before, train_indices_after])

            # 최소 샘플 수 확인
            if (len(train_indices) >= self.config.min_samples_per_fold and
                len(test_indices) >= self.config.min_samples_per_fold):
                yield train_indices, test_indices

    def get_n_splits(self, X: np.ndarray = None, y: np.ndarray = None,
                     groups: np.ndarray = None) -> int:
        """분할 수 반환"""
        return self.config.n_splits


class CombinatorialPurgedCV:
    """
    조합 교차 검증 (CPCV)
    다수의 백테스트 경로를 생성하여 모델의 강건성 평가
    """

    def __init__(self, n_groups: int = 10, n_test_groups: int = 2,
                 purge_size: int = 5, embargo_size: int = 5):
        """
        Args:
            n_groups: 전체 그룹 수
            n_test_groups: 테스트용 그룹 수
            purge_size: 정제할 샘플 수
            embargo_size: 엠바고할 샘플 수
        """
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_size = purge_size
        self.embargo_size = embargo_size

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        조합적 분할 수행

        Args:
            X: 특성 데이터
            y: 타겟 데이터

        Yields:
            (train_indices, test_indices): 훈련/검증 인덱스 쌍
        """
        n_samples = len(X)
        group_size = n_samples // self.n_groups

        # 모든 그룹 조합 생성
        groups = list(range(self.n_groups))
        test_group_combinations = list(itertools.combinations(groups, self.n_test_groups))

        for test_groups in test_group_combinations:
            test_indices = []
            train_indices = []

            # 테스트 그룹 인덱스 생성
            for group_idx in test_groups:
                start_idx = group_idx * group_size
                end_idx = min((group_idx + 1) * group_size, n_samples)
                if group_idx == self.n_groups - 1:  # 마지막 그룹
                    end_idx = n_samples
                test_indices.extend(range(start_idx, end_idx))

            # 훈련 그룹 인덱스 생성 (정제 적용)
            for group_idx in groups:
                if group_idx not in test_groups:
                    start_idx = group_idx * group_size
                    end_idx = min((group_idx + 1) * group_size, n_samples)
                    if group_idx == self.n_groups - 1:
                        end_idx = n_samples

                    # 테스트 그룹과 인접한 경우 정제 적용
                    should_purge = any(abs(group_idx - test_group) <= 1
                                     for test_group in test_groups)

                    if not should_purge:
                        train_indices.extend(range(start_idx, end_idx))

            test_indices = np.array(sorted(test_indices))
            train_indices = np.array(sorted(train_indices))

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X: np.ndarray = None, y: np.ndarray = None) -> int:
        """분할 수 반환"""
        from math import comb
        return comb(self.n_groups, self.n_test_groups)


class NestedCrossValidator:
    """
    중첩 교차 검증 시스템
    내부 루프: 하이퍼파라미터 최적화
    외부 루프: 모델 성능 객관적 평가
    """

    def __init__(self, estimator: BaseEstimator, param_grid: Dict[str, List],
                 inner_cv: Union[PurgedKFold, CombinatorialPurgedCV] = None,
                 outer_cv: Union[PurgedKFold, CombinatorialPurgedCV] = None,
                 scoring: str = 'neg_log_loss',
                 n_jobs: int = 1, verbose: bool = True):
        """
        Args:
            estimator: 기본 추정기
            param_grid: 하이퍼파라미터 그리드
            inner_cv: 내부 교차 검증기
            outer_cv: 외부 교차 검증기
            scoring: 성과 지표
            n_jobs: 병렬 작업 수
            verbose: 상세 출력 여부
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.inner_cv = inner_cv or PurgedKFold(PurgedFoldConfig(n_splits=3))
        self.outer_cv = outer_cv or PurgedKFold(PurgedFoldConfig(n_splits=5))
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

        # 결과 저장용
        self.cv_results_ = {}
        self.best_estimators_ = []
        self.outer_scores_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NestedCrossValidator':
        """
        중첩 교차 검증 수행

        Args:
            X: 특성 데이터
            y: 타겟 데이터

        Returns:
            자기 자신 (fitted)
        """
        if self.verbose:
            print("=== 중첩 교차 검증 시작 ===")
            print(f"외부 폴드 수: {self.outer_cv.get_n_splits(X, y)}")
            print(f"내부 폴드 수: {self.inner_cv.get_n_splits(X, y)}")

        outer_fold = 0
        for train_outer, test_outer in self.outer_cv.split(X, y):
            outer_fold += 1
            if self.verbose:
                print(f"\n--- 외부 폴드 {outer_fold} ---")
                print(f"훈련 샘플: {len(train_outer)}, 테스트 샘플: {len(test_outer)}")

            X_train_outer, X_test_outer = X[train_outer], X[test_outer]
            y_train_outer, y_test_outer = y[train_outer], y[test_outer]

            # 내부 루프: 하이퍼파라미터 최적화
            grid_search = GridSearchCV(
                estimator=clone(self.estimator),
                param_grid=self.param_grid,
                cv=self.inner_cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                refit=True
            )

            grid_search.fit(X_train_outer, y_train_outer)
            self.best_estimators_.append(grid_search.best_estimator_)

            # 외부 루프: 객관적 성능 평가
            y_pred = grid_search.predict(X_test_outer)

            if self.scoring == 'neg_log_loss':
                y_pred_proba = grid_search.predict_proba(X_test_outer)[:, 1]
                score = -log_loss(y_test_outer, y_pred_proba)
            elif self.scoring == 'accuracy':
                score = accuracy_score(y_test_outer, y_pred)
            elif self.scoring == 'f1':
                score = f1_score(y_test_outer, y_pred, average='weighted')
            else:
                score = grid_search.score(X_test_outer, y_test_outer)

            self.outer_scores_.append(score)

            if self.verbose:
                print(f"최적 파라미터: {grid_search.best_params_}")
                print(f"외부 점수: {score:.4f}")

        # 결과 요약
        self.cv_results_ = {
            'outer_scores': self.outer_scores_,
            'mean_score': np.mean(self.outer_scores_),
            'std_score': np.std(self.outer_scores_),
            'best_estimators': self.best_estimators_,
            'n_outer_folds': len(self.outer_scores_)
        }

        if self.verbose:
            print(f"\n=== 중첩 교차 검증 완료 ===")
            print(f"평균 점수: {self.cv_results_['mean_score']:.4f} ± {self.cv_results_['std_score']:.4f}")

        return self

    def get_best_estimator(self) -> BaseEstimator:
        """최고 성능 추정기 반환"""
        if not self.best_estimators_:
            raise ValueError("중첩 교차 검증이 아직 수행되지 않았습니다.")

        # 가장 높은 점수를 가진 추정기 반환
        best_idx = np.argmax(self.outer_scores_)
        return self.best_estimators_[best_idx]

    def get_ensemble_prediction(self, X: np.ndarray) -> np.ndarray:
        """앙상블 예측 (모든 폴드의 모델 평균)"""
        if not self.best_estimators_:
            raise ValueError("중첩 교차 검증이 아직 수행되지 않았습니다.")

        predictions = []
        for estimator in self.best_estimators_:
            if hasattr(estimator, 'predict_proba'):
                pred = estimator.predict_proba(X)[:, 1]
            else:
                pred = estimator.predict(X)
            predictions.append(pred)

        return np.mean(predictions, axis=0)


class WalkForwardValidator:
    """
    전진 분석 (Walk-Forward) 검증
    시계열 데이터의 시간 순서를 엄격히 준수하는 검증 방법
    """

    def __init__(self, train_window: int = 252, test_window: int = 21,
                 step_size: int = 21, min_train_size: int = 100):
        """
        Args:
            train_window: 훈련 윈도우 크기 (일수)
            test_window: 테스트 윈도우 크기 (일수)
            step_size: 이동 단계 크기 (일수)
            min_train_size: 최소 훈련 샘플 수
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        전진 분석 분할 수행

        Args:
            X: 특성 데이터
            y: 타겟 데이터

        Yields:
            (train_indices, test_indices): 훈련/검증 인덱스 쌍
        """
        n_samples = len(X)
        current_pos = self.train_window

        while current_pos + self.test_window <= n_samples:
            # 훈련 세트 인덱스
            train_start = max(0, current_pos - self.train_window)
            train_end = current_pos
            train_indices = np.arange(train_start, train_end)

            # 테스트 세트 인덱스
            test_start = current_pos
            test_end = min(current_pos + self.test_window, n_samples)
            test_indices = np.arange(test_start, test_end)

            # 최소 훈련 샘플 수 확인
            if len(train_indices) >= self.min_train_size:
                yield train_indices, test_indices

            current_pos += self.step_size

    def get_n_splits(self, X: np.ndarray = None, y: np.ndarray = None) -> int:
        """분할 수 반환"""
        if X is None:
            return 0

        n_samples = len(X)
        splits = 0
        current_pos = self.train_window

        while current_pos + self.test_window <= n_samples:
            if current_pos - self.train_window >= 0:
                splits += 1
            current_pos += self.step_size

        return splits


class ValidationReportGenerator:
    """검증 결과 보고서 생성기"""

    def __init__(self):
        pass

    def generate_purged_cv_report(self, cv_results: Dict[str, Any],
                                save_path: str = None) -> Dict[str, Any]:
        """PurgedKFold 검증 보고서 생성"""
        report = {
            'validation_type': 'PurgedKFold',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'mean_score': cv_results.get('mean_score', 0),
                'std_score': cv_results.get('std_score', 0),
                'n_folds': cv_results.get('n_outer_folds', 0),
                'all_scores': cv_results.get('outer_scores', [])
            },
            'statistical_significance': self._test_significance(cv_results.get('outer_scores', [])),
            'recommendations': self._generate_recommendations(cv_results)
        }

        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        return report

    def _test_significance(self, scores: List[float]) -> Dict[str, Any]:
        """통계적 유의성 검정"""
        if len(scores) < 2:
            return {'test': 'insufficient_data'}

        from scipy import stats

        # t-test against random baseline (0.5 for binary classification)
        baseline = 0.5 if all(0 <= s <= 1 for s in scores) else 0
        t_stat, p_value = stats.ttest_1samp(scores, baseline)

        return {
            'test_type': 't_test_one_sample',
            'baseline': baseline,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': (np.mean(scores) - baseline) / np.std(scores) if np.std(scores) > 0 else 0
        }

    def _generate_recommendations(self, cv_results: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        mean_score = cv_results.get('mean_score', 0)
        std_score = cv_results.get('std_score', 0)

        if std_score > 0.1:
            recommendations.append("성능 분산이 큽니다. 모델 안정성을 개선하세요.")

        if mean_score < 0.6:
            recommendations.append("평균 성능이 낮습니다. 특성 엔지니어링이나 모델 선택을 검토하세요.")

        if len(cv_results.get('outer_scores', [])) < 5:
            recommendations.append("더 많은 폴드로 검증하여 신뢰도를 높이세요.")

        return recommendations


# 사용 예시 및 테스트
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000

    # 시계열 특성을 가진 데이터 생성
    X = np.random.randn(n_samples, 10)
    # 약간의 시계열 의존성 추가
    for i in range(1, n_samples):
        X[i] += 0.1 * X[i-1]

    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

    print("=== PurgedKFold 및 중첩 교차 검증 테스트 ===")

    # 1. PurgedKFold 테스트
    print("\n1. PurgedKFold 테스트")
    purged_config = PurgedFoldConfig(n_splits=5, purge_size=10, embargo_size=10)
    purged_cv = PurgedKFold(purged_config)

    fold_count = 0
    for train_idx, test_idx in purged_cv.split(X, y):
        fold_count += 1
        print(f"폴드 {fold_count}: 훈련={len(train_idx)}, 테스트={len(test_idx)}")

    # 2. CombinatorialPurgedCV 테스트
    print("\n2. CombinatorialPurgedCV 테스트")
    cpcv = CombinatorialPurgedCV(n_groups=8, n_test_groups=2)
    cpcv_splits = list(cpcv.split(X, y))
    print(f"CPCV 분할 수: {len(cpcv_splits)}")

    # 3. 중첩 교차 검증 테스트
    print("\n3. 중첩 교차 검증 테스트")
    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [3, 5]
    }

    nested_cv = NestedCrossValidator(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        inner_cv=PurgedKFold(PurgedFoldConfig(n_splits=3)),
        outer_cv=PurgedKFold(PurgedFoldConfig(n_splits=3)),
        verbose=True
    )

    nested_cv.fit(X, y)

    # 4. Walk-Forward 검증 테스트
    print("\n4. Walk-Forward 검증 테스트")
    wf_validator = WalkForwardValidator(train_window=200, test_window=50, step_size=25)
    wf_splits = list(wf_validator.split(X, y))
    print(f"Walk-Forward 분할 수: {len(wf_splits)}")

    # 5. 보고서 생성
    print("\n5. 검증 보고서 생성")
    report_gen = ValidationReportGenerator()
    report = report_gen.generate_purged_cv_report(nested_cv.cv_results_)

    print(f"평균 성능: {report['summary']['mean_score']:.4f}")
    print(f"표준편차: {report['summary']['std_score']:.4f}")
    print(f"통계적 유의성: {report['statistical_significance']['significant']}")

    print("\n✅ PurgedKFold 및 중첩 교차 검증 시스템 테스트 완료")
    print("금융 시계열 데이터 유출 방지 및 객관적 성능 평가 가능")