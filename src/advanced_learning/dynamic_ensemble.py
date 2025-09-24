#!/usr/bin/env python3
"""
동적 앙상블 및 CombinatorialPurgedCV 시스템
최근 성과 기반 동적 가중치 부여 및 강건성 검증 시스템
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score, f1_score
from scipy.special import softmax
import warnings
import logging
from datetime import datetime, timedelta
import itertools
import json
from collections import deque

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class DynamicEnsembleConfig:
    """동적 앙상블 설정 클래스"""
    window_size: int = 50  # 최근 성과 평가 윈도우
    update_frequency: int = 10  # 가중치 업데이트 빈도
    min_weight: float = 0.01  # 최소 가중치
    max_weight: float = 0.9   # 최대 가중치
    decay_factor: float = 0.95  # 시간 감쇠 인수
    performance_metric: str = 'log_loss'  # 성과 지표


@dataclass
class ModelPerformance:
    """모델 성과 기록 클래스"""
    model_name: str
    timestamps: List[datetime] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    actuals: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)


class DynamicWeightCalculator:
    """동적 가중치 계산기"""

    def __init__(self, config: DynamicEnsembleConfig):
        """
        Args:
            config: 동적 앙상블 설정
        """
        self.config = config

    def calculate_weights(self, model_performances: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """
        최근 성과를 기반으로 동적 가중치 계산

        Args:
            model_performances: 모델별 성과 기록

        Returns:
            model_weights: 모델별 가중치
        """
        model_names = list(model_performances.keys())
        if not model_names:
            return {}

        recent_losses = {}

        for model_name, performance in model_performances.items():
            if len(performance.losses) > 0:
                # 최근 window_size만큼의 손실 가져오기
                recent = performance.losses[-self.config.window_size:]

                # 시간 가중 평균 (최근일수록 높은 가중치)
                weights = [self.config.decay_factor ** i for i in range(len(recent))]
                weights.reverse()  # 최근이 더 높은 가중치

                weighted_loss = np.average(recent, weights=weights)
                recent_losses[model_name] = weighted_loss
            else:
                # 초기값: 중간 성능
                recent_losses[model_name] = 0.693  # log(2)

        # 손실이 낮을수록 높은 가중치 부여
        # 1/loss를 사용하되, 수치 안정성을 위해 소량 추가
        inverse_losses = {name: 1.0 / (loss + 1e-6) for name, loss in recent_losses.items()}

        # Softmax 정규화
        loss_values = list(inverse_losses.values())
        softmax_weights = softmax(loss_values)

        # 최소/최대 가중치 제한 적용
        weights = {}
        for i, model_name in enumerate(model_names):
            weight = softmax_weights[i]
            weight = max(self.config.min_weight, min(self.config.max_weight, weight))
            weights[model_name] = weight

        # 가중치 재정규화
        total_weight = sum(weights.values())
        weights = {name: w / total_weight for name, w in weights.items()}

        return weights

    def update_performance(self, model_performances: Dict[str, ModelPerformance],
                         model_predictions: Dict[str, float],
                         actual: int, timestamp: datetime = None) -> None:
        """
        모델 성과 업데이트

        Args:
            model_performances: 모델별 성과 기록
            model_predictions: 모델별 예측값
            actual: 실제값
            timestamp: 타임스탬프
        """
        if timestamp is None:
            timestamp = datetime.now()

        for model_name, prediction in model_predictions.items():
            if model_name not in model_performances:
                model_performances[model_name] = ModelPerformance(model_name)

            performance = model_performances[model_name]

            # 손실 계산
            try:
                loss = log_loss([actual], [prediction])
            except:
                loss = float('inf')

            # 기록 업데이트
            performance.timestamps.append(timestamp)
            performance.predictions.append(prediction)
            performance.actuals.append(actual)
            performance.losses.append(loss)

            # 최대 기록 수 제한 (메모리 관리)
            max_records = self.config.window_size * 3
            if len(performance.losses) > max_records:
                performance.timestamps = performance.timestamps[-max_records:]
                performance.predictions = performance.predictions[-max_records:]
                performance.actuals = performance.actuals[-max_records:]
                performance.losses = performance.losses[-max_records:]


class DynamicEnsemble(BaseEstimator, ClassifierMixin):
    """
    동적 앙상블 시스템
    최근 성과에 기반하여 모델 가중치를 실시간으로 조정
    """

    def __init__(self, base_models: Dict[str, BaseEstimator] = None,
                 config: DynamicEnsembleConfig = None):
        """
        Args:
            base_models: 기본 모델들
            config: 동적 앙상블 설정
        """
        self.base_models = base_models or {}
        self.config = config or DynamicEnsembleConfig()
        self.weight_calculator = DynamicWeightCalculator(self.config)
        self.model_performances = {}
        self.current_weights = {}
        self.is_fitted = False
        self.prediction_count = 0

    def add_model(self, name: str, model: BaseEstimator) -> None:
        """앙상블에 모델 추가"""
        self.base_models[name] = model
        self.model_performances[name] = ModelPerformance(name)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DynamicEnsemble':
        """
        앙상블 훈련

        Args:
            X: 특성 데이터
            y: 타겟 데이터

        Returns:
            자기 자신 (fitted)
        """
        if not self.base_models:
            raise ValueError("앙상블에 모델이 추가되지 않았습니다.")

        # 각 기본 모델 훈련
        for name, model in self.base_models.items():
            logger.info(f"모델 {name} 훈련 중...")
            model.fit(X, y)

        # 초기 가중치 설정 (동일 가중치)
        n_models = len(self.base_models)
        initial_weight = 1.0 / n_models
        self.current_weights = {name: initial_weight for name in self.base_models.keys()}

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        확률 예측

        Args:
            X: 특성 데이터

        Returns:
            predictions: 확률 예측값
        """
        if not self.is_fitted:
            raise ValueError("앙상블이 아직 훈련되지 않았습니다.")

        # 각 모델의 예측 수집
        model_predictions = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
                if len(pred_proba.shape) > 1 and pred_proba.shape[1] > 1:
                    pred = pred_proba[:, 1]  # 양성 클래스 확률
                else:
                    pred = pred_proba.flatten()
            else:
                pred = model.predict(X)

            model_predictions[name] = pred

        # 가중 평균 계산
        ensemble_predictions = np.zeros(len(X))
        for name, predictions in model_predictions.items():
            weight = self.current_weights.get(name, 0)
            ensemble_predictions += weight * predictions

        # scikit-learn 호환을 위해 2열로 변환
        proba = np.column_stack([1 - ensemble_predictions, ensemble_predictions])
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이진 예측"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def update_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        실시간 가중치 업데이트

        Args:
            X: 새로운 특성 데이터
            y: 새로운 타겟 데이터
        """
        if not self.is_fitted:
            return

        # 각 모델의 예측 수집
        model_predictions = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
                if len(pred_proba.shape) > 1 and pred_proba.shape[1] > 1:
                    pred = pred_proba[:, 1]
                else:
                    pred = pred_proba.flatten()
            else:
                pred = model.predict(X)
            model_predictions[name] = pred

        # 성과 업데이트
        for i in range(len(X)):
            single_predictions = {name: pred[i] for name, pred in model_predictions.items()}
            self.weight_calculator.update_performance(
                self.model_performances, single_predictions, y[i]
            )

        # 가중치 재계산 (일정 주기마다)
        self.prediction_count += len(X)
        if self.prediction_count >= self.config.update_frequency:
            self.current_weights = self.weight_calculator.calculate_weights(self.model_performances)
            self.prediction_count = 0
            logger.info(f"가중치 업데이트: {self.current_weights}")

    def get_model_weights(self) -> Dict[str, float]:
        """현재 모델 가중치 반환"""
        return self.current_weights.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """성과 요약 반환"""
        summary = {}
        for name, performance in self.model_performances.items():
            if len(performance.losses) > 0:
                summary[name] = {
                    'recent_loss': np.mean(performance.losses[-self.config.window_size:]),
                    'total_predictions': len(performance.predictions),
                    'current_weight': self.current_weights.get(name, 0),
                    'loss_trend': np.mean(performance.losses[-10:]) if len(performance.losses) >= 10 else None
                }
        return summary


class CombinatorialBacktester:
    """
    조합적 백테스팅 시스템
    CombinatorialPurgedCV를 사용한 강건성 검증
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

    def run_combinatorial_backtest(self, ensemble: DynamicEnsemble,
                                 X: np.ndarray, y: np.ndarray,
                                 returns: np.ndarray = None) -> Dict[str, Any]:
        """
        조합적 백테스트 실행

        Args:
            ensemble: 동적 앙상블 모델
            X: 특성 데이터
            y: 타겟 데이터
            returns: 수익률 데이터 (선택사항)

        Returns:
            backtest_results: 백테스트 결과
        """
        results = {
            'fold_results': [],
            'performance_distributions': {},
            'ensemble_weights_evolution': [],
            'model_stability': {}
        }

        n_samples = len(X)
        group_size = n_samples // self.n_groups

        # 모든 그룹 조합 생성
        groups = list(range(self.n_groups))
        test_group_combinations = list(itertools.combinations(groups, self.n_test_groups))

        fold_count = 0
        for test_groups in test_group_combinations:
            fold_count += 1

            # 훈련/테스트 인덱스 생성
            train_indices, test_indices = self._get_train_test_indices(
                test_groups, group_size, n_samples
            )

            if len(train_indices) < 50 or len(test_indices) < 10:
                continue

            # 데이터 분할
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # 앙상블 복사 및 훈련
            ensemble_copy = self._clone_ensemble(ensemble)
            ensemble_copy.fit(X_train, y_train)

            # 예측 및 평가
            y_pred_proba = ensemble_copy.predict_proba(X_test)[:, 1]
            y_pred = ensemble_copy.predict(X_test)

            # 성과 지표 계산
            fold_metrics = self._calculate_fold_metrics(
                y_test, y_pred, y_pred_proba, returns[test_indices] if returns is not None else None
            )

            # 폴드 결과 저장
            fold_result = {
                'fold': fold_count,
                'test_groups': test_groups,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                'metrics': fold_metrics,
                'model_weights': ensemble_copy.get_model_weights(),
                'predictions': y_pred_proba.tolist()
            }
            results['fold_results'].append(fold_result)

            # 앙상블 가중치 진화 기록
            results['ensemble_weights_evolution'].append({
                'fold': fold_count,
                'weights': ensemble_copy.get_model_weights()
            })

            if fold_count % 10 == 0:
                logger.info(f"백테스트 진행: {fold_count}/{len(test_group_combinations)} 폴드 완료")

        # 성과 분포 계산
        results['performance_distributions'] = self._calculate_performance_distributions(
            results['fold_results']
        )

        # 모델 안정성 분석
        results['model_stability'] = self._analyze_model_stability(
            results['ensemble_weights_evolution']
        )

        return results

    def _get_train_test_indices(self, test_groups: Tuple[int],
                              group_size: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """훈련/테스트 인덱스 생성"""
        test_indices = []
        train_indices = []

        # 테스트 그룹 인덱스
        for group_idx in test_groups:
            start_idx = group_idx * group_size
            end_idx = min((group_idx + 1) * group_size, n_samples)
            if group_idx == self.n_groups - 1:
                end_idx = n_samples
            test_indices.extend(range(start_idx, end_idx))

        # 훈련 그룹 인덱스 (정제 적용)
        for group_idx in range(self.n_groups):
            if group_idx not in test_groups:
                # 테스트 그룹과 인접한지 확인
                is_adjacent = any(abs(group_idx - test_group) <= 1 for test_group in test_groups)

                if not is_adjacent:
                    start_idx = group_idx * group_size
                    end_idx = min((group_idx + 1) * group_size, n_samples)
                    if group_idx == self.n_groups - 1:
                        end_idx = n_samples
                    train_indices.extend(range(start_idx, end_idx))

        return np.array(train_indices), np.array(test_indices)

    def _clone_ensemble(self, ensemble: DynamicEnsemble) -> DynamicEnsemble:
        """앙상블 복사"""
        from sklearn.base import clone

        cloned_models = {}
        for name, model in ensemble.base_models.items():
            try:
                cloned_models[name] = clone(model)
            except:
                # clone이 실패하면 원본 사용
                cloned_models[name] = model

        return DynamicEnsemble(cloned_models, ensemble.config)

    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: np.ndarray, returns: np.ndarray = None) -> Dict[str, float]:
        """폴드별 성과 지표 계산"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred_proba),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

        # 금융 지표 (수익률이 제공된 경우)
        if returns is not None:
            # 간단한 수익률 기반 지표
            portfolio_returns = returns * (2 * y_pred - 1)  # 예측에 따른 롱/숏
            metrics['portfolio_return'] = np.sum(portfolio_returns)
            metrics['sharpe_ratio'] = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-6) * np.sqrt(252)

        return metrics

    def _calculate_performance_distributions(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """성과 분포 계산"""
        metrics_names = fold_results[0]['metrics'].keys() if fold_results else []
        distributions = {}

        for metric_name in metrics_names:
            values = [fold['metrics'][metric_name] for fold in fold_results]
            distributions[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q05': np.percentile(values, 5),
                'q25': np.percentile(values, 25),
                'q50': np.percentile(values, 50),
                'q75': np.percentile(values, 75),
                'q95': np.percentile(values, 95),
                'values': values
            }

        return distributions

    def _analyze_model_stability(self, weights_evolution: List[Dict]) -> Dict[str, Any]:
        """모델 안정성 분석"""
        if not weights_evolution:
            return {}

        # 모델명 추출
        model_names = list(weights_evolution[0]['weights'].keys())
        stability_analysis = {}

        for model_name in model_names:
            weights = [fold['weights'][model_name] for fold in weights_evolution]
            stability_analysis[model_name] = {
                'mean_weight': np.mean(weights),
                'std_weight': np.std(weights),
                'min_weight': np.min(weights),
                'max_weight': np.max(weights),
                'coefficient_of_variation': np.std(weights) / (np.mean(weights) + 1e-6),
                'weight_evolution': weights
            }

        return stability_analysis


# 사용 예시 및 테스트
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    returns = np.random.randn(n_samples) * 0.02  # 일일 수익률

    # 훈련/테스트 분할
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    returns_test = returns[split_idx:]

    print("=== 동적 앙상블 및 CombinatorialPurgedCV 테스트 ===")

    # 1. 동적 앙상블 생성
    print("\n1. 동적 앙상블 생성 및 훈련")
    config = DynamicEnsembleConfig(
        window_size=30,
        update_frequency=10,
        decay_factor=0.95
    )

    ensemble = DynamicEnsemble(config=config)
    ensemble.add_model('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ensemble.add_model('lr', LogisticRegression(random_state=42, max_iter=200))
    ensemble.add_model('svm', SVC(probability=True, random_state=42))

    ensemble.fit(X_train, y_train)

    # 초기 예측
    initial_pred = ensemble.predict_proba(X_test[:10])
    print(f"초기 가중치: {ensemble.get_model_weights()}")

    # 2. 동적 가중치 업데이트
    print("\n2. 동적 가중치 업데이트 테스트")
    for i in range(0, min(100, len(X_test)), 10):
        end_idx = min(i + 10, len(X_test))
        X_batch = X_test[i:end_idx]
        y_batch = y_test[i:end_idx]

        ensemble.update_weights(X_batch, y_batch)

        if i % 20 == 0:
            current_weights = ensemble.get_model_weights()
            print(f"Batch {i//10 + 1} 가중치: {current_weights}")

    # 3. 성과 요약
    print("\n3. 성과 요약")
    performance_summary = ensemble.get_performance_summary()
    for model_name, stats in performance_summary.items():
        print(f"{model_name}: 최근 손실={stats['recent_loss']:.4f}, "
              f"현재 가중치={stats['current_weight']:.3f}")

    # 4. 조합적 백테스트
    print("\n4. 조합적 백테스트 실행")
    backtester = CombinatorialBacktester(n_groups=8, n_test_groups=2)

    # 작은 데이터셋으로 테스트
    test_size = 400
    X_backtest = X[:test_size]
    y_backtest = y[:test_size]
    returns_backtest = returns[:test_size]

    backtest_results = backtester.run_combinatorial_backtest(
        ensemble, X_backtest, y_backtest, returns_backtest
    )

    # 결과 분석
    print(f"\n=== 백테스트 결과 ===")
    print(f"총 폴드 수: {len(backtest_results['fold_results'])}")

    perf_dist = backtest_results['performance_distributions']
    if 'accuracy' in perf_dist:
        acc_stats = perf_dist['accuracy']
        print(f"정확도 분포: 평균={acc_stats['mean']:.4f}, "
              f"표준편차={acc_stats['std']:.4f}, "
              f"95% 신뢰구간=[{acc_stats['q05']:.4f}, {acc_stats['q95']:.4f}]")

    if 'sharpe_ratio' in perf_dist:
        sharpe_stats = perf_dist['sharpe_ratio']
        print(f"샤프 비율 분포: 평균={sharpe_stats['mean']:.4f}, "
              f"95% 신뢰구간=[{sharpe_stats['q05']:.4f}, {sharpe_stats['q95']:.4f}]")

    # 모델 안정성
    stability = backtest_results['model_stability']
    print(f"\n=== 모델 안정성 ===")
    for model_name, stats in stability.items():
        cv = stats['coefficient_of_variation']
        print(f"{model_name}: 가중치 변동계수={cv:.4f} "
              f"({'안정' if cv < 0.5 else '불안정'})")

    print("\n✅ 동적 앙상블 및 CombinatorialPurgedCV 시스템 테스트 완료")
    print("실시간 가중치 조정 및 강건성 검증 기능 구현 성공")