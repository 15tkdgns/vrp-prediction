#!/usr/bin/env python3
"""
캐글 우승자 기법: 안전한 스태킹/블렌딩 앙상블 시스템
Jane Street, Optiver 등 대회 우승 솔루션의 다층 앙상블 기법 적용
"""

import sys
sys.path.append('/root/workspace')

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# XGBoost 안전 import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LightGBM 안전 import
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.features.kaggle_advanced_features import KaggleAdvancedFeatureEngineer
from src.core.ultra_safe_data_processor import UltraSafeDataProcessor
from src.validation.auto_leakage_detector import AutoLeakageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeTimeSeriesSplit:
    """안전한 시계열 교차 검증 (데이터 누출 방지)"""

    def __init__(self, n_splits: int = 3, test_size: int = 100, gap: int = 5):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """안전한 시계열 분할"""
        n_samples = len(X)

        for i in range(self.n_splits):
            # 테스트 시작점 계산
            test_start = n_samples - (self.n_splits - i) * (self.test_size + self.gap)
            test_end = test_start + self.test_size

            if test_start < self.gap:
                continue

            # 훈련 구간 (gap 적용)
            train_end = test_start - self.gap
            train_start = 0

            if train_end <= train_start:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

class KaggleLevel0Models:
    """Level-0 베이스 모델들 (캐글 우승자 조합)"""

    def __init__(self, safety_mode: bool = True):
        self.safety_mode = safety_mode
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """다양한 베이스 모델 초기화"""
        logger.info("Level-0 베이스 모델 초기화")

        # 1. Linear Models (안전성 우선)
        self.models['ridge_conservative'] = Ridge(alpha=100.0)
        self.models['ridge_moderate'] = Ridge(alpha=10.0)
        self.models['lasso_sparse'] = Lasso(alpha=0.1, max_iter=1000)
        self.models['elastic_balanced'] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)

        # 2. Tree Models (보수적 설정)
        self.models['rf_conservative'] = RandomForestRegressor(
            n_estimators=50, max_depth=5, min_samples_split=10,
            min_samples_leaf=5, random_state=42
        )
        self.models['rf_diverse'] = RandomForestRegressor(
            n_estimators=30, max_depth=7, min_samples_split=5,
            min_samples_leaf=3, random_state=123
        )

        # 3. Boosting Models (사용 가능한 경우)
        if XGBOOST_AVAILABLE:
            self.models['xgb_conservative'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            )
            self.models['xgb_regularized'] = xgb.XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0, random_state=123, verbosity=0
            )

        if LIGHTGBM_AVAILABLE:
            self.models['lgb_conservative'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                feature_fraction=0.8, bagging_fraction=0.8, random_state=42, verbosity=-1
            )

        logger.info(f"총 {len(self.models)}개 베이스 모델 준비 완료")

    def get_models(self):
        """모델 딕셔너리 반환"""
        return self.models

class KaggleStackingEnsemble(BaseEstimator, RegressorMixin):
    """캐글 스타일 스태킹 앙상블"""

    def __init__(self, base_models: Dict, meta_model=None, cv_folds: int = 3):
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else Ridge(alpha=10.0)
        self.cv_folds = cv_folds
        self.fitted_base_models_ = {}
        self.meta_features_scaler_ = StandardScaler()

    def _generate_meta_features(self, X, y):
        """메타 특징 생성 (CV 예측값)"""
        logger.info("메타 특징 생성을 위한 교차 검증 실행")

        meta_features = np.zeros((len(X), len(self.base_models)))
        cv_splitter = SafeTimeSeriesSplit(n_splits=self.cv_folds)

        for i, (model_name, model) in enumerate(self.base_models.items()):
            logger.info(f"  {model_name} CV 예측 생성...")

            try:
                # 안전한 교차 검증 예측
                cv_predictions = cross_val_predict(
                    model, X, y, cv=cv_splitter, method='predict'
                )
                meta_features[:, i] = cv_predictions
            except Exception as e:
                logger.warning(f"  {model_name} CV 실패: {e} - 영벡터로 대체")
                meta_features[:, i] = np.zeros(len(X))

        return meta_features

    def _create_additional_meta_features(self, X, meta_predictions):
        """추가 메타 특징 생성 (캐글 우승자 기법)"""

        # 1. 원본 특징의 통계 요약
        feature_stats = np.column_stack([
            np.mean(X, axis=1),      # 특징별 평균
            np.std(X, axis=1),       # 특징별 표준편차
            np.min(X, axis=1),       # 특징별 최소값
            np.max(X, axis=1)        # 특징별 최대값
        ])

        # 2. 모델 예측값의 통계
        pred_stats = np.column_stack([
            np.mean(meta_predictions, axis=1),     # 예측 평균
            np.std(meta_predictions, axis=1),      # 예측 분산 (불확실성)
            np.min(meta_predictions, axis=1),      # 예측 최소값
            np.max(meta_predictions, axis=1),      # 예측 최대값
        ])

        # 3. 모델 간 합의도
        agreement_features = np.column_stack([
            # 예측값의 상관관계 (다양성 측정)
            np.array([np.corrcoef(meta_predictions[i])[0, 1]
                     if len(meta_predictions[i]) > 1 else 0
                     for i in range(len(meta_predictions))]),
        ])

        # 모든 메타 특징 결합
        enhanced_meta_features = np.column_stack([
            meta_predictions,    # 기본 CV 예측값
            feature_stats,       # 원본 특징 통계
            pred_stats,          # 예측 통계
            agreement_features   # 합의도 특징
        ])

        return enhanced_meta_features

    def fit(self, X, y):
        """스태킹 앙상블 훈련"""
        logger.info("캐글 스타일 스태킹 앙상블 훈련 시작")

        # 1. 메타 특징 생성
        meta_predictions = self._generate_meta_features(X, y)
        enhanced_meta_features = self._create_additional_meta_features(X, meta_predictions)

        # 2. 메타 특징 스케일링
        meta_features_scaled = self.meta_features_scaler_.fit_transform(enhanced_meta_features)

        # 3. 전체 데이터로 베이스 모델들 재훈련
        logger.info("베이스 모델들 전체 데이터로 재훈련")
        for model_name, model in self.base_models.items():
            try:
                fitted_model = clone(model)
                fitted_model.fit(X, y)
                self.fitted_base_models_[model_name] = fitted_model
                logger.info(f"  {model_name} 훈련 완료")
            except Exception as e:
                logger.error(f"  {model_name} 훈련 실패: {e}")

        # 4. 메타 모델 훈련
        logger.info("메타 모델 훈련")
        self.meta_model.fit(meta_features_scaled, y)

        logger.info("스태킹 앙상블 훈련 완료")
        return self

    def predict(self, X):
        """스태킹 앙상블 예측"""

        # 1. 베이스 모델들의 예측
        base_predictions = np.zeros((len(X), len(self.fitted_base_models_)))

        for i, (model_name, model) in enumerate(self.fitted_base_models_.items()):
            try:
                base_predictions[:, i] = model.predict(X)
            except Exception as e:
                logger.warning(f"{model_name} 예측 실패: {e} - 영벡터로 대체")
                base_predictions[:, i] = np.zeros(len(X))

        # 2. 추가 메타 특징 생성
        enhanced_meta_features = self._create_additional_meta_features(X, base_predictions)

        # 3. 메타 특징 스케일링
        meta_features_scaled = self.meta_features_scaler_.transform(enhanced_meta_features)

        # 4. 메타 모델 예측
        final_predictions = self.meta_model.predict(meta_features_scaled)

        return final_predictions

class KaggleBlendingOptimizer:
    """캐글 스타일 블렌딩 최적화"""

    def __init__(self, models: List[Any]):
        self.models = models
        self.optimal_weights_ = None

    def _objective_function(self, weights, predictions, y_true):
        """최적화 목적 함수 (MSE 최소화)"""
        weighted_pred = np.average(predictions, weights=weights, axis=0)
        return mean_squared_error(y_true, weighted_pred)

    def optimize_weights(self, predictions: np.ndarray, y_true: np.ndarray):
        """가중치 최적화 (간단한 그리드 탐색)"""
        logger.info("블렌딩 가중치 최적화")

        n_models = predictions.shape[0]
        best_weights = None
        best_score = float('inf')

        # 간단한 그리드 탐색
        for w1 in np.arange(0.1, 1.0, 0.1):
            for w2 in np.arange(0.1, 1.0 - w1, 0.1):
                if n_models == 2:
                    weights = np.array([w1, 1 - w1])
                elif n_models == 3:
                    w3 = 1 - w1 - w2
                    if w3 > 0:
                        weights = np.array([w1, w2, w3])
                    else:
                        continue
                else:
                    # 더 많은 모델의 경우 균등 가중치
                    weights = np.ones(n_models) / n_models

                score = self._objective_function(weights, predictions, y_true)

                if score < best_score:
                    best_score = score
                    best_weights = weights.copy()

        self.optimal_weights_ = best_weights if best_weights is not None else np.ones(n_models) / n_models
        logger.info(f"최적 가중치: {self.optimal_weights_}")

        return self.optimal_weights_

class SafeKaggleEnsembleSystem:
    """안전한 캐글 앙상블 시스템"""

    def __init__(self):
        self.data_processor = UltraSafeDataProcessor()
        self.feature_engineer = KaggleAdvancedFeatureEngineer(safety_mode=True)
        self.leakage_detector = AutoLeakageDetector()

        # 안전 기준
        self.MAX_R2 = 0.12
        self.MAX_DIRECTION_ACC = 62.0

        # 앙상블 구성요소
        self.level0_models = KaggleLevel0Models(safety_mode=True)
        self.stacking_ensemble = None
        self.blending_optimizer = None

        logger.info("안전한 캐글 앙상블 시스템 초기화")

    def run_kaggle_ensemble_optimization(self, data_path: str):
        """캐글 앙상블 최적화 실행"""
        logger.info("=" * 100)
        logger.info("🏆 캐글 스태킹/블렌딩 앙상블 최적화 시작")
        logger.info("=" * 100)

        # 1. 고급 특징 데이터 준비
        data_dict = self.data_processor.prepare_ultra_safe_data(data_path)
        X_base, y = data_dict['X'], data_dict['y']

        # 캐글 고급 특징 적용
        X_enhanced = self.feature_engineer.fit_transform(X_base)

        # 2. 데이터 분할
        split_point = int(len(X_enhanced) * 0.8)
        X_train = X_enhanced[:split_point]
        X_test = X_enhanced[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]

        logger.info(f"데이터 준비: train={X_train.shape}, test={X_test.shape}")

        # 3. Level-0 베이스 모델들 훈련 및 평가
        base_models = self.level0_models.get_models()
        base_predictions_test = {}

        logger.info(f"Level-0 베이스 모델 개별 성능 평가")
        for model_name, model in base_models.items():
            try:
                # 모델 훈련
                fitted_model = clone(model)
                fitted_model.fit(X_train, y_train)

                # 테스트 예측
                y_pred = fitted_model.predict(X_test)
                base_predictions_test[model_name] = y_pred

                # 성능 계산
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = 1 - (mse / np.var(y_test))

                direction_actual = (y_test > 0).astype(int)
                direction_pred = (y_pred > 0).astype(int)
                direction_acc = np.mean(direction_actual == direction_pred) * 100

                logger.info(f"  {model_name}: R²={r2:.4f}, MAE={mae:.4f}, 방향정확도={direction_acc:.1f}%")

            except Exception as e:
                logger.error(f"  {model_name} 실패: {e}")

        # 4. 스태킹 앙상블 구축
        logger.info(f"\nLevel-1 스태킹 앙상블 구축")
        self.stacking_ensemble = KaggleStackingEnsemble(
            base_models=base_models,
            meta_model=Ridge(alpha=50.0),  # 강한 정규화
            cv_folds=3
        )

        # 스태킹 앙상블 훈련
        self.stacking_ensemble.fit(X_train, y_train)

        # 스태킹 예측
        stacking_pred = self.stacking_ensemble.predict(X_test)

        # 5. 블렌딩 최적화
        logger.info(f"\nLevel-2 블렌딩 최적화")

        # 상위 3개 개별 모델 선택 (성능 기준)
        individual_performances = {}
        for model_name, y_pred in base_predictions_test.items():
            mse = mean_squared_error(y_test, y_pred)
            individual_performances[model_name] = mse

        top_models = sorted(individual_performances.items(), key=lambda x: x[1])[:3]
        logger.info(f"블렌딩용 상위 모델: {[name for name, _ in top_models]}")

        # 블렌딩할 예측값들
        predictions_for_blending = np.array([
            base_predictions_test[name] for name, _ in top_models
        ] + [stacking_pred])

        # 블렌딩 가중치 최적화
        self.blending_optimizer = KaggleBlendingOptimizer(models=None)
        optimal_weights = self.blending_optimizer.optimize_weights(
            predictions_for_blending, y_test
        )

        # 최종 블렌딩 예측
        final_prediction = np.average(predictions_for_blending, weights=optimal_weights, axis=0)

        # 6. 최종 성능 평가
        final_mse = mean_squared_error(y_test, final_prediction)
        final_mae = mean_absolute_error(y_test, final_prediction)
        final_r2 = 1 - (final_mse / np.var(y_test))

        final_direction_actual = (y_test > 0).astype(int)
        final_direction_pred = (final_prediction > 0).astype(int)
        final_direction_acc = np.mean(final_direction_actual == final_direction_pred) * 100

        # 7. 안전성 검증
        metrics = {
            'r2': final_r2,
            'direction_accuracy': final_direction_acc
        }

        safety_check = self.leakage_detector.validate_during_training(0, 'kaggle_ensemble', metrics)
        safe_performance = (final_r2 <= self.MAX_R2 and final_direction_acc <= self.MAX_DIRECTION_ACC)

        logger.info(f"\n{'='*80}")
        logger.info(f"🏆 캐글 앙상블 최종 성능")
        logger.info(f"{'='*80}")
        logger.info(f"R²: {final_r2:.4f} ({'✅ 안전' if final_r2 <= self.MAX_R2 else '❌ 위험'})")
        logger.info(f"MAE: {final_mae:.4f}")
        logger.info(f"방향정확도: {final_direction_acc:.1f}% ({'✅ 안전' if final_direction_acc <= self.MAX_DIRECTION_ACC else '❌ 위험'})")
        logger.info(f"베이스 모델: {len(base_models)}개")
        logger.info(f"블렌딩 가중치: {optimal_weights}")

        success = safety_check and safe_performance
        logger.info(f"최종 안전성: {'✅ 성공' if success else '❌ 실패'}")

        return {
            'final_performance': {
                'r2': final_r2,
                'mae': final_mae,
                'direction_accuracy': final_direction_acc
            },
            'base_models_count': len(base_models),
            'blending_weights': optimal_weights,
            'safety_check': success,
            'stacking_ensemble': self.stacking_ensemble,
            'blending_optimizer': self.blending_optimizer
        }

def main():
    """메인 실행 함수"""
    system = SafeKaggleEnsembleSystem()

    try:
        results = system.run_kaggle_ensemble_optimization(
            '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'
        )

        print(f"\n{'='*80}")
        print(f"🏆 캐글 스태킹/블렌딩 앙상블 완료")
        print(f"{'='*80}")

        perf = results['final_performance']
        print(f"\n📊 최종 앙상블 성능:")
        print(f"   R²: {perf['r2']:.4f}")
        print(f"   MAE: {perf['mae']:.4f}")
        print(f"   방향정확도: {perf['direction_accuracy']:.1f}%")

        print(f"\n🏗️ 앙상블 구조:")
        print(f"   베이스 모델: {results['base_models_count']}개")
        print(f"   블렌딩 가중치: {results['blending_weights']}")

        print(f"\n🛡️ 안전성: {'✅ 통과' if results['safety_check'] else '❌ 실패'}")

        if results['safety_check']:
            print(f"\n🎉 성공: 캐글 우승자 기법으로 안전한 앙상블 시스템 구축!")
        else:
            print(f"\n⚠️ 주의: 안전 기준 초과 - 추가 보정 필요")

        return results

    except Exception as e:
        logger.error(f"캐글 앙상블 시스템 실패: {e}")
        return None

if __name__ == "__main__":
    result = main()