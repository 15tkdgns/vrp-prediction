#!/usr/bin/env python3
"""
안정적인 캐글 앙상블 시스템 (에러 수정 버전)
Jane Street, Optiver 등 우승 기법을 안전하게 적용
"""

import sys
sys.path.append('/root/workspace')

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# XGBoost 안전 import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.features.kaggle_advanced_features import KaggleAdvancedFeatureEngineer
from src.core.ultra_safe_data_processor import UltraSafeDataProcessor
from src.validation.auto_leakage_detector import AutoLeakageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StableKaggleEnsemble:
    """안정적인 캐글 앙상별 시스템"""

    def __init__(self):
        self.data_processor = UltraSafeDataProcessor()
        self.feature_engineer = KaggleAdvancedFeatureEngineer(safety_mode=True)
        self.leakage_detector = AutoLeakageDetector()

        # 엄격한 안전 기준
        self.MAX_R2 = 0.12
        self.MAX_DIRECTION_ACC = 62.0

        # 베이스 모델들
        self.base_models = self._initialize_base_models()
        self.fitted_models_ = {}
        self.ensemble_weights_ = None

        logger.info("안정적인 캐글 앙상블 시스템 초기화")

    def _initialize_base_models(self):
        """베이스 모델 초기화"""
        models = {
            # Linear models (강한 정규화)
            'ridge_strong': Ridge(alpha=100.0),
            'ridge_moderate': Ridge(alpha=10.0),
            'lasso_conservative': Lasso(alpha=0.1, max_iter=1000),
            'elastic_balanced': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),

            # Tree models (보수적 설정)
            'rf_small': RandomForestRegressor(
                n_estimators=30, max_depth=4, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            ),
            'rf_diverse': RandomForestRegressor(
                n_estimators=20, max_depth=6, min_samples_split=8,
                min_samples_leaf=4, random_state=123
            ),
        }

        # XGBoost 추가 (사용 가능한 경우)
        if XGBOOST_AVAILABLE:
            models['xgb_conservative'] = xgb.XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbosity=0
            )

        logger.info(f"베이스 모델 {len(models)}개 초기화 완료")
        return models

    def _safe_time_series_split(self, X, n_splits=3, test_size=80, gap=5):
        """안전한 시계열 분할"""
        n_samples = len(X)
        folds = []

        for i in range(n_splits):
            # 테스트 구간 계산
            test_start = n_samples - (n_splits - i) * (test_size + gap)
            test_end = test_start + test_size

            if test_start <= gap:
                continue

            # 훈련 구간 (gap 적용)
            train_end = test_start - gap
            train_start = 0

            if train_end <= train_start:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            folds.append((train_idx, test_idx))

        return folds

    def _generate_holdout_predictions(self, X, y):
        """Hold-out 방식으로 메타 특징 생성"""
        logger.info("Hold-out 방식으로 메타 특징 생성")

        n_samples = len(X)
        meta_predictions = np.zeros((n_samples, len(self.base_models)))

        # 시계열 분할
        folds = self._safe_time_series_split(X, n_splits=3)

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"  Fold {fold_idx+1}/{len(folds)} 처리")

            X_fold_train = X[train_idx]
            y_fold_train = y[train_idx]
            X_fold_val = X[val_idx]

            # 각 모델별 예측
            for model_idx, (model_name, model) in enumerate(self.base_models.items()):
                try:
                    # 모델 복사 후 훈련
                    fold_model = clone(model)
                    fold_model.fit(X_fold_train, y_fold_train)

                    # 검증 세트 예측
                    val_pred = fold_model.predict(X_fold_val)
                    meta_predictions[val_idx, model_idx] = val_pred

                except Exception as e:
                    logger.warning(f"    {model_name} fold {fold_idx} 실패: {e}")
                    # 실패 시 평균값으로 채움
                    meta_predictions[val_idx, model_idx] = np.mean(y_fold_train)

        return meta_predictions

    def _optimize_ensemble_weights(self, meta_predictions, y_true):
        """앙상블 가중치 최적화 (그리드 탐색)"""
        logger.info("앙상블 가중치 최적화")

        n_models = meta_predictions.shape[1]
        best_weights = None
        best_score = float('inf')

        # 단순 그리드 탐색
        if n_models == 2:
            # 2개 모델
            for w1 in np.arange(0.1, 1.0, 0.1):
                weights = np.array([w1, 1-w1])
                pred = np.average(meta_predictions, weights=weights, axis=1)
                score = mean_squared_error(y_true, pred)
                if score < best_score:
                    best_score = score
                    best_weights = weights

        elif n_models >= 3:
            # 3개 이상 모델
            for w1 in np.arange(0.2, 0.7, 0.1):
                for w2 in np.arange(0.2, 0.8-w1, 0.1):
                    if n_models == 3:
                        w3 = 1 - w1 - w2
                        if w3 > 0.1:
                            weights = np.array([w1, w2, w3])
                        else:
                            continue
                    else:
                        # 4개 이상은 나머지를 균등분할
                        remaining = 1 - w1 - w2
                        remaining_weight = remaining / (n_models - 2)
                        if remaining_weight > 0.05:
                            weights = np.array([w1, w2] + [remaining_weight] * (n_models - 2))
                        else:
                            continue

                    pred = np.average(meta_predictions, weights=weights, axis=1)
                    score = mean_squared_error(y_true, pred)
                    if score < best_score:
                        best_score = score
                        best_weights = weights

        # 기본값 (균등 가중치)
        if best_weights is None:
            best_weights = np.ones(n_models) / n_models

        logger.info(f"최적 가중치: {dict(zip(self.base_models.keys(), best_weights))}")
        return best_weights

    def fit(self, X, y):
        """앙상블 모델 훈련"""
        logger.info("=" * 80)
        logger.info("🏆 안정적인 캐글 앙상블 훈련 시작")
        logger.info("=" * 80)

        # 1. 메타 특징 생성 (Hold-out 방식)
        meta_predictions = self._generate_holdout_predictions(X, y)

        # 2. 유효한 예측값이 있는지 확인
        valid_mask = ~np.isnan(meta_predictions).any(axis=1)
        if np.sum(valid_mask) < len(X) * 0.5:
            logger.warning("유효한 메타 예측이 부족하여 균등 가중치 사용")
            self.ensemble_weights_ = np.ones(len(self.base_models)) / len(self.base_models)
        else:
            # 3. 가중치 최적화 (유효한 데이터만 사용)
            valid_meta_pred = meta_predictions[valid_mask]
            valid_y = y[valid_mask]
            self.ensemble_weights_ = self._optimize_ensemble_weights(valid_meta_pred, valid_y)

        # 4. 전체 데이터로 최종 모델 훈련
        logger.info("전체 데이터로 최종 모델 훈련")
        for model_name, model in self.base_models.items():
            try:
                fitted_model = clone(model)
                fitted_model.fit(X, y)
                self.fitted_models_[model_name] = fitted_model
                logger.info(f"  {model_name} 훈련 완료")
            except Exception as e:
                logger.error(f"  {model_name} 훈련 실패: {e}")

        logger.info("앙상블 훈련 완료")
        return self

    def predict(self, X):
        """앙상블 예측"""
        if not self.fitted_models_:
            raise ValueError("모델이 먼저 훈련되어야 합니다")

        # 각 모델의 예측값 수집
        predictions = []
        model_names = []

        for model_name, model in self.fitted_models_.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                model_names.append(model_name)
            except Exception as e:
                logger.warning(f"{model_name} 예측 실패: {e}")

        if len(predictions) == 0:
            raise ValueError("사용 가능한 모델이 없습니다")

        # 가중 평균 계산
        predictions = np.array(predictions).T  # (n_samples, n_models)

        # 가중치 조정 (사용 가능한 모델만)
        available_weights = self.ensemble_weights_[:len(predictions[0])]
        available_weights = available_weights / np.sum(available_weights)

        ensemble_pred = np.average(predictions, weights=available_weights, axis=1)

        return ensemble_pred

    def run_stable_ensemble_training(self, data_path: str):
        """안정적인 앙상블 훈련 실행"""
        logger.info("=" * 100)
        logger.info("🏆 안정적인 캐글 앙상블 시스템 실행")
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

        # 3. 개별 모델 성능 평가
        logger.info("\n개별 모델 성능 평가")
        individual_results = {}

        for model_name, model in self.base_models.items():
            try:
                # 모델 훈련
                fitted_model = clone(model)
                fitted_model.fit(X_train, y_train)

                # 예측 및 평가
                y_pred = fitted_model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = 1 - (mse / np.var(y_test))

                direction_actual = (y_test > 0).astype(int)
                direction_pred = (y_pred > 0).astype(int)
                direction_acc = np.mean(direction_actual == direction_pred) * 100

                individual_results[model_name] = {
                    'r2': r2, 'mae': mae, 'direction_acc': direction_acc
                }

                logger.info(f"  {model_name}: R²={r2:.4f}, MAE={mae:.4f}, 방향정확도={direction_acc:.1f}%")

            except Exception as e:
                logger.error(f"  {model_name} 평가 실패: {e}")

        # 4. 앙상블 훈련
        logger.info("\n앙상블 훈련")
        self.fit(X_train, y_train)

        # 5. 앙상블 예측 및 평가
        ensemble_pred = self.predict(X_test)

        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = 1 - (ensemble_mse / np.var(y_test))

        ensemble_direction_actual = (y_test > 0).astype(int)
        ensemble_direction_pred = (ensemble_pred > 0).astype(int)
        ensemble_direction_acc = np.mean(ensemble_direction_actual == ensemble_direction_pred) * 100

        # 6. 안전성 검증
        metrics = {
            'r2': ensemble_r2,
            'direction_accuracy': ensemble_direction_acc
        }

        safety_check = self.leakage_detector.validate_during_training(0, 'stable_ensemble', metrics)
        safe_performance = (ensemble_r2 <= self.MAX_R2 and ensemble_direction_acc <= self.MAX_DIRECTION_ACC)

        # 7. 결과 출력
        logger.info(f"\n{'='*80}")
        logger.info(f"🏆 최종 앙상블 성능")
        logger.info(f"{'='*80}")
        logger.info(f"R²: {ensemble_r2:.4f} ({'✅ 안전' if ensemble_r2 <= self.MAX_R2 else '❌ 위험'})")
        logger.info(f"MAE: {ensemble_mae:.4f}")
        logger.info(f"방향정확도: {ensemble_direction_acc:.1f}% ({'✅ 안전' if ensemble_direction_acc <= self.MAX_DIRECTION_ACC else '❌ 위험'})")

        logger.info(f"\n앙상블 구성:")
        for i, (model_name, weight) in enumerate(zip(self.base_models.keys(), self.ensemble_weights_)):
            logger.info(f"  {model_name}: {weight:.3f}")

        success = safety_check and safe_performance
        logger.info(f"\n최종 안전성: {'✅ 성공' if success else '❌ 실패'}")

        return {
            'ensemble_performance': {
                'r2': ensemble_r2,
                'mae': ensemble_mae,
                'direction_accuracy': ensemble_direction_acc
            },
            'individual_results': individual_results,
            'ensemble_weights': dict(zip(self.base_models.keys(), self.ensemble_weights_)),
            'safety_check': success,
            'model_count': len(self.fitted_models_),
            'feature_count': X_enhanced.shape[1]
        }

def main():
    """메인 실행 함수"""
    system = StableKaggleEnsemble()

    try:
        results = system.run_stable_ensemble_training(
            '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'
        )

        print(f"\n{'='*100}")
        print(f"🏆 안정적인 캐글 앙상블 시스템 완료")
        print(f"{'='*100}")

        # 앙상블 성능
        ensemble_perf = results['ensemble_performance']
        print(f"\n📊 앙상블 성능:")
        print(f"   R²: {ensemble_perf['r2']:.4f}")
        print(f"   MAE: {ensemble_perf['mae']:.4f}")
        print(f"   방향정확도: {ensemble_perf['direction_accuracy']:.1f}%")

        # 구성 정보
        print(f"\n🏗️ 앙상블 구성:")
        print(f"   사용된 모델: {results['model_count']}개")
        print(f"   특징 수: {results['feature_count']}개")

        # 가중치 정보
        print(f"\n⚖️ 모델 가중치:")
        for model_name, weight in results['ensemble_weights'].items():
            print(f"   {model_name}: {weight:.3f}")

        # 개별 모델 성능 비교
        print(f"\n📈 개별 모델 성능:")
        for model_name, perf in results['individual_results'].items():
            print(f"   {model_name}: R²={perf['r2']:.4f}, 방향정확도={perf['direction_acc']:.1f}%")

        print(f"\n🛡️ 안전성: {'✅ 통과' if results['safety_check'] else '❌ 실패'}")

        if results['safety_check']:
            print(f"\n🎉 성공: 안정적인 캐글 앙상블 시스템으로 성능 향상 달성!")
        else:
            print(f"\n⚠️ 주의: 안전 기준 초과 - 추가 보정 필요")

        return results

    except Exception as e:
        logger.error(f"안정적인 앙상블 시스템 실패: {e}")
        return None

if __name__ == "__main__":
    result = main()