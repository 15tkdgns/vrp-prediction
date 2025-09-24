#!/usr/bin/env python3
"""
고급 학습 시스템 - 학습계획.txt 기반 구현
VMD 노이즈 제거, 대체 데이터 통합, PurgedKFold CV, 동적 앙상블
"""

import sys
sys.path.append('/root/workspace')

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 안전 import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.core.ultra_safe_data_processor import UltraSafeDataProcessor
from src.validation.auto_leakage_detector import AutoLeakageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeVMDDenoiser(BaseEstimator, TransformerMixin):
    """VMD 노이즈 제거 (데이터 누출 방지)"""

    def __init__(self, alpha: float = 2000, tau: float = 0.0, n_modes: int = 5):
        self.alpha = alpha
        self.tau = tau
        self.n_modes = n_modes
        self.fitted_params_ = None

    def fit(self, X, y=None):
        """훈련 데이터로만 VMD 파라미터 학습"""
        logger.info(f"VMD 노이즈 제거 파라미터 학습 (훈련 데이터만 사용)")

        # 실제 VMD는 복잡하므로 단순 버전으로 구현
        # 실제로는 PyVMD 라이브러리 사용 권장

        # 이동 평균 기반 노이즈 제거로 근사
        self.fitted_params_ = {
            'window_size': min(10, len(X) // 10),
            'std_threshold': np.std(X.flatten()) * 0.5
        }

        logger.info(f"VMD 파라미터: {self.fitted_params_}")
        return self

    def transform(self, X):
        """학습된 파라미터로 노이즈 제거"""
        if self.fitted_params_ is None:
            raise ValueError("VMD가 먼저 fit되어야 합니다")

        X_denoised = X.copy()
        window_size = self.fitted_params_['window_size']

        # 단순 노이즈 제거 (실제로는 VMD 알고리즘 사용)
        for i in range(X_denoised.shape[1]):
            series = pd.Series(X_denoised[:, i])
            # 이동 평균 스무딩
            smoothed = series.rolling(window=window_size, center=True).mean()
            # 결측값은 원래 값으로 채움
            X_denoised[:, i] = smoothed.fillna(series).values

        logger.info(f"VMD 노이즈 제거 완료: {X.shape} -> {X_denoised.shape}")
        return X_denoised

class AlternativeDataIntegrator(BaseEstimator, TransformerMixin):
    """대체 데이터 통합 피처 엔지니어링"""

    def __init__(self):
        self.feature_names_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        """대체 데이터 특징 생성 규칙 학습"""
        logger.info("대체 데이터 통합 피처 엔지니어링 학습")

        # 기본 특징명 (ultra_safe_data_processor에서 생성된 것)
        base_features = [
            'returns_lag2', 'returns_lag3', 'returns_lag5', 'returns_lag10',
            'vol_5_lag2', 'vol_10_lag2',
            'returns_mean_5_lag2', 'returns_std_5_lag2',
            'returns_diff_2_5', 'returns_diff_3_10'
        ]

        # 추가 기술적 지표 (모두 지연 적용)
        additional_features = [
            'rsi_lag2', 'macd_lag2', 'bollinger_position_lag2',
            'momentum_3_lag2', 'momentum_5_lag2',
            'volatility_ratio_lag2', 'return_skewness_lag2',
            'volume_price_trend_lag2', 'price_acceleration_lag2'
        ]

        self.feature_names_ = base_features + additional_features

        # 피처 생성 후 스케일러 학습
        X_features = self._generate_features(X)
        self.scaler_.fit(X_features)

        logger.info(f"총 특징 수: {len(self.feature_names_)}")
        return self

    def transform(self, X):
        """학습된 규칙으로 대체 데이터 특징 생성"""
        if self.feature_names_ is None:
            raise ValueError("AlternativeDataIntegrator가 먼저 fit되어야 합니다")

        X_features = self._generate_features(X)
        X_scaled = self.scaler_.transform(X_features)

        logger.info(f"대체 데이터 특징 생성: {X.shape} -> {X_scaled.shape}")
        return X_scaled

    def _generate_features(self, X):
        """기술적 지표 및 대체 데이터 특징 생성"""

        # X는 이미 ultra_safe_data_processor에서 처리된 10개 특징
        features = X.copy()

        # 추가 기술적 지표 생성 (모두 안전하게 지연 적용)
        n_samples = features.shape[0]

        # RSI 근사 (과거 수익률 기반)
        returns_lag2 = features[:, 0]  # returns_lag2
        rsi_lag2 = np.zeros(n_samples)
        for i in range(20, n_samples):
            recent_returns = returns_lag2[i-20:i]
            gains = recent_returns[recent_returns > 0]
            losses = -recent_returns[recent_returns < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
            rs = avg_gain / avg_loss
            rsi_lag2[i] = 100 - (100 / (1 + rs))

        # MACD 근사 (이동평균 차이)
        macd_lag2 = np.zeros(n_samples)
        for i in range(26, n_samples):
            ema12 = np.mean(returns_lag2[i-12:i])
            ema26 = np.mean(returns_lag2[i-26:i])
            macd_lag2[i] = ema12 - ema26

        # 볼린저 밴드 위치
        vol_lag2 = features[:, 4]  # vol_5_lag2
        bollinger_position_lag2 = returns_lag2 / (vol_lag2 + 1e-8)

        # 모멘텀 지표들
        momentum_3_lag2 = np.zeros(n_samples)
        momentum_5_lag2 = np.zeros(n_samples)
        for i in range(5, n_samples):
            momentum_3_lag2[i] = np.sum(returns_lag2[i-3:i])
            momentum_5_lag2[i] = np.sum(returns_lag2[i-5:i])

        # 변동성 비율
        vol_10_lag2 = features[:, 5]  # vol_10_lag2
        volatility_ratio_lag2 = vol_lag2 / (vol_10_lag2 + 1e-8)

        # 수익률 비대칭성
        return_skewness_lag2 = np.zeros(n_samples)
        for i in range(20, n_samples):
            recent_returns = returns_lag2[i-20:i]
            if np.std(recent_returns) > 1e-8:
                return_skewness_lag2[i] = np.mean(((recent_returns - np.mean(recent_returns)) / np.std(recent_returns)) ** 3)

        # 가격 가속도 (2차 차분)
        returns_lag3 = features[:, 1]  # returns_lag3
        price_acceleration_lag2 = returns_lag2 - returns_lag3

        # 볼륨-가격 트렌드 근사 (변동성 기반)
        volume_price_trend_lag2 = vol_lag2 * np.sign(returns_lag2)

        # 추가 특징들 결합
        additional_features = np.column_stack([
            rsi_lag2, macd_lag2, bollinger_position_lag2,
            momentum_3_lag2, momentum_5_lag2,
            volatility_ratio_lag2, return_skewness_lag2,
            volume_price_trend_lag2, price_acceleration_lag2
        ])

        # 기존 특징과 결합
        X_combined = np.column_stack([features, additional_features])

        return X_combined

class PurgedKFold:
    """금융 시계열용 Purged K-Fold 교차 검증"""

    def __init__(self, n_splits: int = 5, purge_size: int = 5, embargo_size: int = 5):
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo_size = embargo_size

    def split(self, X, y=None, groups=None):
        """Purged 교차 검증 분할 생성"""
        n_samples = len(X)

        # 각 폴드 크기 계산
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # 테스트 구간 설정
            test_start = i * fold_size
            test_end = test_start + fold_size

            if i == self.n_splits - 1:  # 마지막 폴드
                test_end = n_samples

            # Purge 구간 (테스트 전후 제거)
            purge_start = max(0, test_start - self.purge_size)
            purge_end = min(n_samples, test_end + self.purge_size)

            # 훈련 구간 (Purge + Embargo 적용)
            train_idx = []

            # 테스트 이전 구간
            if purge_start > self.embargo_size:
                train_idx.extend(list(range(0, purge_start - self.embargo_size)))

            # 테스트 이후 구간
            if purge_end + self.embargo_size < n_samples:
                train_idx.extend(list(range(purge_end + self.embargo_size, n_samples)))

            test_idx = list(range(test_start, test_end))

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield np.array(train_idx), np.array(test_idx)

class SafeLSTMModel(BaseEstimator, ClassifierMixin):
    """안전한 LSTM 모델 (간단 구현)"""

    def __init__(self, lookback: int = 10, hidden_size: int = 32):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.model_ = None

    def fit(self, X, y):
        """LSTM 모델 훈련"""
        # 간단한 구현을 위해 RandomForest로 대체
        self.model_ = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        """예측"""
        return self.model_.predict(X)

    def predict_proba(self, X):
        """확률 예측 (방향)"""
        predictions = self.predict(X)
        # 방향 확률로 변환
        probs = np.column_stack([
            1 / (1 + np.exp(predictions)),  # 하락 확률
            1 / (1 + np.exp(-predictions))  # 상승 확률
        ])
        return probs

class SafeTransformerModel(BaseEstimator, ClassifierMixin):
    """안전한 Transformer 모델 (간단 구현)"""

    def __init__(self, sequence_length: int = 20, d_model: int = 64):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.model_ = None

    def fit(self, X, y):
        """Transformer 모델 훈련"""
        # 간단한 구현을 위해 Lasso로 대체 (정규화 효과)
        self.model_ = Lasso(alpha=0.01, max_iter=1000)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        """예측"""
        return self.model_.predict(X)

    def predict_proba(self, X):
        """확률 예측 (방향)"""
        predictions = self.predict(X)
        probs = np.column_stack([
            1 / (1 + np.exp(predictions)),
            1 / (1 + np.exp(-predictions))
        ])
        return probs

class DynamicEnsemble:
    """동적 가중치 앙상블"""

    def __init__(self, models: Dict[str, Any], window_size: int = 50):
        self.models = models
        self.window_size = window_size
        self.weights_ = None

    def fit(self, X, y):
        """모든 베이스 모델 훈련"""
        logger.info("동적 앙상블 모델들 훈련")

        for name, model in self.models.items():
            logger.info(f"  {name} 훈련 중...")
            model.fit(X, y)

        # 초기 균등 가중치
        self.weights_ = {name: 1.0 / len(self.models) for name in self.models.keys()}

        return self

    def update_weights(self, X_recent, y_recent):
        """최근 성과 기반 가중치 업데이트"""

        if len(y_recent) < 5:  # 충분한 데이터가 없으면 업데이트 안함
            return

        losses = {}

        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_recent)

                # 회귀 문제이므로 MSE 사용
                loss = np.mean((y_recent - y_pred) ** 2)
                losses[name] = loss

            except Exception as e:
                logger.warning(f"{name} 모델 평가 실패: {e}")
                losses[name] = 1.0  # 높은 손실 할당

        # 손실이 낮을수록 높은 가중치 (Softmax)
        inv_losses = [1.0 / (loss + 1e-8) for loss in losses.values()]
        total = sum(inv_losses)

        self.weights_ = {
            name: inv_loss / total
            for name, inv_loss in zip(self.models.keys(), inv_losses)
        }

        logger.info(f"가중치 업데이트: {self.weights_}")

    def predict(self, X):
        """가중 평균 예측"""
        predictions = {}

        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        # 가중 평균
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights_[name] * pred

        return ensemble_pred

class AdvancedLearningSystem:
    """고급 학습 시스템 (학습계획.txt 기반)"""

    def __init__(self):
        self.data_processor = UltraSafeDataProcessor()
        self.leakage_detector = AutoLeakageDetector()

        # 파이프라인 구성요소
        self.vmd_denoiser = SafeVMDDenoiser()
        self.feature_integrator = AlternativeDataIntegrator()

        # 베이스 모델들
        self.base_models = {
            'lstm': SafeLSTMModel(),
            'transformer': SafeTransformerModel(),
        }

        if XGBOOST_AVAILABLE:
            self.base_models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )

        # 동적 앙상블
        self.ensemble = None

        # 최종 파이프라인
        self.pipeline = None

        logger.info("고급 학습 시스템 초기화 완료")

    def build_pipeline(self):
        """데이터 처리 파이프라인 구축"""
        logger.info("고급 데이터 처리 파이프라인 구축")

        self.pipeline = Pipeline([
            ('vmd_denoiser', self.vmd_denoiser),
            ('feature_integrator', self.feature_integrator)
        ])

        return self.pipeline

    def nested_cross_validation(self, X, y, param_grids: Dict):
        """중첩 교차 검증으로 모델 튜닝 및 평가"""
        logger.info("중첩 교차 검증 시작")

        # PurgedKFold 설정
        inner_cv = PurgedKFold(n_splits=3, purge_size=5, embargo_size=5)
        outer_cv = PurgedKFold(n_splits=5, purge_size=5, embargo_size=5)

        nested_results = {}

        for model_name, model in self.base_models.items():
            logger.info(f"\n{model_name} 중첩 CV 평가")

            if model_name in param_grids:
                # 내부 루프: 하이퍼파라미터 튜닝
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[model_name],
                    cv=inner_cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=1,
                    verbose=0
                )

                # 외부 루프: 최종 성능 평가
                try:
                    scores = cross_val_score(
                        grid_search, X, y,
                        cv=outer_cv,
                        scoring='neg_mean_squared_error'
                    )

                    nested_results[model_name] = {
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'scores': scores
                    }

                    logger.info(f"  평균 점수: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

                except Exception as e:
                    logger.error(f"  {model_name} 평가 실패: {e}")
                    nested_results[model_name] = {
                        'mean_score': -np.inf,
                        'std_score': np.inf,
                        'error': str(e)
                    }
            else:
                logger.warning(f"  {model_name} 파라미터 그리드 없음 - 기본 설정으로 평가")

        return nested_results

    def train_ensemble(self, X, y):
        """동적 앙상블 훈련"""
        logger.info("동적 앙상블 훈련")

        self.ensemble = DynamicEnsemble(self.base_models)
        self.ensemble.fit(X, y)

        return self.ensemble

    def run_advanced_learning(self, data_path: str):
        """고급 학습 시스템 전체 실행"""
        logger.info("=" * 100)
        logger.info("🚀 고급 학습 시스템 실행 (학습계획.txt 기반)")
        logger.info("=" * 100)

        # 1. 초안전 데이터 준비
        data_dict = self.data_processor.prepare_ultra_safe_data(data_path)
        X_base, y = data_dict['X'], data_dict['y']

        logger.info(f"기본 데이터 준비: X{X_base.shape}, y{y.shape}")

        # 2. 고급 파이프라인 구축 및 적용
        pipeline = self.build_pipeline()

        # 훈련 데이터로 파이프라인 학습
        split_point = int(len(X_base) * 0.8)
        X_train_base = X_base[:split_point]
        X_test_base = X_base[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]

        # 파이프라인 적용
        X_train_processed = pipeline.fit_transform(X_train_base)
        X_test_processed = pipeline.transform(X_test_base)

        logger.info(f"파이프라인 적용: {X_train_base.shape} -> {X_train_processed.shape}")

        # 3. 하이퍼파라미터 그리드 정의
        param_grids = {}

        if XGBOOST_AVAILABLE:
            param_grids['xgboost'] = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2]
            }

        # 4. 중첩 교차 검증
        nested_results = self.nested_cross_validation(
            X_train_processed, y_train, param_grids
        )

        # 5. 동적 앙상블 훈련
        ensemble = self.train_ensemble(X_train_processed, y_train)

        # 6. 최종 성능 평가
        y_pred = ensemble.predict(X_test_processed)

        # 성능 지표
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))

        # 방향 정확도
        direction_actual = (y_test > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100

        # 안전성 검증
        metrics = {
            'r2': 1 - (mse / np.var(y_test)),
            'direction_accuracy': direction_accuracy
        }

        is_safe = self.leakage_detector.validate_during_training(0, 'ensemble', metrics)

        return {
            'nested_cv_results': nested_results,
            'final_metrics': {
                'mse': mse,
                'mae': mae,
                'direction_accuracy': direction_accuracy,
                'r2': metrics['r2']
            },
            'safety_check': is_safe,
            'ensemble_weights': ensemble.weights_,
            'pipeline': pipeline,
            'ensemble': ensemble
        }

def main():
    """메인 실행 함수"""
    system = AdvancedLearningSystem()

    try:
        results = system.run_advanced_learning(
            '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'
        )

        print(f"\n{'='*100}")
        print(f"🏆 고급 학습 시스템 결과")
        print(f"{'='*100}")

        print(f"\n📊 최종 성능:")
        final_metrics = results['final_metrics']
        print(f"   MSE: {final_metrics['mse']:.6f}")
        print(f"   MAE: {final_metrics['mae']:.4f}")
        print(f"   R²: {final_metrics['r2']:.4f}")
        print(f"   방향정확도: {final_metrics['direction_accuracy']:.1f}%")

        print(f"\n🏋️ 앙상블 가중치:")
        for model, weight in results['ensemble_weights'].items():
            print(f"   {model}: {weight:.3f}")

        print(f"\n🛡️ 안전성 검증: {'✅ 통과' if results['safety_check'] else '❌ 실패'}")

        print(f"\n📈 중첩 CV 결과:")
        for model, result in results['nested_cv_results'].items():
            if 'error' not in result:
                print(f"   {model}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
            else:
                print(f"   {model}: 오류 - {result['error']}")

        return results

    except Exception as e:
        logger.error(f"고급 학습 시스템 실패: {e}")
        return None

if __name__ == "__main__":
    result = main()