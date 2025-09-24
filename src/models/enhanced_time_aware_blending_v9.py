#!/usr/bin/env python3
"""
Enhanced Time Aware Blending V9.0
목표: MDD 최소화 (<0.6) + Log Loss 감소 (<0.7)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class EnhancedTimeAwareBlendingV9:
    def __init__(self, max_drawdown_threshold=0.15, confidence_threshold=0.6):
        """
        Enhanced Time Aware Blending with MDD control and log loss optimization

        Args:
            max_drawdown_threshold: 최대 허용 낙폭 (기본: 15%)
            confidence_threshold: 예측 신뢰도 임계값 (기본: 60%)
        """
        self.max_dd_threshold = max_drawdown_threshold
        self.confidence_threshold = confidence_threshold

        # 기본 모델들 (보수적 설정)
        self.base_models = {
            'ridge_ultra_conservative': Ridge(alpha=50.0, random_state=42),
            'lasso_ultra_conservative': Lasso(alpha=0.05, random_state=42),
            'rf_ultra_conservative': RandomForestRegressor(
                n_estimators=20, max_depth=3, random_state=42,
                min_samples_split=20, min_samples_leaf=10
            ),
            'gb_ultra_conservative': GradientBoostingRegressor(
                n_estimators=30, max_depth=2, learning_rate=0.05,
                random_state=42, subsample=0.8
            )
        }

        # 모델 가중치 (초기값)
        self.model_weights = np.array([0.3, 0.25, 0.25, 0.2])
        self.weight_decay = 0.98  # 가중치 감쇠율
        self.min_weight = 0.05   # 최소 가중치

        # 리스크 관리 파라미터
        self.volatility_window = 20
        self.position_sizing_factor = 0.5  # 보수적 포지션 크기
        self.rebalance_threshold = 0.1     # 리밸런싱 임계값

        # 예측 보정 관련
        self.prediction_smoothing = 0.3    # 예측 스무딩 팩터
        self.calibration_window = 50       # 보정 윈도우

        # 성능 추적
        self.performance_history = []
        self.drawdown_history = []
        self.prediction_history = []

    def calculate_volatility_adjusted_weights(self, returns_history):
        """변동성 기반 가중치 조정"""
        if len(returns_history) < self.volatility_window:
            return self.model_weights

        recent_volatility = np.std(returns_history[-self.volatility_window:])

        # 변동성이 높을수록 보수적으로 조정
        volatility_adjustment = 1.0 / (1.0 + recent_volatility * 10)

        # 가중치 재조정
        adjusted_weights = self.model_weights * volatility_adjustment

        # 정규화
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

        return adjusted_weights

    def calculate_current_drawdown(self, returns_history):
        """현재 드로다운 계산"""
        if len(returns_history) == 0:
            return 0.0

        cumulative_returns = np.cumprod(1 + np.array(returns_history))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    def apply_drawdown_control(self, prediction, current_drawdown):
        """드로다운 기반 예측 조정"""
        if current_drawdown > self.max_dd_threshold:
            # 드로다운이 임계값을 초과하면 매우 보수적으로 조정
            conservative_factor = 0.1  # 90% 감소
            return prediction * conservative_factor
        elif current_drawdown > self.max_dd_threshold * 0.7:
            # 드로다운이 임계값의 70%를 초과하면 보수적 조정
            conservative_factor = 0.5  # 50% 감소
            return prediction * conservative_factor
        else:
            return prediction

    def calibrate_prediction_probabilities(self, predictions, targets):
        """예측 확률 보정으로 Log Loss 개선"""
        try:
            # 예측값을 확률로 변환
            pred_probs = 1 / (1 + np.exp(-predictions))  # 시그모이드 변환

            # 극값 방지 (Log Loss 안정화)
            pred_probs = np.clip(pred_probs, 0.001, 0.999)

            # 타겟을 이진 분류로 변환
            binary_targets = (targets > 0).astype(int)

            # 보정 적용 (단순 스케일링)
            if len(np.unique(binary_targets)) > 1:
                # 실제 양성 비율 계산
                actual_positive_rate = np.mean(binary_targets)
                predicted_positive_rate = np.mean(pred_probs > 0.5)

                if predicted_positive_rate > 0:
                    calibration_factor = actual_positive_rate / predicted_positive_rate
                    pred_probs = pred_probs * calibration_factor
                    pred_probs = np.clip(pred_probs, 0.001, 0.999)

            return pred_probs
        except Exception as e:
            print(f"Calibration error: {e}")
            # 기본값 반환
            return np.full(len(predictions), 0.5)

    def dynamic_weight_update(self, model_performances, current_period):
        """모델 성능 기반 동적 가중치 업데이트"""
        if len(model_performances) == 0:
            return self.model_weights

        # 최근 성능에 더 높은 가중치
        recent_window = min(10, len(model_performances))
        recent_performances = model_performances[-recent_window:]

        # 각 모델의 평균 성능 계산 (MAE 기준, 낮을수록 좋음)
        model_maes = []
        for i in range(len(self.base_models)):
            model_mae = np.mean([perf[i] for perf in recent_performances])
            model_maes.append(model_mae)

        # 성능이 좋을수록 높은 가중치 (MAE의 역수 사용)
        inverse_maes = [1.0 / (mae + 1e-6) for mae in model_maes]
        new_weights = np.array(inverse_maes)

        # 정규화
        new_weights = new_weights / np.sum(new_weights)

        # 급격한 변화 방지 (스무딩)
        smoothing_factor = 0.1
        self.model_weights = (1 - smoothing_factor) * self.model_weights + smoothing_factor * new_weights

        # 최소 가중치 보장
        self.model_weights = np.maximum(self.model_weights, self.min_weight)
        self.model_weights = self.model_weights / np.sum(self.model_weights)

        return self.model_weights

    def enhanced_ensemble_prediction(self, X, y_history=None, return_history=None):
        """향상된 앙상블 예측"""
        predictions = []
        model_maes = []

        # 각 모델로 예측
        for model_name, model in self.base_models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)

                # 모델 성능 추적 (가능한 경우)
                if y_history is not None and len(y_history) > 0:
                    mae = mean_absolute_error(y_history[-len(pred):], pred)
                    model_maes.append(mae)
                else:
                    model_maes.append(0.01)  # 기본값

            except Exception as e:
                print(f"Model {model_name} prediction error: {e}")
                predictions.append(np.zeros(len(X)))
                model_maes.append(1.0)  # 높은 오류값

        # 성능 기록 업데이트
        if len(model_maes) == len(self.base_models):
            self.performance_history.append(model_maes)

        # 동적 가중치 업데이트
        if len(self.performance_history) > 5:  # 충분한 이력이 있을 때만
            self.dynamic_weight_update(self.performance_history, len(self.performance_history))

        # 변동성 기반 가중치 조정
        if return_history is not None and len(return_history) > 0:
            vol_adjusted_weights = self.calculate_volatility_adjusted_weights(return_history)
        else:
            vol_adjusted_weights = self.model_weights

        # 가중 평균 예측
        predictions_array = np.array(predictions)
        ensemble_pred = np.average(predictions_array, axis=0, weights=vol_adjusted_weights)

        # 예측 스무딩 적용
        if len(self.prediction_history) > 0:
            previous_pred = self.prediction_history[-1]
            ensemble_pred = (1 - self.prediction_smoothing) * ensemble_pred + \
                          self.prediction_smoothing * previous_pred

        # 예측 이력 업데이트
        self.prediction_history.append(ensemble_pred)

        # 드로다운 제어 적용
        if return_history is not None and len(return_history) > 0:
            current_dd = self.calculate_current_drawdown(return_history)
            self.drawdown_history.append(current_dd)
            ensemble_pred = self.apply_drawdown_control(ensemble_pred, current_dd)

        return ensemble_pred

    def fit(self, X, y):
        """모델 훈련"""
        print("🔧 Enhanced Time Aware Blending V9 훈련 시작...")

        # 데이터 스케일링
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 각 모델 훈련
        for model_name, model in self.base_models.items():
            print(f"  📈 {model_name} 훈련 중...")
            model.fit(X_scaled, y)

        print("✅ 모든 모델 훈련 완료")
        return self

    def predict(self, X, y_history=None, return_history=None):
        """예측 실행"""
        X_scaled = self.scaler.transform(X)
        return self.enhanced_ensemble_prediction(X_scaled, y_history, return_history)

    def predict_proba(self, X, y_history=None, return_history=None):
        """확률 예측 (Log Loss 계산용)"""
        predictions = self.predict(X, y_history, return_history)

        # 실제 타겟이 있다면 보정 적용
        if y_history is not None and len(y_history) > 0:
            calibrated_probs = self.calibrate_prediction_probabilities(predictions, y_history)
        else:
            # 기본 시그모이드 변환
            calibrated_probs = 1 / (1 + np.exp(-predictions))
            calibrated_probs = np.clip(calibrated_probs, 0.001, 0.999)

        return calibrated_probs

    def get_performance_summary(self):
        """성능 요약 반환"""
        if len(self.drawdown_history) == 0:
            return {
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'drawdown_control_active': False,
                'model_weights': self.model_weights.tolist()
            }

        return {
            'max_drawdown': max(self.drawdown_history),
            'current_drawdown': self.drawdown_history[-1],
            'avg_drawdown': np.mean(self.drawdown_history),
            'drawdown_control_active': self.drawdown_history[-1] > self.max_dd_threshold,
            'model_weights': self.model_weights.tolist(),
            'weight_evolution': len(self.performance_history),
            'prediction_smoothing_factor': self.prediction_smoothing
        }

def create_enhanced_time_aware_blending_v9():
    """Enhanced Time Aware Blending V9 모델 생성"""
    return EnhancedTimeAwareBlendingV9(
        max_drawdown_threshold=0.12,  # 12% MDD 제한
        confidence_threshold=0.65      # 65% 신뢰도 임계값
    )

if __name__ == "__main__":
    print("🚀 Enhanced Time Aware Blending V9 - MDD 최소화 & Log Loss 최적화")

    # 테스트용 데이터 생성
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.02  # 2% 변동성

    # 모델 테스트
    model = create_enhanced_time_aware_blending_v9()
    model.fit(X[:150], y[:150])

    # 예측 테스트
    predictions = model.predict(X[150:], y_history=y[:150])
    proba_predictions = model.predict_proba(X[150:], y_history=y[:150])

    # 성능 평가
    test_y = y[150:]
    mae = mean_absolute_error(test_y, predictions)

    # Log Loss 계산
    binary_targets = (test_y > 0).astype(int)
    log_loss_val = log_loss(binary_targets, proba_predictions)

    print(f"\n📊 테스트 결과:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Log Loss: {log_loss_val:.4f}")

    # 성능 요약
    summary = model.get_performance_summary()
    print(f"\n🔍 모델 상태:")
    for key, value in summary.items():
        print(f"  {key}: {value}")