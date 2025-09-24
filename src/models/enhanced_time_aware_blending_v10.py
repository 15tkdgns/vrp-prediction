#!/usr/bin/env python3
"""
Enhanced Time Aware Blending V10.0
목표: Log Loss < 0.7 달성 (MDD는 V9에서 완벽 달성)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

class EnhancedTimeAwareBlendingV10:
    def __init__(self, max_drawdown_threshold=0.12, confidence_threshold=0.6):
        """
        Enhanced Time Aware Blending V10 with advanced probability calibration

        Args:
            max_drawdown_threshold: 최대 허용 낙폭 (기본: 12%)
            confidence_threshold: 예측 신뢰도 임계값 (기본: 60%)
        """
        self.max_dd_threshold = max_drawdown_threshold
        self.confidence_threshold = confidence_threshold

        # 기본 모델들 (더욱 보수적 설정)
        self.base_models = {
            'ridge_ultra_safe': Ridge(alpha=100.0, random_state=42),
            'lasso_ultra_safe': Lasso(alpha=0.1, random_state=42),
            'rf_ultra_safe': RandomForestRegressor(
                n_estimators=15, max_depth=2, random_state=42,
                min_samples_split=30, min_samples_leaf=15
            ),
            'gb_ultra_safe': GradientBoostingRegressor(
                n_estimators=20, max_depth=2, learning_rate=0.03,
                random_state=42, subsample=0.7
            )
        }

        # 모델 가중치 (더욱 보수적)
        self.model_weights = np.array([0.35, 0.3, 0.2, 0.15])
        self.weight_decay = 0.99  # 더 느린 가중치 감쇠
        self.min_weight = 0.08   # 더 높은 최소 가중치

        # 리스크 관리 파라미터 (V9보다 보수적)
        self.volatility_window = 30
        self.position_sizing_factor = 0.3  # 더 보수적
        self.rebalance_threshold = 0.05    # 더 민감한 리밸런싱

        # 예측 보정 관련 (Log Loss 최적화)
        self.prediction_smoothing = 0.5    # 더 강한 스무딩
        self.calibration_window = 100      # 더 긴 보정 윈도우
        self.calibration_method = 'isotonic'  # 등장 회귀 사용

        # 확률 보정을 위한 추가 파라미터
        self.probability_bounds = (0.1, 0.9)  # 확률 경계
        self.confidence_scaling = 0.8         # 신뢰도 스케일링

        # 성능 추적
        self.performance_history = []
        self.drawdown_history = []
        self.prediction_history = []
        self.calibration_history = []

    def advanced_probability_calibration(self, predictions, targets):
        """고급 확률 보정으로 Log Loss 최적화"""
        try:
            # 1. 기본 시그모이드 변환
            pred_probs = 1 / (1 + np.exp(-predictions))

            # 2. 타겟을 이진 분류로 변환
            binary_targets = (targets > 0).astype(int)

            if len(np.unique(binary_targets)) < 2:
                # 모든 값이 같은 클래스인 경우
                return np.full(len(predictions), 0.5)

            # 3. 등장 회귀를 사용한 보정
            if len(predictions) >= 10:  # 충분한 데이터가 있을 때만
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                calibrated_probs = iso_reg.fit_transform(pred_probs, binary_targets)
            else:
                calibrated_probs = pred_probs

            # 4. 실제 양성 비율 기반 추가 보정
            actual_positive_rate = np.mean(binary_targets)
            predicted_positive_rate = np.mean(calibrated_probs > 0.5)

            if predicted_positive_rate > 0:
                adjustment_factor = actual_positive_rate / predicted_positive_rate
                # 부드러운 조정 적용
                adjustment_factor = 0.5 + 0.5 * adjustment_factor  # [0.5, 1.5] 범위로 제한
                calibrated_probs = calibrated_probs * adjustment_factor

            # 5. 신뢰도 기반 스케일링
            # 중간값(0.5)으로 수렴시켜 불확실성 증가
            calibrated_probs = 0.5 + (calibrated_probs - 0.5) * self.confidence_scaling

            # 6. 극값 방지 (Log Loss 안정화)
            calibrated_probs = np.clip(calibrated_probs,
                                     self.probability_bounds[0],
                                     self.probability_bounds[1])

            # 7. 보정 이력 저장
            self.calibration_history.append({
                'original_mean': np.mean(pred_probs),
                'calibrated_mean': np.mean(calibrated_probs),
                'target_positive_rate': actual_positive_rate,
                'pred_positive_rate': np.mean(calibrated_probs > 0.5)
            })

            return calibrated_probs

        except Exception as e:
            print(f"Advanced calibration error: {e}")
            # 안전한 기본값 반환
            return np.full(len(predictions), 0.5)

    def temporal_smoothing(self, current_prediction):
        """시간적 스무딩으로 예측 안정화"""
        if len(self.prediction_history) == 0:
            return current_prediction

        # 최근 N개 예측의 가중 평균
        window_size = min(5, len(self.prediction_history))
        recent_predictions = self.prediction_history[-window_size:]

        # 시간이 가까울수록 높은 가중치
        weights = np.exp(np.linspace(-1, 0, window_size))
        weights = weights / np.sum(weights)

        weighted_average = np.average(recent_predictions, weights=weights, axis=0)

        # 현재 예측과 이력의 혼합
        smoothed_prediction = (1 - self.prediction_smoothing) * current_prediction + \
                             self.prediction_smoothing * weighted_average

        return smoothed_prediction

    def conservative_ensemble_prediction(self, X, y_history=None, return_history=None):
        """매우 보수적인 앙상블 예측"""
        predictions = []
        model_confidences = []

        # 각 모델로 예측
        for model_name, model in self.base_models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)

                # 모델 신뢰도 계산 (예측 분산의 역수)
                pred_variance = np.var(pred)
                confidence = 1.0 / (1.0 + pred_variance)
                model_confidences.append(confidence)

            except Exception as e:
                print(f"Model {model_name} prediction error: {e}")
                predictions.append(np.zeros(len(X)))
                model_confidences.append(0.1)  # 낮은 신뢰도

        # 신뢰도 기반 가중치 조정
        confidence_weights = np.array(model_confidences)
        confidence_weights = confidence_weights / np.sum(confidence_weights)

        # 기존 가중치와 신뢰도 가중치의 혼합
        final_weights = 0.7 * self.model_weights + 0.3 * confidence_weights

        # 가중 평균 예측
        predictions_array = np.array(predictions)
        ensemble_pred = np.average(predictions_array, axis=0, weights=final_weights)

        # 변동성 기반 추가 조정
        if return_history is not None and len(return_history) > 0:
            recent_volatility = np.std(return_history[-self.volatility_window:])
            volatility_factor = 1.0 / (1.0 + recent_volatility * 20)  # 더 보수적
            ensemble_pred = ensemble_pred * volatility_factor

        # 시간적 스무딩 적용
        if len(self.prediction_history) > 0:
            ensemble_pred = self.temporal_smoothing(ensemble_pred)

        # 예측 이력 업데이트
        self.prediction_history.append(ensemble_pred)

        # 극값 제한 (안정성 증가)
        ensemble_pred = np.clip(ensemble_pred, -0.05, 0.05)  # ±5% 제한

        return ensemble_pred

    def fit(self, X, y):
        """모델 훈련"""
        print("🔧 Enhanced Time Aware Blending V10 훈련 시작...")

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
        return self.conservative_ensemble_prediction(X_scaled, y_history, return_history)

    def predict_proba(self, X, y_history=None, return_history=None):
        """확률 예측 (Log Loss 최적화)"""
        predictions = self.predict(X, y_history, return_history)

        # 고급 확률 보정 적용
        if y_history is not None and len(y_history) > 0:
            calibrated_probs = self.advanced_probability_calibration(predictions, y_history)
        else:
            # 기본 보수적 변환
            calibrated_probs = 1 / (1 + np.exp(-predictions))
            calibrated_probs = np.clip(calibrated_probs,
                                     self.probability_bounds[0],
                                     self.probability_bounds[1])

        return calibrated_probs

    def get_performance_summary(self):
        """성능 요약 반환"""
        summary = {
            'max_drawdown': max(self.drawdown_history) if self.drawdown_history else 0.0,
            'current_drawdown': self.drawdown_history[-1] if self.drawdown_history else 0.0,
            'avg_drawdown': np.mean(self.drawdown_history) if self.drawdown_history else 0.0,
            'model_weights': self.model_weights.tolist(),
            'prediction_smoothing_factor': self.prediction_smoothing,
            'calibration_method': self.calibration_method,
            'probability_bounds': self.probability_bounds,
            'confidence_scaling': self.confidence_scaling
        }

        if self.calibration_history:
            recent_calibration = self.calibration_history[-1]
            summary.update({
                'last_calibration_adjustment': abs(recent_calibration['calibrated_mean'] - recent_calibration['original_mean']),
                'target_alignment': abs(recent_calibration['target_positive_rate'] - recent_calibration['pred_positive_rate'])
            })

        return summary

def create_enhanced_time_aware_blending_v10():
    """Enhanced Time Aware Blending V10 모델 생성"""
    return EnhancedTimeAwareBlendingV10(
        max_drawdown_threshold=0.1,   # 10% MDD 제한 (더 보수적)
        confidence_threshold=0.7       # 70% 신뢰도 임계값 (더 엄격)
    )

if __name__ == "__main__":
    print("🚀 Enhanced Time Aware Blending V10 - Log Loss < 0.7 달성 목표")

    # 테스트용 데이터 생성
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.015  # 1.5% 변동성

    # 모델 테스트
    model = create_enhanced_time_aware_blending_v10()
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
    print(f"  목표 달성: {'✅' if log_loss_val < 0.7 else '❌'}")

    # 성능 요약
    summary = model.get_performance_summary()
    print(f"\n🔍 모델 상태:")
    for key, value in summary.items():
        print(f"  {key}: {value}")