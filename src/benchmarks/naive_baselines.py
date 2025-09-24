#!/usr/bin/env python3
"""
📊 Naive Baseline 모델 시스템
학술 논문을 위한 기본 베이스라인 모델 구현

주요 모델:
- Random Walk
- Historical Mean
- Previous Value
- Random Prediction
- Trend Following
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RandomWalkBaseline(BaseEstimator, RegressorMixin):
    """
    Random Walk 베이스라인 모델

    다음 수익률이 0이라고 가정 (가격 변화 없음)
    금융 시계열의 가장 기본적인 null hypothesis
    """

    def __init__(self):
        """초기화"""
        self.fitted_ = False
        self.feature_names_in_ = None

    def fit(self, X, y):
        """
        모델 훈련 (실제로는 아무것도 학습하지 않음)

        Args:
            X: 특징 데이터 (사용하지 않음)
            y: 타겟 데이터 (사용하지 않음)

        Returns:
            self
        """
        self.fitted_ = True
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def predict(self, X):
        """
        예측 수행 (항상 0 반환)

        Args:
            X: 특징 데이터

        Returns:
            0으로 구성된 예측 배열
        """
        if not self.fitted_:
            raise ValueError("모델이 훈련되지 않았습니다. fit()을 먼저 호출하세요.")

        return np.zeros(len(X))

class HistoricalMeanBaseline(BaseEstimator, RegressorMixin):
    """
    Historical Mean 베이스라인 모델

    훈련 데이터의 평균 수익률을 모든 예측에 사용
    """

    def __init__(self):
        """초기화"""
        self.mean_return_ = None
        self.fitted_ = False
        self.feature_names_in_ = None

    def fit(self, X, y):
        """
        모델 훈련 (평균 수익률 계산)

        Args:
            X: 특징 데이터 (사용하지 않음)
            y: 타겟 데이터 (수익률)

        Returns:
            self
        """
        self.mean_return_ = np.mean(y)
        self.fitted_ = True
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def predict(self, X):
        """
        예측 수행 (평균 수익률 반환)

        Args:
            X: 특징 데이터

        Returns:
            평균 수익률로 구성된 예측 배열
        """
        if not self.fitted_:
            raise ValueError("모델이 훈련되지 않았습니다. fit()을 먼저 호출하세요.")

        return np.full(len(X), self.mean_return_)

class PreviousValueBaseline(BaseEstimator, RegressorMixin):
    """
    Previous Value 베이스라인 모델

    이전 기간의 수익률을 다음 기간 예측으로 사용
    """

    def __init__(self):
        """초기화"""
        self.fitted_ = False
        self.feature_names_in_ = None
        self.last_value_ = 0

    def fit(self, X, y):
        """
        모델 훈련 (마지막 값 저장)

        Args:
            X: 특징 데이터 (사용하지 않음)
            y: 타겟 데이터 (수익률)

        Returns:
            self
        """
        self.last_value_ = y.iloc[-1] if hasattr(y, 'iloc') else y[-1]
        self.fitted_ = True
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def predict(self, X):
        """
        예측 수행 (마지막 값 반환)

        Args:
            X: 특징 데이터

        Returns:
            마지막 값으로 구성된 예측 배열
        """
        if not self.fitted_:
            raise ValueError("모델이 훈련되지 않았습니다. fit()을 먼저 호출하세요.")

        return np.full(len(X), self.last_value_)

class RandomPredictionBaseline(BaseEstimator, RegressorMixin):
    """
    Random Prediction 베이스라인 모델

    훈련 데이터와 같은 분포에서 랜덤하게 예측
    """

    def __init__(self, random_state=42):
        """
        초기화

        Args:
            random_state: 랜덤 시드
        """
        self.random_state = random_state
        self.mean_ = None
        self.std_ = None
        self.fitted_ = False
        self.feature_names_in_ = None

    def fit(self, X, y):
        """
        모델 훈련 (분포 파라미터 계산)

        Args:
            X: 특징 데이터 (사용하지 않음)
            y: 타겟 데이터 (수익률)

        Returns:
            self
        """
        self.mean_ = np.mean(y)
        self.std_ = np.std(y)
        self.fitted_ = True
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def predict(self, X):
        """
        예측 수행 (정규분포에서 랜덤 샘플링)

        Args:
            X: 특징 데이터

        Returns:
            랜덤 예측 배열
        """
        if not self.fitted_:
            raise ValueError("모델이 훈련되지 않았습니다. fit()을 먼저 호출하세요.")

        np.random.seed(self.random_state)
        return np.random.normal(self.mean_, self.std_, len(X))

class MovingAverageBaseline(BaseEstimator, RegressorMixin):
    """
    Moving Average 베이스라인 모델

    최근 N개 값의 평균을 예측으로 사용
    """

    def __init__(self, window=5):
        """
        초기화

        Args:
            window: 이동평균 윈도우 크기
        """
        self.window = window
        self.fitted_ = False
        self.feature_names_in_ = None
        self.recent_values_ = None

    def fit(self, X, y):
        """
        모델 훈련 (최근 값들 저장)

        Args:
            X: 특징 데이터 (사용하지 않음)
            y: 타겟 데이터 (수익률)

        Returns:
            self
        """
        # 마지막 window개 값 저장
        if len(y) >= self.window:
            self.recent_values_ = y[-self.window:] if hasattr(y, 'iloc') else y[-self.window:]
        else:
            self.recent_values_ = y

        self.fitted_ = True
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def predict(self, X):
        """
        예측 수행 (최근 값들의 평균)

        Args:
            X: 특징 데이터

        Returns:
            이동평균 예측 배열
        """
        if not self.fitted_:
            raise ValueError("모델이 훈련되지 않았습니다. fit()을 먼저 호출하세요.")

        ma_value = np.mean(self.recent_values_)
        return np.full(len(X), ma_value)

class TrendFollowingBaseline(BaseEstimator, RegressorMixin):
    """
    Trend Following 베이스라인 모델

    최근 트렌드 방향으로 예측
    """

    def __init__(self, lookback=5):
        """
        초기화

        Args:
            lookback: 트렌드 계산을 위한 lookback 기간
        """
        self.lookback = lookback
        self.fitted_ = False
        self.feature_names_in_ = None
        self.trend_direction_ = None
        self.trend_magnitude_ = None

    def fit(self, X, y):
        """
        모델 훈련 (트렌드 방향 계산)

        Args:
            X: 특징 데이터 (사용하지 않음)
            y: 타겟 데이터 (수익률)

        Returns:
            self
        """
        if len(y) >= self.lookback:
            recent_values = y[-self.lookback:] if hasattr(y, 'iloc') else y[-self.lookback:]

            # 선형 회귀로 트렌드 계산
            x_vals = np.arange(len(recent_values))
            slope = np.polyfit(x_vals, recent_values, 1)[0]

            self.trend_direction_ = 1 if slope > 0 else -1
            self.trend_magnitude_ = abs(slope)
        else:
            self.trend_direction_ = 0
            self.trend_magnitude_ = 0

        self.fitted_ = True
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def predict(self, X):
        """
        예측 수행 (트렌드 방향으로 예측)

        Args:
            X: 특징 데이터

        Returns:
            트렌드 기반 예측 배열
        """
        if not self.fitted_:
            raise ValueError("모델이 훈련되지 않았습니다. fit()을 먼저 호출하세요.")

        prediction_value = self.trend_direction_ * self.trend_magnitude_ * 0.5  # 보수적 예측
        return np.full(len(X), prediction_value)

class NaiveBaselineEvaluator:
    """
    Naive Baseline 모델들의 성능 평가 시스템
    """

    def __init__(self):
        """초기화"""
        self.models = {
            'Random Walk': RandomWalkBaseline(),
            'Historical Mean': HistoricalMeanBaseline(),
            'Previous Value': PreviousValueBaseline(),
            'Random Prediction': RandomPredictionBaseline(),
            'Moving Average (5)': MovingAverageBaseline(window=5),
            'Moving Average (20)': MovingAverageBaseline(window=20),
            'Trend Following': TrendFollowingBaseline()
        }

        self.results_ = {}

    def evaluate_all(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        모든 naive baseline 모델 평가

        Args:
            X_train: 훈련 특징 데이터
            y_train: 훈련 타겟 데이터
            X_test: 테스트 특징 데이터
            y_test: 테스트 타겟 데이터

        Returns:
            모든 모델의 성능 결과 딕셔너리
        """
        results = {}

        for model_name, model in self.models.items():
            try:
                # 모델 훈련
                model.fit(X_train, y_train)

                # 예측
                y_pred = model.predict(X_test)

                # 성능 계산
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                # 방향 정확도 계산
                direction_accuracy = self._calculate_direction_accuracy(y_test, y_pred)

                # MAPE 계산 (0으로 나누기 방지)
                mape = self._calculate_mape(y_test, y_pred)

                results[model_name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'direction_accuracy': direction_accuracy,
                    'n_predictions': len(y_test),
                    'model_type': 'naive_baseline'
                }

            except Exception as e:
                results[model_name] = {
                    'error': str(e),
                    'model_type': 'naive_baseline'
                }

        self.results_ = results
        return results

    def _calculate_direction_accuracy(self, y_true, y_pred) -> float:
        """방향 정확도 계산"""
        if len(y_true) <= 1:
            return 0.5

        # 실제 방향 (상승/하락)
        true_directions = np.sign(y_true)
        pred_directions = np.sign(y_pred)

        # 0인 경우 처리 (변화 없음)
        true_directions[true_directions == 0] = 1
        pred_directions[pred_directions == 0] = 1

        accuracy = np.mean(true_directions == pred_directions)
        return accuracy * 100  # 백분율 변환

    def _calculate_mape(self, y_true, y_pred) -> float:
        """MAPE 계산 (0으로 나누기 방지)"""
        mask = y_true != 0
        if not np.any(mask):
            return np.inf

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def get_best_baseline(self, metric='mae') -> Tuple[str, Dict]:
        """
        지정된 지표 기준으로 최고 성능 베이스라인 반환

        Args:
            metric: 평가 지표 ('mae', 'mse', 'rmse', 'r2', 'direction_accuracy')

        Returns:
            (모델명, 성능 결과) 튜플
        """
        if not self.results_:
            raise ValueError("evaluate_all()을 먼저 실행하세요.")

        valid_results = {k: v for k, v in self.results_.items() if 'error' not in v}

        if not valid_results:
            raise ValueError("유효한 결과가 없습니다.")

        if metric == 'r2' or metric == 'direction_accuracy':
            # 높을수록 좋은 지표
            best_model = max(valid_results.items(), key=lambda x: x[1][metric])
        else:
            # 낮을수록 좋은 지표
            best_model = min(valid_results.items(), key=lambda x: x[1][metric])

        return best_model

    def generate_baseline_report(self) -> str:
        """베이스라인 성능 보고서 생성"""
        if not self.results_:
            return "결과가 없습니다. evaluate_all()을 먼저 실행하세요."

        report = ["📊 Naive Baseline 모델 성능 보고서", "=" * 50, ""]

        # 성능 표 생성
        headers = ["Model", "MAE", "RMSE", "R²", "Direction Acc(%)", "MAPE(%)"]
        rows = []

        for model_name, result in self.results_.items():
            if 'error' in result:
                rows.append([model_name, "ERROR", "", "", "", ""])
            else:
                rows.append([
                    model_name,
                    f"{result['mae']:.6f}",
                    f"{result['rmse']:.6f}",
                    f"{result['r2']:.4f}",
                    f"{result['direction_accuracy']:.1f}",
                    f"{result['mape']:.1f}" if not np.isinf(result['mape']) else "∞"
                ])

        # 표 포맷팅
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

        # 헤더
        header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
        report.append(header_line)
        report.append("-" * len(header_line))

        # 데이터 행들
        for row in rows:
            data_line = " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
            report.append(data_line)

        report.append("")

        # 최고 성능 모델들
        try:
            best_mae = self.get_best_baseline('mae')
            best_r2 = self.get_best_baseline('r2')
            best_direction = self.get_best_baseline('direction_accuracy')

            report.append("🏆 최고 성능 베이스라인:")
            report.append(f"   MAE 기준: {best_mae[0]} (MAE: {best_mae[1]['mae']:.6f})")
            report.append(f"   R² 기준: {best_r2[0]} (R²: {best_r2[1]['r2']:.4f})")
            report.append(f"   방향 정확도 기준: {best_direction[0]} ({best_direction[1]['direction_accuracy']:.1f}%)")

        except Exception as e:
            report.append(f"최고 성능 계산 오류: {str(e)}")

        return "\n".join(report)

def main():
    """테스트 및 예제 실행"""
    print("📊 Naive Baseline 모델 테스트")
    print("=" * 50)

    # 테스트 데이터 생성
    np.random.seed(42)

    # 시뮬레이션 데이터 (SPY 수익률과 유사)
    n_samples = 500
    returns = np.random.normal(0.0005, 0.015, n_samples)  # 일일 수익률

    # 약간의 트렌드와 자기상관 추가
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1] + 0.05 * np.random.normal(0, 0.01)

    # 특징 데이터 (실제로는 사용되지 않음)
    X = np.random.randn(n_samples, 5)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])

    # 훈련/테스트 분할
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_df[:split_idx], X_df[split_idx:]
    y_train, y_test = returns[:split_idx], returns[split_idx:]

    # 베이스라인 평가기 생성 및 실행
    evaluator = NaiveBaselineEvaluator()
    results = evaluator.evaluate_all(X_train, y_train, X_test, y_test)

    # 보고서 출력
    report = evaluator.generate_baseline_report()
    print(report)

    # 개별 모델 테스트
    print(f"\n📈 개별 모델 테스트 예제:")
    print("-" * 40)

    # Random Walk 모델
    rw_model = RandomWalkBaseline()
    rw_model.fit(X_train, y_train)
    rw_pred = rw_model.predict(X_test[:5])
    print(f"Random Walk 예측 (첫 5개): {rw_pred}")

    # Historical Mean 모델
    hm_model = HistoricalMeanBaseline()
    hm_model.fit(X_train, y_train)
    hm_pred = hm_model.predict(X_test[:5])
    print(f"Historical Mean 예측 (첫 5개): {hm_pred}")

    # 실제 값과 비교
    print(f"실제 값 (첫 5개): {y_test[:5]}")

if __name__ == "__main__":
    main()