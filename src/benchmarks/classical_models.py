#!/usr/bin/env python3
"""
📊 Classical 머신러닝 모델 벤치마크 시스템
학술 논문을 위한 전통적인 ML 모델 구현

주요 모델:
- Linear Regression
- Ridge Regression
- LASSO Regression
- Elastic Net
- Support Vector Regression
- Decision Tree
- Random Forest
- ARIMA
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class ClassicalModelBenchmark:
    """
    전통적인 머신러닝 모델 벤치마크 시스템

    다양한 classical 모델의 성능을 비교하고 평가
    """

    def __init__(self, random_state=42, scale_features=True):
        """
        초기화

        Args:
            random_state: 랜덤 시드
            scale_features: 특징 스케일링 여부
        """
        self.random_state = random_state
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.results_ = {}

        # 모델 정의
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=random_state),
            'LASSO Regression': Lasso(alpha=0.1, random_state=random_state, max_iter=2000),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=2000),
            'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'SVR (Linear)': SVR(kernel='linear', C=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state, max_depth=10),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                max_depth=10,
                n_jobs=-1
            )
        }

        # 하이퍼파라미터 그리드 정의
        self.param_grids = {
            'Ridge Regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'LASSO Regression': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'Elastic Net': {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            },
            'SVR (RBF)': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'SVR (Linear)': {
                'C': [0.1, 1.0, 10.0, 100.0]
            },
            'Decision Tree': {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5]
            }
        }

    def evaluate_all_models(self, X_train, y_train, X_test, y_test,
                           tune_hyperparameters=False, cv_folds=3) -> Dict:
        """
        모든 classical 모델 평가

        Args:
            X_train: 훈련 특징 데이터
            y_train: 훈련 타겟 데이터
            X_test: 테스트 특징 데이터
            y_test: 테스트 타겟 데이터
            tune_hyperparameters: 하이퍼파라미터 튜닝 여부
            cv_folds: 교차 검증 폴드 수

        Returns:
            모든 모델의 성능 결과 딕셔너리
        """
        results = {}

        # 특징 스케일링
        if self.scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        for model_name, model in self.models.items():
            try:
                print(f"평가 중: {model_name}")

                if tune_hyperparameters and model_name in self.param_grids:
                    # 하이퍼파라미터 튜닝
                    tscv = TimeSeriesSplit(n_splits=cv_folds)
                    grid_search = GridSearchCV(
                        model,
                        self.param_grids[model_name],
                        cv=tscv,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1 if model_name != 'SVR (RBF)' else 1  # SVR은 병렬 처리 제한
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    cv_score = -grid_search.best_score_
                else:
                    # 기본 하이퍼파라미터 사용
                    model.fit(X_train_scaled, y_train)
                    best_model = model
                    best_params = {}
                    cv_score = None

                # 예측 및 성능 계산
                y_pred = best_model.predict(X_test_scaled)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                # 방향 정확도 계산
                direction_accuracy = self._calculate_direction_accuracy(y_test, y_pred)

                # MAPE 계산
                mape = self._calculate_mape(y_test, y_pred)

                # 특징 중요도 (가능한 경우)
                feature_importance = self._get_feature_importance(best_model, X_train.columns if hasattr(X_train, 'columns') else None)

                results[model_name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'direction_accuracy': direction_accuracy,
                    'cv_score': cv_score,
                    'best_params': best_params,
                    'feature_importance': feature_importance,
                    'n_predictions': len(y_test),
                    'model_type': 'classical_ml',
                    'hyperparameter_tuned': tune_hyperparameters and model_name in self.param_grids
                }

            except Exception as e:
                print(f"오류 발생 ({model_name}): {str(e)}")
                results[model_name] = {
                    'error': str(e),
                    'model_type': 'classical_ml'
                }

        self.results_ = results
        return results

    def evaluate_arima_models(self, y_train, y_test, max_p=3, max_d=2, max_q=3) -> Dict:
        """
        ARIMA 모델 평가

        Args:
            y_train: 훈련 시계열 데이터
            y_test: 테스트 시계열 데이터
            max_p: 최대 AR 차수
            max_d: 최대 차분 차수
            max_q: 최대 MA 차수

        Returns:
            ARIMA 모델 결과 딕셔너리
        """
        results = {}

        try:
            # 정상성 검정
            adf_result = adfuller(y_train)
            is_stationary = adf_result[1] < 0.05

            print(f"ADF 검정 p-value: {adf_result[1]:.6f}")
            print(f"시계열 정상성: {'정상' if is_stationary else '비정상'}")

            # 최적 ARIMA 차수 찾기
            best_aic = np.inf
            best_order = None
            best_model = None

            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(y_train, order=(p, d, q))
                            fitted_model = model.fit()

                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                                best_model = fitted_model

                        except Exception:
                            continue

            if best_model is not None:
                # 예측 수행
                forecast = best_model.forecast(steps=len(y_test))

                # 성능 계산
                mae = mean_absolute_error(y_test, forecast)
                mse = mean_squared_error(y_test, forecast)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, forecast)

                direction_accuracy = self._calculate_direction_accuracy(y_test, forecast)
                mape = self._calculate_mape(y_test, forecast)

                results['ARIMA'] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'direction_accuracy': direction_accuracy,
                    'best_order': best_order,
                    'aic': best_aic,
                    'bic': best_model.bic,
                    'is_stationary': is_stationary,
                    'adf_pvalue': adf_result[1],
                    'n_predictions': len(y_test),
                    'model_type': 'time_series'
                }

            else:
                results['ARIMA'] = {
                    'error': 'ARIMA 모델 최적화 실패',
                    'model_type': 'time_series'
                }

        except Exception as e:
            results['ARIMA'] = {
                'error': str(e),
                'model_type': 'time_series'
            }

        return results

    def _calculate_direction_accuracy(self, y_true, y_pred) -> float:
        """방향 정확도 계산"""
        if len(y_true) <= 1:
            return 0.5

        true_directions = np.sign(y_true)
        pred_directions = np.sign(y_pred)

        # 0인 경우 처리
        true_directions[true_directions == 0] = 1
        pred_directions[pred_directions == 0] = 1

        accuracy = np.mean(true_directions == pred_directions)
        return accuracy * 100

    def _calculate_mape(self, y_true, y_pred) -> float:
        """MAPE 계산"""
        mask = y_true != 0
        if not np.any(mask):
            return np.inf

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def _get_feature_importance(self, model, feature_names=None):
        """특징 중요도 추출"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based 모델
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear 모델
                importance = np.abs(model.coef_)
            else:
                return None

            if feature_names is not None:
                return dict(zip(feature_names, importance))
            else:
                return importance.tolist()

        except Exception:
            return None

    def get_best_classical_model(self, metric='mae') -> Tuple[str, Dict]:
        """
        지정된 지표 기준으로 최고 성능 classical 모델 반환

        Args:
            metric: 평가 지표

        Returns:
            (모델명, 성능 결과) 튜플
        """
        if not self.results_:
            raise ValueError("evaluate_all_models()를 먼저 실행하세요.")

        valid_results = {k: v for k, v in self.results_.items() if 'error' not in v}

        if not valid_results:
            raise ValueError("유효한 결과가 없습니다.")

        if metric == 'r2' or metric == 'direction_accuracy':
            best_model = max(valid_results.items(), key=lambda x: x[1][metric])
        else:
            best_model = min(valid_results.items(), key=lambda x: x[1][metric])

        return best_model

    def generate_classical_report(self) -> str:
        """Classical 모델 성능 보고서 생성"""
        if not self.results_:
            return "결과가 없습니다. evaluate_all_models()를 먼저 실행하세요."

        report = ["📊 Classical 머신러닝 모델 성능 보고서", "=" * 60, ""]

        # 성능 표 생성
        headers = ["Model", "MAE", "RMSE", "R²", "Dir Acc(%)", "MAPE(%)", "Tuned"]
        rows = []

        for model_name, result in self.results_.items():
            if 'error' in result:
                rows.append([model_name, "ERROR", "", "", "", "", ""])
            else:
                tuned = "Yes" if result.get('hyperparameter_tuned', False) else "No"
                rows.append([
                    model_name,
                    f"{result['mae']:.6f}",
                    f"{result['rmse']:.6f}",
                    f"{result['r2']:.4f}",
                    f"{result['direction_accuracy']:.1f}",
                    f"{result['mape']:.1f}" if not np.isinf(result['mape']) else "∞",
                    tuned
                ])

        # 표 포맷팅
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

        header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
        report.append(header_line)
        report.append("-" * len(header_line))

        for row in rows:
            data_line = " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
            report.append(data_line)

        report.append("")

        # 최고 성능 모델들
        try:
            best_mae = self.get_best_classical_model('mae')
            best_r2 = self.get_best_classical_model('r2')
            best_direction = self.get_best_classical_model('direction_accuracy')

            report.append("🏆 최고 성능 Classical 모델:")
            report.append(f"   MAE 기준: {best_mae[0]} (MAE: {best_mae[1]['mae']:.6f})")
            report.append(f"   R² 기준: {best_r2[0]} (R²: {best_r2[1]['r2']:.4f})")
            report.append(f"   방향 정확도 기준: {best_direction[0]} ({best_direction[1]['direction_accuracy']:.1f}%)")

        except Exception as e:
            report.append(f"최고 성능 계산 오류: {str(e)}")

        # 특징 중요도 정보 (Random Forest 기준)
        rf_result = self.results_.get('Random Forest')
        if rf_result and 'feature_importance' in rf_result and rf_result['feature_importance']:
            report.append("")
            report.append("🔍 특징 중요도 (Random Forest):")

            if isinstance(rf_result['feature_importance'], dict):
                # 중요도별 정렬
                sorted_features = sorted(rf_result['feature_importance'].items(),
                                       key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:10]):  # 상위 10개
                    report.append(f"   {i+1}. {feature}: {importance:.4f}")

        return "\n".join(report)

def main():
    """테스트 및 예제 실행"""
    print("📊 Classical 머신러닝 모델 벤치마크 테스트")
    print("=" * 60)

    # 테스트 데이터 생성
    np.random.seed(42)

    # SPY 수익률과 유사한 시뮬레이션 데이터
    n_samples = 300
    n_features = 8

    # 특징 데이터 생성 (기술적 지표 시뮬레이션)
    X = np.random.randn(n_samples, n_features)
    feature_names = ['MA_5', 'MA_20', 'RSI', 'BB_position', 'Volume_ratio',
                    'Returns_lag_1', 'Returns_lag_2', 'Volatility_20']

    X_df = pd.DataFrame(X, columns=feature_names)

    # 타겟 데이터 생성 (약간의 패턴 포함)
    y = 0.0005 + 0.1 * X[:, 0] + 0.05 * X[:, 1] + np.random.normal(0, 0.015, n_samples)

    # 훈련/테스트 분할 (시계열이므로 시간순)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_df[:split_idx], X_df[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Classical 모델 벤치마크 실행
    benchmark = ClassicalModelBenchmark(random_state=42, scale_features=True)

    print("기본 하이퍼파라미터로 평가 중...")
    results = benchmark.evaluate_all_models(
        X_train, y_train, X_test, y_test,
        tune_hyperparameters=False
    )

    # ARIMA 모델 평가
    print("\nARIMA 모델 평가 중...")
    arima_results = benchmark.evaluate_arima_models(y_train, y_test)
    results.update(arima_results)

    # 보고서 생성
    benchmark.results_.update(arima_results)
    report = benchmark.generate_classical_report()
    print("\n" + report)

    # 하이퍼파라미터 튜닝 예제 (소규모)
    print(f"\n🔧 하이퍼파라미터 튜닝 예제 (Ridge Regression):")
    print("-" * 50)

    tuned_benchmark = ClassicalModelBenchmark(random_state=42)
    tuned_benchmark.models = {'Ridge Regression': tuned_benchmark.models['Ridge Regression']}

    tuned_results = tuned_benchmark.evaluate_all_models(
        X_train, y_train, X_test, y_test,
        tune_hyperparameters=True, cv_folds=3
    )

    ridge_result = tuned_results['Ridge Regression']
    print(f"최적 하이퍼파라미터: {ridge_result['best_params']}")
    print(f"CV MAE: {ridge_result['cv_score']:.6f}")
    print(f"Test MAE: {ridge_result['mae']:.6f}")

if __name__ == "__main__":
    main()