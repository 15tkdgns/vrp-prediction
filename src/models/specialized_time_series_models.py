#!/usr/bin/env python3
"""
🎯 전문 시계열 회귀 모델들

ARIMA, VAR, State Space Models 등 전문 시계열 알고리즘 적용
"""

import sys
sys.path.append('/root/workspace/src')

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Core imports
from core.data_processor import DataProcessor

# Standard ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Specialized time series imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels 미설치 - 일부 시계열 모델 제외")

# Advanced time series imports
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

# Facebook Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet 미설치 - Prophet 모델 제외")

class SpecializedTimeSeriesModels:
    """전문 시계열 모델 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.max_allowed_correlation = 0.15  # 극도로 엄격한 기준
        self.target_r2_threshold = 0.15      # 더 높은 R² 목표
        self.target_mae_threshold = 0.008    # 더 낮은 MAE 목표

        print(f"🎯 전문 시계열 모델 연구 시스템")
        print(f"   🎯 목표: R²>{self.target_r2_threshold}, MAE<{self.target_mae_threshold}")
        print(f"   🔒 최대 허용 상관관계: {self.max_allowed_correlation}")

    def create_specialized_features(self, df):
        """전문 시계열 특성 생성"""
        print("🔧 전문 시계열 특성 생성...")

        ts_df = df.copy()

        # 기본 수익률
        ts_df['returns'] = ts_df['Close'].pct_change()
        ts_df['log_returns'] = np.log(ts_df['Close'] / ts_df['Close'].shift(1))

        # 1. 시계열 분해 기반 특성
        if len(ts_df) > 50:  # 충분한 데이터가 있을 때만
            try:
                # STL 분해 (계절성, 추세, 잔차)
                if STATSMODELS_AVAILABLE:
                    stl = STL(ts_df['Close'].fillna(method='ffill'), seasonal=13)
                    result = stl.fit()

                    ts_df['trend'] = result.trend
                    ts_df['seasonal'] = result.seasonal
                    ts_df['residual'] = result.resid

                    # 분해 성분 기반 특성
                    ts_df['trend_strength'] = ts_df['trend'] / ts_df['Close']
                    ts_df['seasonal_strength'] = ts_df['seasonal'] / ts_df['Close']
                    ts_df['residual_strength'] = ts_df['residual'] / ts_df['Close']
            except:
                pass

        # 2. 고급 래그 특성 (자기상관 기반)
        optimal_lags = [1, 2, 3, 5, 8, 13, 21]  # 피보나치 기반 래그
        for lag in optimal_lags:
            ts_df[f'returns_lag_{lag}'] = ts_df['returns'].shift(lag)
            ts_df[f'log_returns_lag_{lag}'] = ts_df['log_returns'].shift(lag)

            # 래그간 차분
            if lag > 1:
                ts_df[f'returns_diff_{lag}'] = ts_df['returns'] - ts_df['returns'].shift(lag)

        # 3. 이동통계 특성 (다양한 윈도우)
        windows = [5, 10, 21, 50, 100]
        for window in windows:
            # 이동평균 기반
            ma = ts_df['returns'].rolling(window).mean()
            ts_df[f'ma_{window}'] = ma
            ts_df[f'returns_vs_ma_{window}'] = ts_df['returns'] - ma

            # 이동분산 기반
            ts_df[f'rolling_vol_{window}'] = ts_df['returns'].rolling(window).std()

            # 이동첨도/왜도 (scipy.stats 사용)
            from scipy.stats import skew, kurtosis
            ts_df[f'rolling_skew_{window}'] = ts_df['returns'].rolling(window).apply(lambda x: skew(x, nan_policy='omit'))
            ts_df[f'rolling_kurt_{window}'] = ts_df['returns'].rolling(window).apply(lambda x: kurtosis(x, nan_policy='omit'))

            # 이동 최대/최소
            ts_df[f'rolling_max_{window}'] = ts_df['returns'].rolling(window).max()
            ts_df[f'rolling_min_{window}'] = ts_df['returns'].rolling(window).min()

        # 4. 변동성 클러스터링 특성
        for window in [10, 20, 50]:
            vol = ts_df['returns'].rolling(window).std()
            ts_df[f'vol_regime_{window}'] = (vol > vol.rolling(100).quantile(0.75)).astype(int)

            # GARCH-like 특성
            ts_df[f'vol_persistence_{window}'] = vol / vol.shift(1)

        # 5. 모멘텀 및 반전 특성
        for period in [5, 10, 20]:
            # 가격 모멘텀
            ts_df[f'price_momentum_{period}'] = (
                ts_df['Close'] / ts_df['Close'].shift(period) - 1
            )

            # 수익률 모멘텀
            ts_df[f'return_momentum_{period}'] = (
                ts_df['returns'].rolling(period).sum()
            )

            # 반전 신호
            ts_df[f'reversal_{period}'] = -ts_df[f'return_momentum_{period}'].shift(1)

        # 6. 볼륨 기반 고급 특성
        for period in [5, 10, 20]:
            # 볼륨 가중 수익률
            vol_weight = ts_df['Volume'] / ts_df['Volume'].rolling(period).sum()
            ts_df[f'vol_weighted_return_{period}'] = ts_df['returns'] * vol_weight

            # 볼륨 트렌드
            ts_df[f'volume_trend_{period}'] = (
                ts_df['Volume'] / ts_df['Volume'].rolling(period).mean() - 1
            )

        # 7. 프랙탈 및 복잡성 특성
        for window in [20, 50]:
            returns_window = ts_df['returns'].rolling(window)

            # 허스트 지수 근사
            ts_df[f'hurst_approx_{window}'] = self._estimate_hurst(returns_window)

            # 엔트로피 근사
            ts_df[f'entropy_approx_{window}'] = returns_window.apply(self._estimate_entropy)

        # 8. 타겟 변수들
        ts_df['target_return_1d'] = ts_df['returns'].shift(-1)
        ts_df['target_return_3d'] = ts_df['Close'].pct_change(3).shift(-3)
        ts_df['target_return_5d'] = ts_df['Close'].pct_change(5).shift(-5)

        # NaN 처리
        ts_df = ts_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        ts_df = ts_df.replace([np.inf, -np.inf], 0)

        print(f"   ✅ 전문 시계열 특성 생성 완료: {ts_df.shape}")
        return ts_df

    def _estimate_hurst(self, series):
        """허스트 지수 추정"""
        try:
            if len(series) < 20:
                return 0.5
            lags = range(2, min(20, len(series)//2))
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5

    def _estimate_entropy(self, series):
        """엔트로피 추정"""
        try:
            _, counts = np.unique(series.round(6), return_counts=True)
            probabilities = counts / len(series)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        except:
            return 0.0

    def create_specialized_models(self):
        """전문 시계열 모델들 생성"""
        print("🎯 전문 시계열 모델들 생성...")

        models = {}

        # 1. ARIMA 모델들 (statsmodels)
        if STATSMODELS_AVAILABLE:
            models['ARIMA_101'] = {
                'type': 'arima',
                'params': {'order': (1, 0, 1)}
            }

            models['ARIMA_111'] = {
                'type': 'arima',
                'params': {'order': (1, 1, 1)}
            }

            models['ARIMA_212'] = {
                'type': 'arima',
                'params': {'order': (2, 1, 2)}
            }

            # SARIMAX 모델
            models['SARIMAX'] = {
                'type': 'sarimax',
                'params': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 5)}
            }

        # 2. 지수평활법 모델들
        if STATSMODELS_AVAILABLE:
            models['ExponentialSmoothing'] = {
                'type': 'exp_smoothing',
                'params': {'trend': 'add', 'seasonal': None}
            }

        # 3. VAR 모델 (다변량)
        if STATSMODELS_AVAILABLE:
            models['VAR_2'] = {
                'type': 'var',
                'params': {'maxlags': 2}
            }

        # 4. 가우시안 프로세스 시계열 모델들
        if GP_AVAILABLE:
            # RBF 커널 (평활함)
            rbf_kernel = RBF(length_scale=1.0)
            models['GP_RBF'] = {
                'type': 'gp',
                'kernel': rbf_kernel
            }

            # Matern 커널 (일반적)
            matern_kernel = Matern(length_scale=1.0, nu=1.5)
            models['GP_Matern'] = {
                'type': 'gp',
                'kernel': matern_kernel
            }

            # 주기적 커널 (계절성)
            periodic_kernel = ExpSineSquared(length_scale=1.0, periodicity=1.0)
            models['GP_Periodic'] = {
                'type': 'gp',
                'kernel': periodic_kernel
            }

            # 복합 커널
            composite_kernel = RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=2.5)
            models['GP_Composite'] = {
                'type': 'gp',
                'kernel': composite_kernel
            }

        # 5. Prophet 모델
        if PROPHET_AVAILABLE:
            models['Prophet_Basic'] = {
                'type': 'prophet',
                'params': {'daily_seasonality': False, 'weekly_seasonality': True}
            }

        # 6. 사용자 정의 시계열 모델들
        models['AutoRegressive_Custom'] = {
            'type': 'custom_ar',
            'params': {'lags': [1, 2, 3, 5]}
        }

        models['MovingAverage_Adaptive'] = {
            'type': 'adaptive_ma',
            'params': {'windows': [5, 10, 20]}
        }

        print(f"   ✅ 생성된 전문 모델: {len(models)}개")
        return models

    def fit_specialized_model(self, model_name, model_config, X_train, y_train, X_val, y_val):
        """전문 모델 훈련 및 예측"""

        model_type = model_config.get('type', 'unknown')

        try:
            if model_type == 'arima' and STATSMODELS_AVAILABLE:
                order = model_config['params']['order']
                model = ARIMA(y_train, order=order)
                fitted_model = model.fit()
                y_pred = fitted_model.forecast(steps=len(y_val))

            elif model_type == 'sarimax' and STATSMODELS_AVAILABLE:
                order = model_config['params']['order']
                seasonal_order = model_config['params']['seasonal_order']
                model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
                fitted_model = model.fit(disp=False)
                y_pred = fitted_model.forecast(steps=len(y_val))

            elif model_type == 'exp_smoothing' and STATSMODELS_AVAILABLE:
                model = ExponentialSmoothing(y_train, **model_config['params'])
                fitted_model = model.fit()
                y_pred = fitted_model.forecast(steps=len(y_val))

            elif model_type == 'var' and STATSMODELS_AVAILABLE:
                # VAR은 다변량이므로 여러 특성 사용
                if X_train.shape[1] >= 2:
                    data = np.column_stack([y_train, X_train[:, :min(3, X_train.shape[1])]])
                    model = VAR(data)
                    fitted_model = model.fit(maxlags=model_config['params']['maxlags'])
                    forecast = fitted_model.forecast(data[-fitted_model.k_ar:], steps=len(y_val))
                    y_pred = forecast[:, 0]  # 첫 번째 변수 (타겟) 예측
                else:
                    y_pred = np.full(len(y_val), np.mean(y_train))

            elif model_type == 'gp' and GP_AVAILABLE:
                kernel = model_config['kernel']
                model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42)

                # 시계열용 인덱스 생성
                X_train_idx = np.arange(len(X_train)).reshape(-1, 1)
                X_val_idx = np.arange(len(X_train), len(X_train) + len(X_val)).reshape(-1, 1)

                model.fit(X_train_idx, y_train)
                y_pred, _ = model.predict(X_val_idx, return_std=True)

            elif model_type == 'prophet' and PROPHET_AVAILABLE:
                # Prophet용 데이터 준비
                train_df = pd.DataFrame({
                    'ds': pd.date_range(start='2020-01-01', periods=len(y_train), freq='D'),
                    'y': y_train
                })

                model = Prophet(**model_config['params'])
                model.fit(train_df)

                future = pd.DataFrame({
                    'ds': pd.date_range(start=train_df['ds'].iloc[-1] + pd.Timedelta(days=1),
                                       periods=len(y_val), freq='D')
                })
                forecast = model.predict(future)
                y_pred = forecast['yhat'].values

            elif model_type == 'custom_ar':
                # 사용자 정의 자기회귀 모델
                lags = model_config['params']['lags']
                y_pred = self._custom_autoregressive_predict(y_train, lags, len(y_val))

            elif model_type == 'adaptive_ma':
                # 적응형 이동평균
                windows = model_config['params']['windows']
                y_pred = self._adaptive_moving_average_predict(y_train, windows, len(y_val))

            else:
                # 기본 예측 (평균)
                y_pred = np.full(len(y_val), np.mean(y_train))

            return y_pred

        except Exception as e:
            print(f"      모델 {model_name} 예측 실패: {e}")
            return np.full(len(y_val), np.mean(y_train))

    def _custom_autoregressive_predict(self, y_train, lags, n_pred):
        """사용자 정의 자기회귀 예측"""
        try:
            # 간단한 선형 자기회귀
            X_ar = []
            y_ar = []

            max_lag = max(lags)
            for i in range(max_lag, len(y_train)):
                X_ar.append([y_train[i-lag] for lag in lags])
                y_ar.append(y_train[i])

            X_ar = np.array(X_ar)
            y_ar = np.array(y_ar)

            # 선형 회귀로 계수 추정
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X_ar, y_ar)

            # 예측
            predictions = []
            current = list(y_train[-max_lag:])

            for _ in range(n_pred):
                X_pred = np.array([[current[-lag] for lag in lags]])
                pred = lr.predict(X_pred)[0]
                predictions.append(pred)
                current.append(pred)

            return np.array(predictions)

        except:
            return np.full(n_pred, np.mean(y_train))

    def _adaptive_moving_average_predict(self, y_train, windows, n_pred):
        """적응형 이동평균 예측"""
        try:
            # 여러 윈도우의 가중평균
            predictions = []

            for _ in range(n_pred):
                window_preds = []
                weights = []

                for window in windows:
                    if len(y_train) >= window:
                        ma_pred = np.mean(y_train[-window:])
                        # 짧은 윈도우에 더 높은 가중치
                        weight = 1.0 / window
                        window_preds.append(ma_pred)
                        weights.append(weight)

                if window_preds:
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    pred = np.average(window_preds, weights=weights)
                else:
                    pred = np.mean(y_train[-5:]) if len(y_train) >= 5 else np.mean(y_train)

                predictions.append(pred)
                y_train = np.append(y_train, pred)  # 예측값을 다음 예측에 사용

            return np.array(predictions)

        except:
            return np.full(n_pred, np.mean(y_train))

    def evaluate_specialized_models(self, models, X, y, safe_features):
        """전문 시계열 모델 평가"""
        print(f"\n📊 전문 시계열 모델 평가 (특성 수: {len(safe_features)})")

        # 데이터 준비
        X_features = X[safe_features].values if len(safe_features) > 0 else np.zeros((len(X), 1))
        y_values = y.values

        # 안전 처리
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)

        # 유효 데이터만 선택
        valid_idx = ~(pd.isna(y) | (y == 0))
        X_features = X_features[valid_idx]
        y_values = y_values[valid_idx]

        print(f"   최종 데이터: X={X_features.shape}, y={y_values.shape}")
        print(f"   타겟 통계: 평균={np.mean(y_values):.6f}, 표준편차={np.std(y_values):.6f}")

        # 시간 순서 분할 (80-20)
        split_idx = int(len(y_values) * 0.8)
        X_train, X_val = X_features[:split_idx], X_features[split_idx:]
        y_train, y_val = y_values[:split_idx], y_values[split_idx:]

        results = {}

        for model_name, model_config in models.items():
            print(f"\n   🔬 {model_name} 평가...")

            try:
                # 모델 훈련 및 예측
                y_pred = self.fit_specialized_model(model_name, model_config, X_train, y_train, X_val, y_val)

                # 성능 계산
                r2 = r2_score(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)

                results[model_name] = {
                    'r2': r2,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'model_type': model_config.get('type', 'unknown'),
                    'target_achieved': {
                        'r2_target': r2 > self.target_r2_threshold,
                        'mae_target': mae < self.target_mae_threshold
                    }
                }

                # 성능 평가
                status = "🎯 목표 달성!" if (r2 > self.target_r2_threshold and mae < self.target_mae_threshold) else "📊 기준치"
                print(f"      ✅ R²={r2:.4f}, MAE={mae:.6f}, RMSE={rmse:.6f} - {status}")

            except Exception as e:
                print(f"      ❌ 평가 실패: {e}")
                results[model_name] = {
                    'r2': -1.0,
                    'mae': 1.0,
                    'mse': 1.0,
                    'rmse': 1.0,
                    'model_type': model_config.get('type', 'unknown'),
                    'target_achieved': {'r2_target': False, 'mae_target': False},
                    'error': str(e)
                }

        return results

    def run_specialized_research(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """전문 시계열 연구 실행"""
        print("🎯 전문 시계열 모델 연구 시작")
        print("="*80)

        try:
            # 1. 데이터 로딩 및 전문 특성 생성
            df = self.data_processor.load_and_validate_data(data_path)
            ts_df = self.create_specialized_features(df)

            # 2. 극엄격 누출 검증
            safe_features = self.validate_ultra_strict_leakage(ts_df, 'target_return_1d')

            if len(safe_features) < 5:
                print("⚠️ 안전 특성이 부족하지만 시계열 모델은 적은 특성으로도 가능")

            print(f"✅ 안전 특성 {len(safe_features)}개 확보")

            # 3. 전문 모델들 생성
            models = self.create_specialized_models()

            # 4. 성능 평가
            X = ts_df[safe_features] if safe_features else ts_df[['returns']].fillna(0)
            y = ts_df['target_return_1d']

            results = self.evaluate_specialized_models(models, X, y, safe_features)

            # 5. 결과 분석 및 저장
            self._analyze_specialized_results(results, safe_features)

            return results

        except Exception as e:
            print(f"❌ 전문 시계열 연구 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def validate_ultra_strict_leakage(self, df, target_col='target_return_1d'):
        """극엄격 데이터 누출 검증"""
        print("🔍 극엄격 데이터 누출 검증...")

        exclude_cols = [
            target_col, 'target_return_3d', 'target_return_5d',
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"   검증할 특성 수: {len(feature_cols)}")

        safe_features = []
        suspicious_features = []

        for feature in feature_cols:
            if feature in df.columns and target_col in df.columns:
                corr = abs(df[feature].corr(df[target_col]))
                if not pd.isna(corr):
                    if corr > self.max_allowed_correlation:
                        suspicious_features.append((feature, corr))
                        print(f"   ⚠️ 의심 특성: {feature} (상관관계: {corr:.4f})")
                    else:
                        safe_features.append(feature)

        if suspicious_features:
            print(f"   🚨 의심 특성 {len(suspicious_features)}개 제거!")
        else:
            print("   ✅ 모든 특성이 극엄격 기준 통과")

        return safe_features

    def _analyze_specialized_results(self, results, safe_features):
        """전문 시계열 결과 분석"""
        print("\n📋 전문 시계열 연구 결과 분석")
        print("="*60)

        # R² 순위
        sorted_by_r2 = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)

        print(f"\n🏆 R² 성능 순위 (전문 시계열 모델):")
        for rank, (model_name, metrics) in enumerate(sorted_by_r2[:5], 1):
            r2 = metrics['r2']
            mae = metrics['mae']
            model_type = metrics['model_type']
            target_status = "🎯" if metrics['target_achieved']['r2_target'] and metrics['target_achieved']['mae_target'] else "📊"

            print(f"   {rank}. {model_name} ({model_type}): R²={r2:.4f}, MAE={mae:.6f} {target_status}")

        # 모델 타입별 최고 성능
        type_best = {}
        for model_name, metrics in results.items():
            model_type = metrics['model_type']
            if model_type not in type_best or metrics['r2'] > type_best[model_type]['r2']:
                type_best[model_type] = {**metrics, 'name': model_name}

        print(f"\n📊 모델 타입별 최고 성능:")
        for model_type, best_metrics in type_best.items():
            print(f"   {model_type}: {best_metrics['name']} - R²={best_metrics['r2']:.4f}, MAE={best_metrics['mae']:.6f}")

        # 목표 달성 모델들
        target_achieved = [(name, metrics) for name, metrics in results.items()
                          if metrics['target_achieved']['r2_target'] and metrics['target_achieved']['mae_target']]

        if target_achieved:
            print(f"\n🎉 목표 달성 모델들 ({len(target_achieved)}개):")
            for name, metrics in target_achieved:
                print(f"   ✅ {name}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.6f}")
        else:
            print(f"\n⚠️ 목표 달성 모델 없음 - 추가 연구 필요")

        # 결과 저장
        output_path = f"/root/workspace/data/results/specialized_time_series_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(output_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'experiment_type': 'specialized_time_series_models',
                    'target_thresholds': {
                        'r2_threshold': self.target_r2_threshold,
                        'mae_threshold': self.target_mae_threshold
                    },
                    'safe_features_count': len(safe_features),
                    'max_allowed_correlation': self.max_allowed_correlation,
                    'results': results
                }, f, indent=2)
            print(f"\n💾 전문 시계열 결과 저장: {output_path}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 실행"""
    research = SpecializedTimeSeriesModels()
    results = research.run_specialized_research()

    if results:
        print("\n🎉 전문 시계열 모델 연구 완료!")
    else:
        print("\n❌ 전문 시계열 모델 연구 실패!")

    return results

if __name__ == "__main__":
    main()