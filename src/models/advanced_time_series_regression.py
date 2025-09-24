#!/usr/bin/env python3
"""
🚀 고도화된 시계열 회귀 모델 연구

R² 및 MAE 중심 성능 최적화 with 완전 데이터 누출 방지
수익률 직접 예측에 초점
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

# Advanced ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, LassoCV, RidgeCV
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# Advanced regressor imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Time series specific imports
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

class AdvancedTimeSeriesRegression:
    """고도화된 시계열 회귀 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.max_allowed_correlation = 0.2  # 매우 엄격한 기준
        self.target_r2_threshold = 0.1      # R² 목표 임계값
        self.target_mae_threshold = 0.01    # MAE 목표 임계값

        print(f"🚀 고도화된 시계열 회귀 연구 시스템")
        print(f"   🎯 목표: R²>{self.target_r2_threshold}, MAE<{self.target_mae_threshold}")
        print(f"   🔒 최대 허용 상관관계: {self.max_allowed_correlation}")

    def create_advanced_time_series_features(self, df):
        """고도화된 시계열 특성 공학"""
        print("🔧 고도화된 시계열 특성 공학...")

        advanced_df = df.copy()

        # 1. 기본 수익률 및 로그 수익률
        advanced_df['returns'] = advanced_df['Close'].pct_change()
        advanced_df['log_returns'] = np.log(advanced_df['Close'] / advanced_df['Close'].shift(1))

        # 2. 다중 시점 모멘텀 (과거만)
        momentum_periods = [3, 5, 7, 10, 14, 20, 30, 50, 100]
        for period in momentum_periods:
            advanced_df[f'momentum_{period}d'] = (
                advanced_df['Close'] / advanced_df['Close'].shift(period) - 1
            )
            advanced_df[f'log_momentum_{period}d'] = (
                np.log(advanced_df['Close'] / advanced_df['Close'].shift(period))
            )

        # 3. 다중 변동성 측정 (과거만)
        volatility_windows = [5, 10, 15, 20, 30, 50, 100]
        for window in volatility_windows:
            # 단순 변동성
            advanced_df[f'volatility_{window}d'] = (
                advanced_df['returns'].rolling(window).std()
            )
            # 로그 수익률 변동성
            advanced_df[f'log_volatility_{window}d'] = (
                advanced_df['log_returns'].rolling(window).std()
            )
            # EWMA 변동성
            advanced_df[f'ewm_volatility_{window}d'] = (
                advanced_df['returns'].ewm(span=window).std()
            )

        # 4. 고급 기술적 지표
        ta_periods = [5, 10, 14, 20, 30, 50]
        for period in ta_periods:
            # 이동평균 및 비율
            sma = advanced_df['Close'].rolling(period).mean()
            advanced_df[f'sma_{period}d'] = sma
            advanced_df[f'sma_ratio_{period}d'] = advanced_df['Close'] / sma

            # 지수이동평균
            ema = advanced_df['Close'].ewm(span=period).mean()
            advanced_df[f'ema_{period}d'] = ema
            advanced_df[f'ema_ratio_{period}d'] = advanced_df['Close'] / ema

            # 볼린저 밴드
            rolling_std = advanced_df['Close'].rolling(period).std()
            advanced_df[f'bb_upper_{period}d'] = sma + (rolling_std * 2)
            advanced_df[f'bb_lower_{period}d'] = sma - (rolling_std * 2)
            advanced_df[f'bb_position_{period}d'] = (
                (advanced_df['Close'] - advanced_df[f'bb_lower_{period}d']) /
                (advanced_df[f'bb_upper_{period}d'] - advanced_df[f'bb_lower_{period}d'])
            )

            # RSI
            delta = advanced_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            advanced_df[f'rsi_{period}d'] = 100 - (100 / (1 + rs))

        # 5. 볼륨 기반 고급 특성
        volume_periods = [5, 10, 20, 30, 50]
        for period in volume_periods:
            # 볼륨 이동평균
            vol_sma = advanced_df['Volume'].rolling(period).mean()
            advanced_df[f'vol_sma_{period}d'] = vol_sma
            advanced_df[f'vol_ratio_{period}d'] = advanced_df['Volume'] / vol_sma

            # 볼륨 가중 평균 가격 (VWAP)
            vwap_num = (advanced_df['Close'] * advanced_df['Volume']).rolling(period).sum()
            vwap_den = advanced_df['Volume'].rolling(period).sum()
            advanced_df[f'vwap_{period}d'] = vwap_num / vwap_den
            advanced_df[f'vwap_ratio_{period}d'] = advanced_df['Close'] / advanced_df[f'vwap_{period}d']

            # 온밸런스 볼륨 (OBV)
            obv = (advanced_df['Volume'] * np.sign(advanced_df['returns'])).cumsum()
            advanced_df[f'obv_sma_{period}d'] = obv.rolling(period).mean()

        # 6. 가격 레인지 및 갭 특성
        advanced_df['hl_range'] = (advanced_df['High'] - advanced_df['Low']) / advanced_df['Close']
        advanced_df['oc_range'] = (advanced_df['Close'] - advanced_df['Open']) / advanced_df['Open']
        advanced_df['ho_gap'] = (advanced_df['High'] - advanced_df['Open']) / advanced_df['Open']
        advanced_df['ol_gap'] = (advanced_df['Open'] - advanced_df['Low']) / advanced_df['Low']

        # 7. 통계적 특성 (과거만)
        stat_windows = [10, 20, 50]
        for window in stat_windows:
            # 왜도 및 첨도 (scipy.stats 사용)
            from scipy.stats import skew, kurtosis
            advanced_df[f'skew_{window}d'] = advanced_df['returns'].rolling(window).apply(lambda x: skew(x, nan_policy='omit'))
            advanced_df[f'kurtosis_{window}d'] = advanced_df['returns'].rolling(window).apply(lambda x: kurtosis(x, nan_policy='omit'))

            # 분위수
            advanced_df[f'q25_{window}d'] = advanced_df['returns'].rolling(window).quantile(0.25)
            advanced_df[f'q75_{window}d'] = advanced_df['returns'].rolling(window).quantile(0.75)
            advanced_df[f'iqr_{window}d'] = (
                advanced_df[f'q75_{window}d'] - advanced_df[f'q25_{window}d']
            )

        # 8. 래그 특성 (과거만)
        lag_periods = [1, 2, 3, 5, 10, 20]
        for lag in lag_periods:
            advanced_df[f'returns_lag_{lag}'] = advanced_df['returns'].shift(lag)
            advanced_df[f'volume_lag_{lag}'] = advanced_df['Volume'].shift(lag)
            advanced_df[f'volatility_lag_{lag}'] = advanced_df[f'volatility_20d'].shift(lag)

        # 9. 다중 타겟 변수 (미래 정보 - 타겟만)
        # 다음날 수익률 (주 타겟)
        advanced_df['target_return_1d'] = advanced_df['returns'].shift(-1)

        # 다중 시점 수익률 (추가 타겟들)
        advanced_df['target_return_3d'] = advanced_df['Close'].pct_change(3).shift(-3)
        advanced_df['target_return_5d'] = advanced_df['Close'].pct_change(5).shift(-5)

        # 방향 예측 (보조 타겟)
        advanced_df['target_direction'] = (advanced_df['target_return_1d'] > 0).astype(int)

        # NaN 처리
        advanced_df = advanced_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        advanced_df = advanced_df.replace([np.inf, -np.inf], 0)

        print(f"   ✅ 고급 특성 생성 완료: {advanced_df.shape}")
        return advanced_df

    def validate_ultra_strict_leakage(self, df, target_col='target_return_1d'):
        """초엄격 데이터 누출 검증"""
        print("🔍 초엄격 데이터 누출 검증...")

        # 특성 선택 (타겟 및 기본 컬럼 제외)
        exclude_cols = [
            target_col, 'target_return_3d', 'target_return_5d', 'target_direction',
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"   검증할 특성 수: {len(feature_cols)}")

        # 상관관계 검사
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
            print("   ✅ 모든 특성이 초엄격 기준 통과")

        return safe_features

    def create_advanced_regression_models(self):
        """고도화된 회귀 모델들 생성"""
        print("🎯 고도화된 회귀 모델들 생성...")

        models = {}

        # 1. 고급 선형 모델들
        models['Ridge_Optimized'] = RidgeCV(
            alphas=np.logspace(-4, 4, 50),
            cv=TimeSeriesSplit(n_splits=3)
        )

        models['ElasticNet_Advanced'] = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', ElasticNet(
                alpha=0.01, l1_ratio=0.5, max_iter=2000, random_state=42
            ))
        ])

        models['BayesianRidge_Enhanced'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', BayesianRidge(
                alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6
            ))
        ])

        # 2. 앙상블 모델들
        models['RandomForest_Tuned'] = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
        )

        models['ExtraTrees_Optimized'] = ExtraTreesRegressor(
            n_estimators=150, max_depth=10, min_samples_split=3,
            min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
        )

        models['GradientBoosting_Advanced'] = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, max_features='sqrt', random_state=42
        )

        # 3. XGBoost 모델들 (사용 가능한 경우)
        if XGBOOST_AVAILABLE:
            models['XGBoost_Regression'] = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0
            )

            models['XGBoost_Dart'] = xgb.XGBRegressor(
                booster='dart', n_estimators=150, learning_rate=0.1,
                max_depth=5, subsample=0.9, colsample_bytree=0.9,
                random_state=42, n_jobs=-1, verbosity=0
            )

        # 4. LightGBM 모델들 (사용 가능한 경우)
        if LIGHTGBM_AVAILABLE:
            models['LightGBM_Regression'] = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1
            )

        # 5. 가우시안 프로세스 (사용 가능한 경우)
        if GP_AVAILABLE:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            models['GaussianProcess'] = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, random_state=42
            )

        # 6. 타겟 변환 모델들
        models['Ridge_LogTransformed'] = TransformedTargetRegressor(
            regressor=Ridge(alpha=1.0, random_state=42),
            transformer=PowerTransformer(method='yeo-johnson')
        )

        print(f"   ✅ 생성된 모델: {len(models)}개")
        return models

    def evaluate_regression_performance(self, models, X, y, safe_features):
        """회귀 성능 평가 (R² 및 MAE 중심)"""
        print(f"\n📊 회귀 성능 평가 (특성 수: {len(safe_features)})")

        # 데이터 준비
        X_features = X[safe_features].values
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

        # 시간 순서 교차 검증
        tscv = TimeSeriesSplit(n_splits=5)
        results = {}

        for model_name, model in models.items():
            print(f"\n   🔬 {model_name} 평가...")

            fold_r2s = []
            fold_maes = []
            fold_mses = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_features)):
                try:
                    X_train, X_val = X_features[train_idx], X_features[val_idx]
                    y_train, y_val = y_values[train_idx], y_values[val_idx]

                    # 모델 훈련
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                    # 성능 계산
                    r2 = r2_score(y_val, y_pred)
                    mae = mean_absolute_error(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)

                    fold_r2s.append(r2)
                    fold_maes.append(mae)
                    fold_mses.append(mse)

                    print(f"      Fold {fold+1}: R²={r2:.4f}, MAE={mae:.6f}, MSE={mse:.6f}")

                except Exception as e:
                    print(f"      Fold {fold+1} 실패: {e}")
                    fold_r2s.append(-1.0)
                    fold_maes.append(1.0)
                    fold_mses.append(1.0)

            # 평균 성능
            avg_r2 = np.mean(fold_r2s)
            avg_mae = np.mean(fold_maes)
            avg_mse = np.mean(fold_mses)

            results[model_name] = {
                'r2': avg_r2,
                'mae': avg_mae,
                'mse': avg_mse,
                'rmse': np.sqrt(avg_mse),
                'fold_r2s': fold_r2s,
                'fold_maes': fold_maes,
                'target_achieved': {
                    'r2_target': avg_r2 > self.target_r2_threshold,
                    'mae_target': avg_mae < self.target_mae_threshold
                }
            }

            # 성능 평가
            status = "🎯 목표 달성!" if (avg_r2 > self.target_r2_threshold and avg_mae < self.target_mae_threshold) else "📊 기준치"
            print(f"   ✅ {model_name}: R²={avg_r2:.4f}, MAE={avg_mae:.6f} - {status}")

        return results

    def run_advanced_regression_research(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """고도화된 회귀 연구 실행"""
        print("🚀 고도화된 시계열 회귀 연구 시작")
        print("="*80)

        try:
            # 1. 데이터 로딩 및 고급 특성 생성
            df = self.data_processor.load_and_validate_data(data_path)
            advanced_df = self.create_advanced_time_series_features(df)

            # 2. 초엄격 누출 검증
            safe_features = self.validate_ultra_strict_leakage(advanced_df, 'target_return_1d')

            if len(safe_features) < 10:
                print("❌ 사용 가능한 안전 특성이 부족합니다!")
                return None

            print(f"✅ 안전 특성 {len(safe_features)}개 확보")

            # 3. 고도화된 모델들 생성
            models = self.create_advanced_regression_models()

            # 4. 성능 평가
            X = advanced_df[safe_features]
            y = advanced_df['target_return_1d']

            results = self.evaluate_regression_performance(models, X, y, safe_features)

            # 5. 결과 분석 및 저장
            self._analyze_and_save_results(results, safe_features)

            return results

        except Exception as e:
            print(f"❌ 고도화된 회귀 연구 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_and_save_results(self, results, safe_features):
        """결과 분석 및 저장"""
        print("\n📋 고도화된 회귀 연구 결과 분석")
        print("="*60)

        # 성능 순위 (R² 기준)
        sorted_by_r2 = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)

        print(f"\n🏆 R² 성능 순위:")
        for rank, (model_name, metrics) in enumerate(sorted_by_r2[:5], 1):
            r2 = metrics['r2']
            mae = metrics['mae']
            target_status = "🎯" if metrics['target_achieved']['r2_target'] and metrics['target_achieved']['mae_target'] else "📊"

            print(f"   {rank}. {model_name}: R²={r2:.4f}, MAE={mae:.6f} {target_status}")

        # MAE 순위
        sorted_by_mae = sorted(results.items(), key=lambda x: x[1]['mae'])

        print(f"\n📉 MAE 성능 순위:")
        for rank, (model_name, metrics) in enumerate(sorted_by_mae[:5], 1):
            r2 = metrics['r2']
            mae = metrics['mae']
            target_status = "🎯" if metrics['target_achieved']['r2_target'] and metrics['target_achieved']['mae_target'] else "📊"

            print(f"   {rank}. {model_name}: MAE={mae:.6f}, R²={r2:.4f} {target_status}")

        # 목표 달성 모델들
        target_achieved_models = [(name, metrics) for name, metrics in results.items()
                                if metrics['target_achieved']['r2_target'] and metrics['target_achieved']['mae_target']]

        if target_achieved_models:
            print(f"\n🎉 목표 달성 모델들 ({len(target_achieved_models)}개):")
            for name, metrics in target_achieved_models:
                print(f"   ✅ {name}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.6f}")
        else:
            print(f"\n⚠️ 목표 달성 모델 없음 - 추가 연구 필요")

        # 결과 저장
        output_path = f"/root/workspace/data/results/advanced_time_series_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(output_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'experiment_type': 'advanced_time_series_regression',
                    'target_thresholds': {
                        'r2_threshold': self.target_r2_threshold,
                        'mae_threshold': self.target_mae_threshold
                    },
                    'safe_features_count': len(safe_features),
                    'max_allowed_correlation': self.max_allowed_correlation,
                    'results': {k: {**v, 'fold_r2s': [float(x) for x in v['fold_r2s']],
                                          'fold_maes': [float(x) for x in v['fold_maes']]}
                              for k, v in results.items()}
                }, f, indent=2)
            print(f"\n💾 결과 저장: {output_path}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 실행"""
    research = AdvancedTimeSeriesRegression()
    results = research.run_advanced_regression_research()

    if results:
        print("\n🎉 고도화된 시계열 회귀 연구 완료!")
    else:
        print("\n❌ 고도화된 시계열 회귀 연구 실패!")

    return results

if __name__ == "__main__":
    main()