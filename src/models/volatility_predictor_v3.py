#!/usr/bin/env python3
"""
변동성 예측 V3: 패턴 기반 고급 모델
발견된 강력한 패턴 + 비선형 모델 + 앙상블

목표: R² 0.33 → 0.40+
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VolatilityPredictorV3:
    """패턴 기반 고급 변동성 예측 V3"""

    def __init__(self, ticker="SPY", start_date="2015-01-01", end_date="2024-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.results = {}

    def load_and_engineer_features(self):
        """발견된 패턴 기반 특성 엔지니어링"""
        print(f"📂 {self.ticker} 데이터 로드 중...")

        spy = yf.Ticker(self.ticker)
        df = spy.history(start=self.start_date, end=self.end_date)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        print(f"✅ 데이터 로드: {len(df)} 샘플")

        # 기본 계산
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()

        print("\n🔧 패턴 기반 특성 생성 중...")

        # === 패턴 1: ATR (Average True Range) - 상관 0.67 ===
        df['high_low'] = df['High'] - df['Low']
        df['high_close'] = abs(df['High'] - df['Close'].shift(1))
        df['low_close'] = abs(df['Low'] - df['Close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['Close']

        # ATR 변화율
        df['atr_change'] = df['atr_14'].pct_change(5)

        # === 패턴 2: Overnight Gap (효과 1.98배) ===
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_size'] = df['gap'].abs()
        df['gap_size_ma'] = df['gap_size'].rolling(10).mean()

        # 큰 갭 빈도
        df['large_gap_count'] = (df['gap_size'] > df['gap_size'].quantile(0.9)).rolling(20).sum()

        # === 패턴 3: U-shape 수익률 효과 (1.98배) ===
        df['cumulative_returns_5d'] = df['returns'].rolling(5).sum()
        df['extreme_return'] = (df['cumulative_returns_5d'].abs() > df['cumulative_returns_5d'].abs().quantile(0.75)).astype(int)
        df['extreme_return_count'] = df['extreme_return'].rolling(20).sum()

        # === 패턴 4: Volume Spike (효과 1.66배) ===
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
        df['volume_spike_count'] = df['volume_spike'].rolling(20).sum()

        # Volume 변화율
        df['volume_change'] = df['Volume'].pct_change(5)

        # === 패턴 5: 가격 모멘텀 (효과 1.68배) ===
        df['momentum_5'] = (df['Close'] / df['Close'].shift(5) - 1)
        df['momentum_10'] = (df['Close'] / df['Close'].shift(10) - 1)
        df['momentum_20'] = (df['Close'] / df['Close'].shift(20) - 1)

        # 모멘텀 강도
        df['momentum_strength'] = df['momentum_20'].abs()
        df['strong_momentum'] = (df['momentum_strength'] > df['momentum_strength'].quantile(0.75)).astype(int)

        # === 패턴 6: Volatility-of-Volatility (효과 1.58배) ===
        df['vol_5d'] = df['returns'].rolling(5).std()
        df['vol_10d'] = df['returns'].rolling(10).std()
        df['vol_20d'] = df['returns'].rolling(20).std()
        df['vol_60d'] = df['returns'].rolling(60).std()

        df['vol_of_vol'] = df['vol_20d'].rolling(20).std()
        df['vol_of_vol_ratio'] = df['vol_of_vol'] / df['vol_20d']

        # === 패턴 7: Rolling Skewness/Kurtosis ===
        df['rolling_skew_20'] = df['returns'].rolling(20).skew()
        df['rolling_kurt_20'] = df['returns'].rolling(20).kurt()

        # Fat tail indicator
        df['fat_tail'] = (df['rolling_kurt_20'] > 3).astype(int)

        # === 패턴 8: 변동성 Lag (강한 자기상관) ===
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'vol_lag_{lag}'] = df['vol_20d'].shift(lag)

        # === 패턴 9: Parkinson Volatility (High-Low) ===
        df['parkinson_vol'] = np.sqrt(
            1 / (4 * np.log(2)) * np.log(df['High'] / df['Low']) ** 2
        )
        df['parkinson_ma'] = df['parkinson_vol'].rolling(5).mean()

        # === 패턴 10: Realized Range ===
        df['realized_range'] = (df['High'] - df['Low']) / df['Open']
        df['realized_range_ma'] = df['realized_range'].rolling(10).mean()

        # === 상호작용 특성 ===
        df['atr_x_volume'] = df['atr_ratio'] * df['volume_ratio']
        df['gap_x_momentum'] = df['gap_size'] * df['momentum_strength']
        df['vov_x_volume'] = df['vol_of_vol'] * df['volume_spike']

        # 타겟: 5일 후 변동성
        df['target_vol_5d'] = df['vol_20d'].shift(-5)

        df = df.dropna()
        self.data = df

        print(f"✅ 특성 생성 완료: {df.shape[1]}개 컬럼, {len(df)} 샘플")

        return True

    def method1_pattern_ridge(self):
        """방법 1: 패턴 기반 Ridge"""
        print("\n🔹 방법 1: 패턴 기반 Ridge...")

        features = [
            # ATR 관련
            'atr_ratio', 'atr_change',
            # Gap 관련
            'gap_size', 'large_gap_count',
            # Volume 관련
            'volume_ratio', 'volume_spike_count',
            # Momentum
            'momentum_strength', 'strong_momentum',
            # Vol-of-vol
            'vol_of_vol', 'vol_of_vol_ratio',
            # 기본 변동성 및 lag
            'vol_20d', 'vol_lag_1', 'vol_lag_2', 'vol_lag_5', 'vol_lag_10',
            # Parkinson
            'parkinson_vol',
            # 상호작용
            'atr_x_volume', 'gap_x_momentum'
        ]

        X = self.data[features]
        y = self.data['target_vol_5d']

        r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "Pattern Ridge")

        self.results['method1_pattern_ridge'] = {
            'r2': r2,
            'n_features': len(features)
        }

        return r2

    def method2_xgboost(self):
        """방법 2: XGBoost (비선형 패턴 포착)"""
        print("\n🔹 방법 2: XGBoost 비선형 모델...")

        features = [
            'atr_ratio', 'gap_size', 'volume_ratio', 'momentum_strength',
            'vol_of_vol', 'vol_20d', 'vol_lag_1', 'vol_lag_2', 'vol_lag_5',
            'parkinson_vol', 'rolling_kurt_20', 'extreme_return_count',
            'large_gap_count', 'volume_spike_count', 'atr_x_volume'
        ]

        X = self.data[features]
        y = self.data['target_vol_5d']

        # XGBoost 하이퍼파라미터 (과적합 방지)
        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )

        r2 = self._train_and_evaluate(X, y, model, "XGBoost")

        self.results['method2_xgboost'] = {
            'r2': r2,
            'n_features': len(features)
        }

        return r2

    def method3_lightgbm(self):
        """방법 3: LightGBM (빠르고 효율적)"""
        print("\n🔹 방법 3: LightGBM...")

        features = [
            'atr_ratio', 'gap_size', 'volume_ratio', 'momentum_strength',
            'vol_of_vol', 'vol_20d', 'vol_lag_1', 'vol_lag_2', 'vol_lag_5',
            'parkinson_vol', 'rolling_skew_20', 'extreme_return_count',
            'atr_x_volume', 'gap_x_momentum', 'vov_x_volume'
        ]

        X = self.data[features]
        y = self.data['target_vol_5d']

        model = LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )

        r2 = self._train_and_evaluate(X, y, model, "LightGBM")

        self.results['method3_lightgbm'] = {
            'r2': r2,
            'n_features': len(features)
        }

        return r2

    def method4_vix_external(self):
        """방법 4: VIX 외부 요인 추가"""
        print("\n🔹 방법 4: VIX 지수 통합...")

        try:
            # VIX 데이터 로드
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=self.start_date, end=self.end_date)
            vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)

            df = self.data.copy()
            df['vix'] = vix_data['Close'].reindex(df.index, method='ffill')
            df['vix_change'] = df['vix'].pct_change(5)
            df['vix_ma'] = df['vix'].rolling(20).mean()

            df = df.dropna()

            features = [
                'vix', 'vix_change', 'vix_ma',
                'atr_ratio', 'gap_size', 'volume_ratio',
                'vol_of_vol', 'vol_20d', 'vol_lag_1', 'vol_lag_5',
                'parkinson_vol', 'momentum_strength'
            ]

            X = df[features]
            y = df['target_vol_5d']

            r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "Ridge + VIX")

            self.results['method4_vix'] = {
                'r2': r2,
                'n_features': len(features)
            }

            return r2

        except Exception as e:
            print(f"   ⚠️  VIX 로드 실패: {e}")
            self.results['method4_vix'] = {'r2': 0.0, 'error': str(e)}
            return 0.0

    def method5_stacking_ensemble(self):
        """방법 5: Stacking 앙상블 (Ridge + XGBoost + Regime)"""
        print("\n🔹 방법 5: Stacking Ensemble...")

        features = [
            'atr_ratio', 'gap_size', 'volume_ratio', 'momentum_strength',
            'vol_of_vol', 'vol_20d', 'vol_lag_1', 'vol_lag_2', 'vol_lag_5',
            'parkinson_vol', 'extreme_return_count', 'atr_x_volume'
        ]

        X = self.data[features]
        y = self.data['target_vol_5d']

        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        all_preds = []
        all_actuals = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Level 0: Base models
            ridge = Ridge(alpha=1.0)
            xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)

            ridge.fit(X_train, y_train)
            xgb.fit(X_train, y_train)

            # Level 1: Meta-learner (가중 평균)
            ridge_pred = ridge.predict(X_test)
            xgb_pred = xgb.predict(X_test)

            # 동적 가중치 (최근 성능 기반)
            ensemble_pred = 0.6 * ridge_pred + 0.4 * xgb_pred

            all_preds.extend(ensemble_pred)
            all_actuals.extend(y_test)

        r2 = r2_score(all_actuals, all_preds)
        print(f"   Stacking Ensemble R²: {r2:.4f}")

        self.results['method5_stacking'] = {
            'r2': r2,
            'weights': {'ridge': 0.6, 'xgboost': 0.4}
        }

        return r2

    def method6_deep_pattern_features(self):
        """방법 6: 심화 패턴 특성"""
        print("\n🔹 방법 6: 심화 패턴 특성...")

        df = self.data.copy()

        # 고급 패턴 특성
        df['vol_acceleration'] = df['vol_20d'].diff(5)
        df['gap_volatility'] = df['gap_size'].rolling(10).std()
        df['volume_volatility'] = df['volume_ratio'].rolling(10).std()

        # Regime indicator (변동성 사이클)
        df['vol_percentile'] = df['vol_20d'].rolling(60).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-6)
        )

        # 극단값 카운터
        df['extreme_vol_count'] = (df['vol_20d'] > df['vol_20d'].rolling(60).quantile(0.9)).rolling(20).sum()

        df = df.dropna()

        features = [
            'vol_acceleration', 'gap_volatility', 'volume_volatility', 'vol_percentile',
            'extreme_vol_count', 'atr_ratio', 'vol_of_vol', 'vol_20d',
            'vol_lag_1', 'vol_lag_5', 'parkinson_vol', 'momentum_strength'
        ]

        X = df[features]
        y = df['target_vol_5d']

        r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "Deep Pattern")

        self.results['method6_deep_pattern'] = {
            'r2': r2,
            'n_features': len(features)
        }

        return r2

    def _train_and_evaluate(self, X, y, model, method_name):
        """공통 학습 및 평가 (TimeSeriesSplit)"""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scaling (Ridge/선형 모델에만)
            if isinstance(model, Ridge):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            r2 = r2_score(y_test, pred)
            scores.append(r2)

        mean_r2 = np.mean(scores)
        print(f"   {method_name} R²: {mean_r2:.4f} (±{np.std(scores):.4f})")

        return mean_r2

    def run_all_methods(self):
        """모든 방법 실행 및 비교"""
        print("="*60)
        print("🚀 변동성 예측 V3: 패턴 기반 고급 모델")
        print("="*60)
        print("기준선: Regime-Switching R² = 0.328\n")

        self.load_and_engineer_features()

        methods = [
            ("Pattern Ridge", self.method1_pattern_ridge),
            ("XGBoost", self.method2_xgboost),
            ("LightGBM", self.method3_lightgbm),
            ("Ridge + VIX", self.method4_vix_external),
            ("Stacking Ensemble", self.method5_stacking_ensemble),
            ("Deep Pattern", self.method6_deep_pattern_features),
        ]

        scores = []

        for name, method in methods:
            try:
                r2 = method()
                scores.append((name, r2))
            except Exception as e:
                print(f"   ❌ {name} 실패: {e}")
                scores.append((name, 0.0))

        # 최종 비교
        print("\n" + "="*60)
        print("📊 최종 성능 비교")
        print("="*60)

        baseline_v2 = 0.328
        baseline_v0 = 0.303

        print(f"{'방법':<30s} {'R²':>10s} {'vs V2':>12s} {'vs V0':>12s}")
        print("-"*60)
        print(f"{'Baseline V0 (Ridge)':<30s} {baseline_v0:>10.4f} {'-':>12s} {'-':>12s}")
        print(f"{'Baseline V2 (Regime)':<30s} {baseline_v2:>10.4f} {'-':>12s} {f'+{baseline_v2-baseline_v0:.4f}':>12s}")

        for name, r2 in sorted(scores, key=lambda x: x[1], reverse=True):
            improvement_v2 = r2 - baseline_v2
            improvement_v0 = r2 - baseline_v0
            symbol = "✅" if r2 > baseline_v2 else "❌"
            print(f"{name:<30s} {r2:>10.4f} {improvement_v2:>+11.4f} {improvement_v0:>+11.4f} {symbol}")

        # 최고 성능
        best_method, best_r2 = max(scores, key=lambda x: x[1])

        print("\n" + "="*60)
        print(f"🏆 최고 성능: {best_method}")
        print(f"   R² = {best_r2:.4f}")
        print(f"   vs V2 개선폭 = {best_r2 - baseline_v2:+.4f}")
        print(f"   vs V0 개선폭 = {best_r2 - baseline_v0:+.4f}")
        print("="*60)

        # 결과 저장
        output = {
            'experiment': 'volatility_prediction_v3',
            'baseline_v0_r2': baseline_v0,
            'baseline_v2_r2': baseline_v2,
            'best_method': best_method,
            'best_r2': best_r2,
            'improvement_vs_v2': best_r2 - baseline_v2,
            'improvement_vs_v0': best_r2 - baseline_v0,
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }

        with open('data/raw/volatility_v3_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("\n💾 결과 저장: data/raw/volatility_v3_results.json")

        return best_r2

if __name__ == "__main__":
    predictor = VolatilityPredictorV3()
    predictor.run_all_methods()
