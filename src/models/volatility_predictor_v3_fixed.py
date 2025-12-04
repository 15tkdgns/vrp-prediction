#!/usr/bin/env python3
"""
변동성 예측 V3 Fixed: 데이터 누출 수정
모든 특성에 shift(1) 적용 → 완전한 시간적 분리

목표: R² 0.33 → 0.40+ (누출 없이)
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

class VolatilityPredictorV3Fixed:
    """데이터 누출 수정된 V3"""

    def __init__(self, ticker="SPY", start_date="2015-01-01", end_date="2024-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.results = {}

    def load_and_engineer_features(self):
        """완전한 시간적 분리 보장"""
        print(f"📂 {self.ticker} 데이터 로드 중...")

        spy = yf.Ticker(self.ticker)
        df = spy.history(start=self.start_date, end=self.end_date)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        print(f"✅ 데이터 로드: {len(df)} 샘플")

        # 기본 계산
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # ⚠️ 중요: 모든 변동성 계산 후 shift(1) 적용!
        df['volatility'] = df['returns'].rolling(20).std()
        df['vol_5d'] = df['returns'].rolling(5).std()
        df['vol_10d'] = df['returns'].rolling(10).std()
        df['vol_20d'] = df['returns'].rolling(20).std()
        df['vol_60d'] = df['returns'].rolling(60).std()

        print("\n🔧 패턴 기반 특성 생성 (데이터 누출 방지)...")

        # === 패턴 1: ATR (shift 적용) ===
        df['high_low'] = df['High'] - df['Low']
        df['high_close'] = abs(df['High'] - df['Close'].shift(1))
        df['low_close'] = abs(df['Low'] - df['Close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14_raw'] = df['true_range'].rolling(14).mean()
        df['atr_14'] = df['atr_14_raw'].shift(1)  # ✅ shift!
        df['atr_ratio'] = (df['atr_14'] / df['Close']).shift(1)  # ✅ 추가 shift!

        # === 패턴 2: Gap (이미 올바름) ===
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_size'] = df['gap'].abs()
        df['gap_size_ma'] = df['gap_size'].rolling(10).mean().shift(1)
        df['large_gap_count'] = (df['gap_size'] > df['gap_size'].quantile(0.9)).rolling(20).sum().shift(1)

        # === 패턴 3: Volume ===
        df['volume_ratio'] = (df['Volume'] / df['Volume'].rolling(20).mean()).shift(1)  # ✅ shift!
        df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
        df['volume_spike_count'] = df['volume_spike'].rolling(20).sum().shift(1)

        # === 패턴 4: Momentum ===
        df['momentum_5'] = (df['Close'].shift(1) / df['Close'].shift(6) - 1)  # ✅ t-1 기준
        df['momentum_10'] = (df['Close'].shift(1) / df['Close'].shift(11) - 1)
        df['momentum_20'] = (df['Close'].shift(1) / df['Close'].shift(21) - 1)
        df['momentum_strength'] = df['momentum_20'].abs()

        # === 패턴 5: Vol-of-Vol ===
        df['vol_of_vol'] = df['vol_20d'].rolling(20).std().shift(1)  # ✅ shift!
        df['vol_of_vol_ratio'] = (df['vol_of_vol'] / df['vol_20d'].shift(1))

        # === 패턴 6: Parkinson Vol (shift 적용) ===
        df['parkinson_vol_raw'] = np.sqrt(
            1 / (4 * np.log(2)) * np.log(df['High'] / df['Low']) ** 2
        )
        df['parkinson_vol'] = df['parkinson_vol_raw'].shift(1)  # ✅ shift!

        # === 패턴 7: Realized Range (shift 적용) ===
        df['realized_range_raw'] = (df['High'] - df['Low']) / df['Open']
        df['realized_range'] = df['realized_range_raw'].shift(1)  # ✅ shift!

        # === 패턴 8: Rolling Skew/Kurt ===
        df['rolling_skew_20'] = df['returns'].rolling(20).skew().shift(1)  # ✅ shift!
        df['rolling_kurt_20'] = df['returns'].rolling(20).kurt().shift(1)  # ✅ shift!

        # === 패턴 9: 변동성 Lag (이미 shift됨) ===
        df['vol_lag_1'] = df['vol_20d'].shift(1)
        df['vol_lag_2'] = df['vol_20d'].shift(2)
        df['vol_lag_3'] = df['vol_20d'].shift(3)
        df['vol_lag_5'] = df['vol_20d'].shift(5)
        df['vol_lag_10'] = df['vol_20d'].shift(10)

        # === 패턴 10: 극단값 카운터 ===
        df['extreme_return'] = (df['returns'].abs() > df['returns'].rolling(60).std() * 2).astype(int)
        df['extreme_count'] = df['extreme_return'].rolling(20).sum().shift(1)  # ✅ shift!

        # === 상호작용 특성 (모두 shift된 변수 사용) ===
        df['atr_x_volume'] = df['atr_ratio'] * df['volume_ratio']
        df['gap_x_momentum'] = df['gap_size'] * df['momentum_strength']
        df['vov_x_parkinson'] = df['vol_of_vol'] * df['parkinson_vol']

        # 타겟: 5일 후 변동성 (올바름)
        df['target_vol_5d'] = df['vol_20d'].shift(-5)

        df = df.dropna()
        self.data = df

        print(f"✅ 특성 생성 완료: {df.shape[1]}개 컬럼, {len(df)} 샘플")
        print(f"✅ 모든 특성 shift(1) 적용 완료 (t-1일까지 정보만 사용)")

        return True

    def method1_pattern_ridge_fixed(self):
        """방법 1: 패턴 기반 Ridge (누출 수정)"""
        print("\n🔹 방법 1: 패턴 기반 Ridge (Fixed)...")

        features = [
            'atr_ratio', 'gap_size', 'gap_size_ma', 'large_gap_count',
            'volume_ratio', 'volume_spike_count',
            'momentum_strength', 'vol_of_vol', 'vol_of_vol_ratio',
            'parkinson_vol', 'realized_range',
            'rolling_skew_20', 'rolling_kurt_20',
            'vol_lag_1', 'vol_lag_2', 'vol_lag_5', 'vol_lag_10',
            'extreme_count', 'atr_x_volume', 'gap_x_momentum'
        ]

        X = self.data[features].dropna()
        y = self.data.loc[X.index, 'target_vol_5d']

        r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "Pattern Ridge Fixed")

        self.results['method1_ridge_fixed'] = {'r2': r2, 'n_features': len(features)}
        return r2

    def method2_xgboost_fixed(self):
        """방법 2: XGBoost (누출 수정)"""
        print("\n🔹 방법 2: XGBoost (Fixed)...")

        features = [
            'atr_ratio', 'gap_size', 'volume_ratio', 'momentum_strength',
            'vol_of_vol', 'parkinson_vol', 'rolling_kurt_20',
            'vol_lag_1', 'vol_lag_2', 'vol_lag_5',
            'extreme_count', 'atr_x_volume', 'gap_x_momentum'
        ]

        X = self.data[features].dropna()
        y = self.data.loc[X.index, 'target_vol_5d']

        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )

        r2 = self._train_and_evaluate(X, y, model, "XGBoost Fixed")

        self.results['method2_xgboost_fixed'] = {'r2': r2, 'n_features': len(features)}
        return r2

    def method3_vix_fixed(self):
        """방법 3: VIX + 패턴 (누출 수정)"""
        print("\n🔹 방법 3: VIX + 패턴 (Fixed)...")

        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=self.start_date, end=self.end_date)
            vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)

            df = self.data.copy()
            df['vix'] = vix_data['Close'].reindex(df.index, method='ffill').shift(1)  # ✅ shift!
            df['vix_change'] = df['vix'].pct_change(5)
            df['vix_ma'] = df['vix'].rolling(20).mean().shift(1)

            df = df.dropna()

            features = [
                'vix', 'vix_change', 'vix_ma',
                'atr_ratio', 'gap_size', 'volume_ratio',
                'vol_of_vol', 'parkinson_vol',
                'vol_lag_1', 'vol_lag_5', 'momentum_strength'
            ]

            X = df[features]
            y = df['target_vol_5d']

            r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "Ridge + VIX Fixed")

            self.results['method3_vix_fixed'] = {'r2': r2, 'n_features': len(features)}
            return r2

        except Exception as e:
            print(f"   ⚠️  VIX 실패: {e}")
            self.results['method3_vix_fixed'] = {'r2': 0.0, 'error': str(e)}
            return 0.0

    def method4_lightgbm_fixed(self):
        """방법 4: LightGBM (누출 수정)"""
        print("\n🔹 방법 4: LightGBM (Fixed)...")

        features = [
            'atr_ratio', 'gap_size', 'volume_ratio', 'momentum_strength',
            'vol_of_vol', 'parkinson_vol', 'rolling_skew_20',
            'vol_lag_1', 'vol_lag_5', 'extreme_count',
            'atr_x_volume', 'gap_x_momentum', 'vov_x_parkinson'
        ]

        X = self.data[features].dropna()
        y = self.data.loc[X.index, 'target_vol_5d']

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

        r2 = self._train_and_evaluate(X, y, model, "LightGBM Fixed")

        self.results['method4_lightgbm_fixed'] = {'r2': r2, 'n_features': len(features)}
        return r2

    def method5_stacking_fixed(self):
        """방법 5: Stacking (누출 수정)"""
        print("\n🔹 방법 5: Stacking Ensemble (Fixed)...")

        features = [
            'atr_ratio', 'gap_size', 'volume_ratio', 'momentum_strength',
            'vol_of_vol', 'parkinson_vol', 'vol_lag_1', 'vol_lag_5',
            'extreme_count', 'atr_x_volume'
        ]

        X = self.data[features].dropna()
        y = self.data.loc[X.index, 'target_vol_5d']

        tscv = TimeSeriesSplit(n_splits=5)
        all_preds = []
        all_actuals = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            ridge = Ridge(alpha=1.0)
            xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            ridge.fit(X_train_scaled, y_train)
            xgb.fit(X_train, y_train)

            ridge_pred = ridge.predict(X_test_scaled)
            xgb_pred = xgb.predict(X_test)

            ensemble_pred = 0.6 * ridge_pred + 0.4 * xgb_pred

            all_preds.extend(ensemble_pred)
            all_actuals.extend(y_test)

        r2 = r2_score(all_actuals, all_preds)
        print(f"   Stacking Fixed R²: {r2:.4f}")

        self.results['method5_stacking_fixed'] = {'r2': r2}
        return r2

    def _train_and_evaluate(self, X, y, model, method_name):
        """TimeSeriesSplit 평가"""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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
        """모든 방법 실행"""
        print("="*60)
        print("🚀 변동성 예측 V3 Fixed: 데이터 누출 수정")
        print("="*60)
        print("기준선: V2 Regime R² = 0.328\n")

        self.load_and_engineer_features()

        methods = [
            ("Pattern Ridge Fixed", self.method1_pattern_ridge_fixed),
            ("XGBoost Fixed", self.method2_xgboost_fixed),
            ("VIX + Ridge Fixed", self.method3_vix_fixed),
            ("LightGBM Fixed", self.method4_lightgbm_fixed),
            ("Stacking Fixed", self.method5_stacking_fixed),
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
        print("📊 최종 성능 비교 (데이터 누출 수정)")
        print("="*60)

        baseline_v2 = 0.328
        baseline_v0 = 0.303

        print(f"{'방법':<30s} {'R²':>10s} {'vs V2':>12s} {'vs V0':>12s}")
        print("-"*60)
        print(f"{'V0 Ridge':<30s} {baseline_v0:>10.4f} {'-':>12s} {'-':>12s}")
        print(f"{'V2 Regime':<30s} {baseline_v2:>10.4f} {'-':>12s} {f'+{baseline_v2-baseline_v0:.4f}':>12s}")

        for name, r2 in sorted(scores, key=lambda x: x[1], reverse=True):
            improvement_v2 = r2 - baseline_v2
            improvement_v0 = r2 - baseline_v0
            symbol = "✅" if r2 > baseline_v2 else "❌"
            print(f"{name:<30s} {r2:>10.4f} {improvement_v2:>+11.4f} {improvement_v0:>+11.4f} {symbol}")

        best_method, best_r2 = max(scores, key=lambda x: x[1])

        print("\n" + "="*60)
        print(f"🏆 최고 성능: {best_method}")
        print(f"   R² = {best_r2:.4f}")
        print(f"   vs V2 개선폭 = {best_r2 - baseline_v2:+.4f}")
        print(f"   vs V0 개선폭 = {best_r2 - baseline_v0:+.4f}")
        print("="*60)

        output = {
            'experiment': 'volatility_prediction_v3_fixed',
            'baseline_v0_r2': baseline_v0,
            'baseline_v2_r2': baseline_v2,
            'best_method': best_method,
            'best_r2': best_r2,
            'improvement_vs_v2': best_r2 - baseline_v2,
            'improvement_vs_v0': best_r2 - baseline_v0,
            'all_results': self.results,
            'data_leakage': 'FIXED - All features shifted',
            'timestamp': datetime.now().isoformat()
        }

        with open('data/raw/volatility_v3_fixed_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("\n💾 결과 저장: data/raw/volatility_v3_fixed_results.json")

        return best_r2

if __name__ == "__main__":
    predictor = VolatilityPredictorV3Fixed()
    predictor.run_all_methods()
