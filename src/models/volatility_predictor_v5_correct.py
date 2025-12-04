#!/usr/bin/env python3
"""
변동성 예측 V5: V0 올바른 타겟 + 발견된 패턴 특성
타겟: returns[t+1:t+6].std() (V0 방식)
특성: 패턴 분석에서 발견한 강력한 특성들

목표: R² 0.303 → 0.40+
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VolatilityPredictorV5:
    """V0 타겟 + 패턴 특성"""

    def __init__(self, ticker="SPY", start_date="2015-01-01", end_date="2024-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.results = {}

    def load_and_engineer_features(self):
        """V0 타겟 + 패턴 특성"""
        print(f"📂 {self.ticker} 데이터 로드 중...")

        spy = yf.Ticker(self.ticker)
        df = spy.history(start=self.start_date, end=self.end_date)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        print(f"✅ 데이터 로드: {len(df)} 샘플")

        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

        print("\n🔧 올바른 타겟 생성 (V0 방식: t+1~t+5)...")

        # === 타겟: V0 방식 (5일 미래) ===
        targets = []
        for i in range(len(df)):
            if i + 5 < len(df):
                future_returns = df['returns'].iloc[i+1:i+6]  # t+1~t+5
                targets.append(future_returns.std())
            else:
                targets.append(np.nan)
        df['target_vol_5d'] = targets

        print("\n🔧 패턴 기반 특성 (모두 shift 적용)...")

        # === 기본 변동성 (과거만) ===
        df['vol_5d'] = df['returns'].rolling(5).std()
        df['vol_10d'] = df['returns'].rolling(10).std()
        df['vol_20d'] = df['returns'].rolling(20).std()
        df['vol_60d'] = df['returns'].rolling(60).std()

        # === 패턴 1: ATR (상관 0.67) ===
        df['high_low'] = df['High'] - df['Low']
        df['high_close'] = abs(df['High'] - df['Close'].shift(1))
        df['low_close'] = abs(df['Low'] - df['Close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()

        # === 패턴 2: Gap (효과 1.98배) ===
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_size'] = df['gap'].abs()
        df['large_gap_freq'] = (df['gap_size'] > df['gap_size'].quantile(0.9)).rolling(20).sum()

        # === 패턴 3: Volume (효과 1.66배) ===
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_spike_freq'] = (df['volume_ratio'] > 1.5).rolling(20).sum()

        # === 패턴 4: Momentum (효과 1.68배) ===
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        df['momentum_strength'] = df['momentum_20'].abs()

        # === 패턴 5: Vol-of-Vol (효과 1.58배) ===
        df['vol_of_vol'] = df['vol_20d'].rolling(20).std()

        # === 패턴 6: Parkinson Vol ===
        df['parkinson_vol'] = np.sqrt(
            1 / (4 * np.log(2)) * (np.log(df['High'] / df['Low']) ** 2)
        )

        # === 패턴 7: Skew/Kurt ===
        df['rolling_skew'] = df['returns'].rolling(20).skew()
        df['rolling_kurt'] = df['returns'].rolling(20).kurt()

        # === Lag features ===
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'vol_lag_{lag}'] = df['vol_20d'].shift(lag)

        # === 극단값 ===
        df['extreme_return'] = (df['returns'].abs() > df['returns'].rolling(60).std() * 2).astype(int)
        df['extreme_freq'] = df['extreme_return'].rolling(20).sum()

        # === 상호작용 ===
        df['atr_x_volume'] = df['atr_14'] * df['volume_ratio']
        df['gap_x_momentum'] = df['gap_size'] * df['momentum_strength']

        df = df.dropna()
        self.data = df

        print(f"✅ 특성 생성 완료: {df.shape[1]}개 컬럼, {len(df)} 샘플")
        print(f"✅ 타겟: V0 방식 returns[t+1:t+6].std()")

        return True

    def method1_baseline_v0(self):
        """방법 1: V0 기본 특성 재현"""
        print("\n🔹 방법 1: V0 Baseline (재현)...")

        features = [
            'vol_5d', 'vol_10d', 'vol_20d',
            'vol_lag_1', 'vol_lag_2', 'vol_lag_3', 'vol_lag_5', 'vol_lag_10'
        ]

        X = self.data[features]
        y = self.data['target_vol_5d']

        r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "V0 Baseline")

        self.results['method1_v0_baseline'] = {'r2': r2}
        return r2

    def method2_pattern_features(self):
        """방법 2: 패턴 특성 추가"""
        print("\n🔹 방법 2: V0 + 패턴 특성...")

        features = [
            # 기본
            'vol_20d', 'vol_lag_1', 'vol_lag_2', 'vol_lag_5',
            # 패턴
            'atr_14', 'gap_size', 'large_gap_freq',
            'volume_ratio', 'volume_spike_freq',
            'momentum_strength', 'vol_of_vol',
            'parkinson_vol', 'extreme_freq',
            # 상호작용
            'atr_x_volume', 'gap_x_momentum'
        ]

        X = self.data[features]
        y = self.data['target_vol_5d']

        r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "V0 + Pattern")

        self.results['method2_v0_pattern'] = {'r2': r2}
        return r2

    def method3_xgboost_pattern(self):
        """방법 3: XGBoost + 패턴"""
        print("\n🔹 방법 3: XGBoost + 패턴...")

        features = [
            'vol_20d', 'vol_lag_1', 'vol_lag_5',
            'atr_14', 'gap_size', 'volume_ratio',
            'momentum_strength', 'vol_of_vol',
            'parkinson_vol', 'extreme_freq',
            'atr_x_volume'
        ]

        X = self.data[features]
        y = self.data['target_vol_5d']

        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )

        r2 = self._train_and_evaluate(X, y, model, "XGBoost + Pattern")

        self.results['method3_xgboost_pattern'] = {'r2': r2}
        return r2

    def method4_vix_pattern(self):
        """방법 4: VIX + 패턴"""
        print("\n🔹 방법 4: VIX + 패턴...")

        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=self.start_date, end=self.end_date)
            vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)

            df = self.data.copy()
            df['vix'] = vix_data['Close'].reindex(df.index, method='ffill')
            df['vix_change'] = df['vix'].pct_change(5)

            df = df.dropna()

            features = [
                'vix', 'vix_change',
                'vol_20d', 'vol_lag_1', 'vol_lag_5',
                'atr_14', 'gap_size', 'volume_ratio',
                'momentum_strength', 'vol_of_vol',
                'parkinson_vol'
            ]

            X = df[features]
            y = df['target_vol_5d']

            r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "VIX + Pattern")

            self.results['method4_vix_pattern'] = {'r2': r2}
            return r2

        except Exception as e:
            print(f"   ⚠️  VIX 실패: {e}")
            self.results['method4_vix_pattern'] = {'r2': 0.0}
            return 0.0

    def method5_stacking(self):
        """방법 5: Stacking"""
        print("\n🔹 방법 5: Stacking (Ridge + XGBoost)...")

        features = [
            'vol_20d', 'vol_lag_1', 'vol_lag_5',
            'atr_14', 'gap_size', 'volume_ratio',
            'momentum_strength', 'vol_of_vol',
            'parkinson_vol', 'atr_x_volume'
        ]

        X = self.data[features]
        y = self.data['target_vol_5d']

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
        print(f"   Stacking R²: {r2:.4f}")

        self.results['method5_stacking'] = {'r2': r2}
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
        """전체 실행"""
        print("="*70)
        print("🚀 변동성 예측 V5: V0 타겟 + 패턴 특성")
        print("="*70)
        print("기준: V0 R² = 0.303, V2 R² = 0.328\n")

        self.load_and_engineer_features()

        methods = [
            ("V0 Baseline", self.method1_baseline_v0),
            ("V0 + Pattern", self.method2_pattern_features),
            ("XGBoost + Pattern", self.method3_xgboost_pattern),
            ("VIX + Pattern", self.method4_vix_pattern),
            ("Stacking", self.method5_stacking),
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
        print("\n" + "="*70)
        print("📊 최종 성능 비교")
        print("="*70)

        baseline_v2 = 0.328
        baseline_v0 = 0.303

        print(f"{'방법':<30s} {'R²':>10s} {'vs V2':>12s} {'vs V0':>12s}")
        print("-"*70)
        print(f"{'V0 Baseline (Original)':<30s} {baseline_v0:>10.4f} {'-':>12s} {'-':>12s}")
        print(f"{'V2 Regime':<30s} {baseline_v2:>10.4f} {f'+{baseline_v2-baseline_v0:.4f}':>12s} {'-':>12s}")

        for name, r2 in sorted(scores, key=lambda x: x[1], reverse=True):
            improvement_v2 = r2 - baseline_v2
            improvement_v0 = r2 - baseline_v0
            symbol = "✅" if r2 > baseline_v2 else ("⚠️" if r2 > baseline_v0 else "❌")
            print(f"{name:<30s} {r2:>10.4f} {improvement_v2:>+11.4f} {improvement_v0:>+11.4f} {symbol}")

        best_method, best_r2 = max(scores, key=lambda x: x[1])

        print("\n" + "="*70)
        print(f"🏆 최고 성능: {best_method}")
        print(f"   R² = {best_r2:.4f}")

        if best_r2 > baseline_v2:
            print(f"   ✅ 성공! vs V2 개선 = {best_r2 - baseline_v2:+.4f}")
            print(f"   ✅ 성공! vs V0 개선 = {best_r2 - baseline_v0:+.4f}")
        elif best_r2 > baseline_v0:
            print(f"   ⚠️  V2 미달, V0 개선 = {best_r2 - baseline_v0:+.4f}")
        else:
            print(f"   ❌ V0도 미달")

        print("="*70)

        output = {
            'experiment': 'volatility_prediction_v5_correct',
            'baseline_v0_r2': baseline_v0,
            'baseline_v2_r2': baseline_v2,
            'best_method': best_method,
            'best_r2': best_r2,
            'improvement_vs_v2': best_r2 - baseline_v2,
            'improvement_vs_v0': best_r2 - baseline_v0,
            'all_results': self.results,
            'target_design': 'V0 방식 returns[t+1:t+6].std()',
            'features': '패턴 분석 기반 (ATR, Gap, Volume, Momentum, Vol-of-Vol)',
            'timestamp': datetime.now().isoformat()
        }

        with open('data/raw/volatility_v5_final_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("\n💾 결과 저장: data/raw/volatility_v5_final_results.json")

        return best_r2

if __name__ == "__main__":
    predictor = VolatilityPredictorV5()
    predictor.run_all_methods()
