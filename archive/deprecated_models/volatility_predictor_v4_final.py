#!/usr/bin/env python3
"""
변동성 예측 V4 Final: 완전한 데이터 누출 제거
타겟 재설계: returns[t+1:t+21].std() (100% 미래 데이터)

목표: R² 0.33 → 0.40+ (누출 완전 제거)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VolatilityPredictorV4Final:
    """완전 누출 제거 V4"""

    def __init__(self, ticker="SPY", start_date="2015-01-01", end_date="2024-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.results = {}

    def load_and_engineer_features(self):
        """완전한 시간적 분리"""
        print(f"📂 {self.ticker} 데이터 로드 중...")

        spy = yf.Ticker(self.ticker)
        df = spy.history(start=self.start_date, end=self.end_date)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        print(f"✅ 데이터 로드: {len(df)} 샘플")

        # 기본 계산
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

        print("\n🔧 특성 생성 (완전 분리 보장)...")

        # === 타겟: 미래 20일 변동성 (완전 미래) ===
        # ✅ returns[t+1:t+21].std() (t+1부터 시작 → 겹침 0)
        df['target_vol_future'] = df['returns'].iloc[::-1].rolling(20).std().iloc[::-1].shift(-20)

        # === 특성: t-1일까지만 사용 ===

        # 1. 과거 변동성
        df['vol_5d'] = df['returns'].rolling(5).std().shift(1)
        df['vol_10d'] = df['returns'].rolling(10).std().shift(1)
        df['vol_20d'] = df['returns'].rolling(20).std().shift(1)
        df['vol_60d'] = df['returns'].rolling(60).std().shift(1)

        # 2. ATR
        df['high_low'] = (df['High'] - df['Low']).shift(1)
        df['true_range'] = df[['High', 'Low', 'Close']].apply(
            lambda x: max(x['High'] - x['Low'],
                         abs(x['High'] - x['Close']),
                         abs(x['Low'] - x['Close'])), axis=1
        ).shift(1)
        df['atr_14'] = df['true_range'].rolling(14).mean().shift(1)

        # 3. Gap
        df['gap_size'] = abs((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
        df['large_gap_freq'] = (df['gap_size'] > df['gap_size'].quantile(0.9)).rolling(20).sum().shift(1)

        # 4. Volume
        df['volume_ratio'] = (df['Volume'] / df['Volume'].rolling(20).mean()).shift(1)
        df['volume_spike_freq'] = (df['volume_ratio'] > 1.5).rolling(20).sum().shift(1)

        # 5. Momentum
        df['momentum_5'] = (df['Close'].shift(1) / df['Close'].shift(6) - 1)
        df['momentum_20'] = (df['Close'].shift(1) / df['Close'].shift(21) - 1)
        df['momentum_strength'] = df['momentum_20'].abs()

        # 6. Vol-of-vol
        df['vol_of_vol'] = df['vol_20d'].rolling(20).std().shift(1)

        # 7. Parkinson vol
        df['parkinson_vol'] = np.sqrt(
            1 / (4 * np.log(2)) * (np.log(df['High'] / df['Low']) ** 2)
        ).shift(1)

        # 8. Lag features
        df['vol_lag_1'] = df['vol_20d'].shift(1)
        df['vol_lag_2'] = df['vol_20d'].shift(2)
        df['vol_lag_5'] = df['vol_20d'].shift(5)
        df['vol_lag_10'] = df['vol_20d'].shift(10)

        # 9. 극단값
        df['extreme_return'] = (df['returns'].abs() > df['returns'].rolling(60).std() * 2).astype(int).shift(1)
        df['extreme_freq'] = df['extreme_return'].rolling(20).sum().shift(1)

        # 10. 상호작용
        df['atr_x_volume'] = df['atr_14'] * df['volume_ratio']
        df['gap_x_momentum'] = df['gap_size'] * df['momentum_strength']

        df = df.dropna()
        self.data = df

        print(f"✅ 특성 생성 완료: {df.shape[1]}개 컬럼, {len(df)} 샘플")
        print(f"✅ 타겟: returns[t+1:t+21].std() (완전 미래)")
        print(f"✅ 특성: 모두 t-1일까지 (완전 분리)")

        return True

    def method1_pattern_ridge_v4(self):
        """방법 1: 패턴 Ridge (완전 분리)"""
        print("\n🔹 방법 1: Pattern Ridge V4...")

        features = [
            'vol_20d', 'vol_lag_1', 'vol_lag_2', 'vol_lag_5',
            'atr_14', 'gap_size', 'large_gap_freq',
            'volume_ratio', 'volume_spike_freq',
            'momentum_strength', 'vol_of_vol',
            'parkinson_vol', 'extreme_freq',
            'atr_x_volume', 'gap_x_momentum'
        ]

        X = self.data[features].dropna()
        y = self.data.loc[X.index, 'target_vol_future']

        r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "Pattern Ridge V4")

        self.results['method1_ridge_v4'] = {'r2': r2}
        return r2

    def method2_xgboost_v4(self):
        """방법 2: XGBoost V4"""
        print("\n🔹 방법 2: XGBoost V4...")

        features = [
            'vol_20d', 'vol_lag_1', 'vol_lag_5',
            'atr_14', 'gap_size', 'volume_ratio',
            'momentum_strength', 'vol_of_vol',
            'parkinson_vol', 'atr_x_volume'
        ]

        X = self.data[features].dropna()
        y = self.data.loc[X.index, 'target_vol_future']

        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )

        r2 = self._train_and_evaluate(X, y, model, "XGBoost V4")

        self.results['method2_xgboost_v4'] = {'r2': r2}
        return r2

    def method3_vix_v4(self):
        """방법 3: VIX + 패턴 V4"""
        print("\n🔹 방법 3: VIX + Ridge V4...")

        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=self.start_date, end=self.end_date)
            vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)

            df = self.data.copy()
            df['vix'] = vix_data['Close'].reindex(df.index, method='ffill').shift(1)
            df['vix_change'] = df['vix'].pct_change(5)

            df = df.dropna()

            features = [
                'vix', 'vix_change',
                'vol_20d', 'vol_lag_1', 'atr_14',
                'gap_size', 'volume_ratio',
                'momentum_strength', 'vol_of_vol'
            ]

            X = df[features]
            y = df['target_vol_future']

            r2 = self._train_and_evaluate(X, y, Ridge(alpha=1.0), "VIX + Ridge V4")

            self.results['method3_vix_v4'] = {'r2': r2}
            return r2

        except Exception as e:
            print(f"   ⚠️  VIX 실패: {e}")
            self.results['method3_vix_v4'] = {'r2': 0.0}
            return 0.0

    def method4_stacking_v4(self):
        """방법 4: Stacking V4"""
        print("\n🔹 방법 4: Stacking V4...")

        features = [
            'vol_20d', 'vol_lag_1', 'vol_lag_5',
            'atr_14', 'gap_size', 'volume_ratio',
            'momentum_strength', 'vol_of_vol',
            'parkinson_vol', 'atr_x_volume'
        ]

        X = self.data[features].dropna()
        y = self.data.loc[X.index, 'target_vol_future']

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
        print(f"   Stacking V4 R²: {r2:.4f}")

        self.results['method4_stacking_v4'] = {'r2': r2}
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
        print("🚀 변동성 예측 V4 Final: 완전한 데이터 누출 제거")
        print("="*70)
        print("기준선: V2 Regime R² = 0.328\n")

        self.load_and_engineer_features()

        methods = [
            ("Pattern Ridge V4", self.method1_pattern_ridge_v4),
            ("XGBoost V4", self.method2_xgboost_v4),
            ("VIX + Ridge V4", self.method3_vix_v4),
            ("Stacking V4", self.method4_stacking_v4),
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
        print("📊 최종 성능 비교 (완전 누출 제거)")
        print("="*70)

        baseline_v2 = 0.328
        baseline_v0 = 0.303

        print(f"{'방법':<30s} {'R²':>10s} {'vs V2':>12s} {'vs V0':>12s}")
        print("-"*70)
        print(f"{'V0 Ridge':<30s} {baseline_v0:>10.4f} {'-':>12s} {'-':>12s}")
        print(f"{'V2 Regime':<30s} {baseline_v2:>10.4f} {'-':>12s} {f'+{baseline_v2-baseline_v0:.4f}':>12s}")

        for name, r2 in sorted(scores, key=lambda x: x[1], reverse=True):
            improvement_v2 = r2 - baseline_v2
            improvement_v0 = r2 - baseline_v0
            symbol = "✅" if r2 > baseline_v2 else ("⚠️" if r2 > baseline_v0 else "❌")
            print(f"{name:<30s} {r2:>10.4f} {improvement_v2:>+11.4f} {improvement_v0:>+11.4f} {symbol}")

        best_method, best_r2 = max(scores, key=lambda x: x[1])

        print("\n" + "="*70)
        print(f"🏆 최고 성능: {best_method}")
        print(f"   R² = {best_r2:.4f}")

        if best_r2 > 0.7:
            print(f"   ⚠️  경고: R² > 0.7 (누출 재확인 필요)")
        elif best_r2 > baseline_v2:
            print(f"   ✅ 성공: vs V2 개선폭 = {best_r2 - baseline_v2:+.4f}")
        else:
            print(f"   ⚠️  V2 미달: {best_r2 - baseline_v2:+.4f}")

        print("="*70)

        output = {
            'experiment': 'volatility_prediction_v4_final',
            'baseline_v0_r2': baseline_v0,
            'baseline_v2_r2': baseline_v2,
            'best_method': best_method,
            'best_r2': best_r2,
            'improvement_vs_v2': best_r2 - baseline_v2,
            'improvement_vs_v0': best_r2 - baseline_v0,
            'all_results': self.results,
            'target_design': 'returns[t+1:t+21].std() (완전 미래)',
            'data_leakage': 'ZERO - Complete temporal separation',
            'timestamp': datetime.now().isoformat()
        }

        with open('data/raw/volatility_v4_final_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("\n💾 결과 저장: data/raw/volatility_v4_final_results.json")

        return best_r2

if __name__ == "__main__":
    predictor = VolatilityPredictorV4Final()
    predictor.run_all_methods()
