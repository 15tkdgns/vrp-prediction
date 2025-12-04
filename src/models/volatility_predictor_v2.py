#!/usr/bin/env python3
"""
변동성 예측 V2: 혁신적 접근 방법들
- GARCH 계열 모델
- Quantile Regression (극단값 예측)
- Regime-Switching (고변동/저변동 구분)
- Multi-horizon 예측
- Rolling Window Optimization

목표: R² 0.30 → 0.40+
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class VolatilityPredictorV2:
    """혁신적 변동성 예측 모델 V2"""

    def __init__(self, ticker="SPY", start_date="2015-01-01", end_date="2024-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.results = {}

    def load_and_prepare_data(self):
        """데이터 로드"""
        print(f"📂 {self.ticker} 데이터 로드 중...")

        spy = yf.Ticker(self.ticker)
        df = spy.history(start=self.start_date, end=self.end_date)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # 기본 계산
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()

        # 다양한 시간대 변동성
        for window in [5, 10, 20, 60]:
            df[f'vol_{window}d'] = df['returns'].rolling(window).std()

        # Lag 특성
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'vol_lag_{lag}'] = df['volatility'].shift(lag)

        # 타겟들 (multi-horizon)
        df['target_vol_1d'] = df['volatility'].shift(-1)
        df['target_vol_5d'] = df['returns'].rolling(5).std().shift(-5)
        df['target_vol_20d'] = df['returns'].rolling(20).std().shift(-20)

        df = df.dropna()
        self.data = df

        print(f"✅ 데이터 준비 완료: {len(df)} 샘플")
        return True

    def method1_garch_features(self):
        """방법 1: GARCH-inspired 특성 (arch 라이브러리 없이)"""
        print("\n🔹 방법 1: GARCH-inspired 특성 생성...")

        df = self.data.copy()

        # GARCH 근사: 과거 제곱 수익률과 과거 변동성의 가중 조합
        df['squared_returns'] = df['returns'] ** 2
        df['garch_proxy'] = (
            0.1 * df['squared_returns'].shift(1) +
            0.8 * df['volatility'].shift(1) ** 2
        )
        df['garch_vol'] = np.sqrt(df['garch_proxy'])

        # 변동성 persistence (자기상관)
        df['vol_persistence'] = df['volatility'].rolling(5).apply(
            lambda x: x.autocorr(lag=1) if len(x) >= 5 else 0
        )

        # 특성 선택
        features = ['volatility', 'garch_vol', 'vol_persistence'] + \
                   [f'vol_lag_{i}' for i in [1, 2, 3, 5, 10]]

        X = df[features].dropna()
        y = df.loc[X.index, 'target_vol_5d']

        # Ridge 모델
        r2 = self._train_and_evaluate(X, y, "GARCH-inspired")

        self.results['method1_garch'] = {
            'r2': r2,
            'features': features,
            'description': 'GARCH proxy features'
        }

        return r2

    def method2_quantile_regression(self):
        """방법 2: Quantile Regression (극단값 예측)"""
        print("\n🔹 방법 2: Quantile Regression (50%, 75%, 90% 분위)...")

        df = self.data.copy()

        features = ['volatility'] + [f'vol_lag_{i}' for i in [1, 2, 3, 5, 10]]
        X = df[features].dropna()
        y = df.loc[X.index, 'target_vol_5d']

        # 3개 분위수 모델
        quantiles = [0.5, 0.75, 0.9]
        predictions = []

        for q in quantiles:
            model = QuantileRegressor(quantile=q, alpha=0.1, solver='highs')

            tscv = TimeSeriesSplit(n_splits=3)
            preds = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                preds.extend(pred)

            predictions.append(preds)

        # 앙상블: 평균
        ensemble_pred = np.mean(predictions, axis=0)

        # 마지막 fold만 평가
        r2 = r2_score(y.iloc[-len(preds):], ensemble_pred)

        print(f"   Quantile Ensemble R²: {r2:.4f}")

        self.results['method2_quantile'] = {
            'r2': r2,
            'quantiles': quantiles,
            'description': 'Quantile regression ensemble'
        }

        return r2

    def method3_regime_switching(self):
        """방법 3: Regime-Switching (고변동/저변동 구분 학습)"""
        print("\n🔹 방법 3: Regime-Switching 모델...")

        df = self.data.copy()

        # Regime 구분: K-means로 고변동/저변동 클러스터링
        vol_values = df['volatility'].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42)
        df['regime'] = kmeans.fit_predict(vol_values)

        # 고변동 = 1, 저변동 = 0으로 정렬
        cluster_means = df.groupby('regime')['volatility'].mean()
        if cluster_means[0] > cluster_means[1]:
            df['regime'] = 1 - df['regime']

        print(f"   고변동 regime: {(df['regime']==1).sum()} 샘플")
        print(f"   저변동 regime: {(df['regime']==0).sum()} 샘플")

        # 각 regime별 모델 학습
        features = ['volatility'] + [f'vol_lag_{i}' for i in [1, 2, 3, 5, 10]]

        predictions = []
        actuals = []

        for regime in [0, 1]:
            df_regime = df[df['regime'] == regime].copy()

            if len(df_regime) < 100:
                continue

            X = df_regime[features]
            y = df_regime['target_vol_5d']

            # 간단한 train/test split (시계열 순서 유지)
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            predictions.extend(pred)
            actuals.extend(y_test)

        r2 = r2_score(actuals, predictions)
        print(f"   Regime-Switching R²: {r2:.4f}")

        self.results['method3_regime'] = {
            'r2': r2,
            'n_regimes': 2,
            'description': 'Separate models for high/low volatility regimes'
        }

        return r2

    def method4_multi_horizon(self):
        """방법 4: Multi-horizon 예측 (1일, 5일, 20일 동시)"""
        print("\n🔹 방법 4: Multi-horizon 예측...")

        df = self.data.copy()

        features = ['volatility'] + [f'vol_lag_{i}' for i in [1, 2, 3, 5, 10]]
        X = df[features].dropna()

        # 3개 타겟 동시 학습
        targets = {
            '1d': df.loc[X.index, 'target_vol_1d'],
            '5d': df.loc[X.index, 'target_vol_5d'],
            '20d': df.loc[X.index, 'target_vol_20d']
        }

        results = {}

        for horizon, y in targets.items():
            y = y.dropna()
            X_aligned = X.loc[y.index]

            split = int(len(X_aligned) * 0.8)
            X_train = X_aligned[:split]
            X_test = X_aligned[split:]
            y_train = y[:split]
            y_test = y[split:]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)

            print(f"   {horizon} horizon R²: {r2:.4f}")
            results[horizon] = r2

        # 5일 기준 평가
        r2_main = results['5d']

        self.results['method4_multihorizon'] = {
            'r2_5d': r2_main,
            'all_horizons': results,
            'description': 'Multi-horizon predictions (1d, 5d, 20d)'
        }

        return r2_main

    def method5_rolling_optimization(self):
        """방법 5: Rolling Window 적응형 학습"""
        print("\n🔹 방법 5: Rolling Window Optimization...")

        df = self.data.copy()

        features = ['volatility'] + [f'vol_lag_{i}' for i in [1, 2, 3, 5, 10]]
        X = df[features].dropna()
        y = df.loc[X.index, 'target_vol_5d']

        # Rolling window parameters
        train_window = 500  # 500일 학습
        test_window = 50    # 50일 테스트

        predictions = []
        actuals = []

        start = train_window
        while start + test_window < len(X):
            # Train window
            X_train = X.iloc[start-train_window:start]
            y_train = y.iloc[start-train_window:start]

            # Test window
            X_test = X.iloc[start:start+test_window]
            y_test = y.iloc[start:start+test_window]

            # 적응형 alpha 선택 (최근 변동성에 따라)
            recent_vol = df['volatility'].iloc[start-10:start].mean()
            alpha = 0.5 if recent_vol > df['volatility'].median() else 1.5

            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            predictions.extend(pred)
            actuals.extend(y_test)

            start += test_window

        r2 = r2_score(actuals, predictions)
        print(f"   Rolling Window R²: {r2:.4f}")
        print(f"   총 {len(predictions)} 예측 생성")

        self.results['method5_rolling'] = {
            'r2': r2,
            'train_window': train_window,
            'test_window': test_window,
            'description': 'Adaptive rolling window with dynamic alpha'
        }

        return r2

    def method6_exponential_weighting(self):
        """방법 6: 지수 가중 특성 (최근 데이터 강조)"""
        print("\n🔹 방법 6: Exponential Weighted Features...")

        df = self.data.copy()

        # 지수 가중 이동평균
        spans = [5, 10, 20, 60]
        for span in spans:
            df[f'ewm_vol_{span}'] = df['volatility'].ewm(span=span).mean()

        # 지수 가중 표준편차
        df['ewm_std'] = df['returns'].ewm(span=20).std()

        features = [f'ewm_vol_{s}' for s in spans] + ['ewm_std'] + \
                   [f'vol_lag_{i}' for i in [1, 2, 3, 5]]

        X = df[features].dropna()
        y = df.loc[X.index, 'target_vol_5d']

        r2 = self._train_and_evaluate(X, y, "Exponential Weighted")

        self.results['method6_ewm'] = {
            'r2': r2,
            'features': features,
            'description': 'Exponentially weighted moving features'
        }

        return r2

    def _train_and_evaluate(self, X, y, method_name):
        """공통 학습 및 평가"""
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)

        print(f"   {method_name} R²: {r2:.4f}")

        return r2

    def run_all_methods(self):
        """모든 방법 실행 및 비교"""
        print("="*60)
        print("🚀 변동성 예측 V2: 혁신적 접근 방법들")
        print("="*60)
        print("기준선: Ridge R² = 0.303\n")

        self.load_and_prepare_data()

        methods = [
            ("GARCH-inspired Features", self.method1_garch_features),
            ("Quantile Regression", self.method2_quantile_regression),
            ("Regime-Switching", self.method3_regime_switching),
            ("Multi-horizon", self.method4_multi_horizon),
            ("Rolling Window", self.method5_rolling_optimization),
            ("Exponential Weighting", self.method6_exponential_weighting),
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

        baseline = 0.303
        print(f"{'방법':<30s} {'R²':>10s} {'vs Baseline':>15s}")
        print("-"*60)
        print(f"{'Baseline (기존 Ridge)':<30s} {baseline:>10.4f} {'-':>15s}")

        for name, r2 in sorted(scores, key=lambda x: x[1], reverse=True):
            improvement = r2 - baseline
            symbol = "✅" if r2 > baseline else "❌"
            print(f"{name:<30s} {r2:>10.4f} {improvement:>+14.4f} {symbol}")

        # 최고 성능
        best_method, best_r2 = max(scores, key=lambda x: x[1])

        print("\n" + "="*60)
        print(f"🏆 최고 성능: {best_method}")
        print(f"   R² = {best_r2:.4f}")
        print(f"   개선폭 = {best_r2 - baseline:+.4f}")
        print("="*60)

        # 결과 저장
        import json
        from datetime import datetime

        output = {
            'experiment': 'volatility_prediction_v2',
            'baseline_r2': baseline,
            'best_method': best_method,
            'best_r2': best_r2,
            'improvement': best_r2 - baseline,
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }

        with open('data/raw/volatility_v2_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("\n💾 결과 저장: data/raw/volatility_v2_results.json")

        return best_r2

if __name__ == "__main__":
    predictor = VolatilityPredictorV2()
    predictor.run_all_methods()
