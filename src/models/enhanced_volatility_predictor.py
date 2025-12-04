#!/usr/bin/env python3
"""
향상된 변동성 예측 모델
Phase 1: 비대칭 특성 추가 (Leverage Effect, Downside Risk)

목표: R² 0.30 → 0.35
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedVolatilityPredictor:
    """향상된 변동성 예측 모델"""

    def __init__(self, ticker="SPY", start_date="2015-01-01", end_date="2024-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}

    def load_and_prepare_data(self):
        """데이터 로드 및 특성 엔지니어링"""
        print(f"📂 {self.ticker} 데이터 로드 중...")

        # SPY 데이터
        spy = yf.Ticker(self.ticker)
        df = spy.history(start=self.start_date, end=self.end_date)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        print(f"✅ 데이터 로드: {len(df)} 샘플")

        # 기본 특성
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility_5d'] = df['returns'].rolling(5).std()
        df['volatility_10d'] = df['returns'].rolling(10).std()
        df['volatility_20d'] = df['returns'].rolling(20).std()

        # **새로운 비대칭 특성 추가**
        print("\n🆕 비대칭 특성 추가 중...")

        # 1. Leverage Effect (하락 시 변동성 증폭)
        df['leverage_effect'] = (df['returns'] < 0).astype(int) * df['returns'].abs()
        df['leverage_vol'] = df['leverage_effect'].rolling(5).std()

        # 2. Downside Volatility (하방 위험)
        df['negative_returns'] = df['returns'].where(df['returns'] < 0, 0)
        df['downside_vol_5d'] = df['negative_returns'].rolling(5).std()
        df['downside_vol_20d'] = df['negative_returns'].rolling(20).std()

        # 3. Upside Volatility (상승 변동성)
        df['positive_returns'] = df['returns'].where(df['returns'] > 0, 0)
        df['upside_vol_5d'] = df['positive_returns'].rolling(5).std()

        # 4. Asymmetry Ratio (비대칭 비율)
        df['asym_ratio'] = df['downside_vol_5d'] / (df['upside_vol_5d'] + 1e-6)

        # 5. Jump Detection (극단 움직임)
        df['jump'] = (df['returns'].abs() > 3 * df['volatility_20d']).astype(int)
        df['jump_frequency'] = df['jump'].rolling(20).sum()

        # 6. Intraday Range (일중 변동폭)
        df['intraday_range'] = (df['High'] - df['Low']) / df['Close']
        df['range_volatility'] = df['intraday_range'].rolling(5).std()

        # 7. Parkinson Volatility (High-Low 기반)
        df['parkinson_vol'] = np.sqrt(
            1 / (4 * np.log(2)) * np.log(df['High'] / df['Low']) ** 2
        )

        # 기존 lag 특성
        for lag in [1, 2, 3, 5, 10]:
            df[f'vol_lag_{lag}'] = df['volatility_5d'].shift(lag)
            df[f'downside_lag_{lag}'] = df['downside_vol_5d'].shift(lag)

        # 타겟: 5일 후 변동성
        df['target_vol'] = df['volatility_5d'].shift(-5)

        # 결측치 제거
        df = df.dropna()

        print(f"✅ 특성 생성 완료: {df.shape[1]}개 컬럼")
        print(f"   - 기본 변동성: 3개")
        print(f"   - 비대칭 특성: 7개 (NEW)")
        print(f"   - Lag 특성: 10개")

        self.data = df
        return True

    def train_model(self):
        """모델 학습 (TimeSeriesSplit)"""
        print("\n🤖 향상된 Ridge 모델 학습 중...")

        # 특성 선택
        feature_cols = [
            # 기본 변동성
            'volatility_5d', 'volatility_10d', 'volatility_20d',

            # 비대칭 특성
            'leverage_effect', 'leverage_vol',
            'downside_vol_5d', 'downside_vol_20d',
            'upside_vol_5d', 'asym_ratio',
            'jump_frequency', 'intraday_range', 'range_volatility',
            'parkinson_vol',

            # Lag 특성
            'vol_lag_1', 'vol_lag_2', 'vol_lag_3', 'vol_lag_5', 'vol_lag_10',
            'downside_lag_1', 'downside_lag_2', 'downside_lag_3', 'downside_lag_5', 'downside_lag_10'
        ]

        X = self.data[feature_cols]
        y = self.data['target_vol']

        print(f"   특성 수: {len(feature_cols)}")
        print(f"   샘플 수: {len(X)}")

        # TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Ridge 모델
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)

            # 예측
            y_pred = model.predict(X_test_scaled)

            # 성능
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            print(f"   Fold {fold_idx}: R² = {r2:.4f}, RMSE = {rmse:.6f}, MAE = {mae:.6f}")

            cv_scores.append({
                'fold': fold_idx,
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            })

        # 평균 성능
        mean_r2 = np.mean([s['r2'] for s in cv_scores])
        std_r2 = np.std([s['r2'] for s in cv_scores])
        mean_rmse = np.mean([s['rmse'] for s in cv_scores])
        mean_mae = np.mean([s['mae'] for s in cv_scores])

        print(f"\n📊 TimeSeriesSplit CV 평균 성능:")
        print(f"   R² = {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"   RMSE = {mean_rmse:.6f}")
        print(f"   MAE = {mean_mae:.6f}")

        # 전체 데이터로 최종 모델 학습
        X_scaled = self.scaler.fit_transform(X)
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, y)

        # 결과 저장
        self.results = {
            'model_name': 'Enhanced Ridge with Asymmetric Features',
            'cv_mean_r2': mean_r2,
            'cv_std_r2': std_r2,
            'cv_mean_rmse': mean_rmse,
            'cv_mean_mae': mean_mae,
            'n_features': len(feature_cols),
            'n_samples': len(X),
            'cv_scores': cv_scores,
            'feature_list': feature_cols
        }

        # Feature importance (Ridge 계수)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)

        print(f"\n🔍 Top 10 중요 특성:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:30s}: {row['coefficient']:+.4f}")

        self.results['feature_importance'] = feature_importance.to_dict('records')

        return True

    def compare_with_baseline(self):
        """기존 모델과 비교"""
        print(f"\n📈 기존 모델과 비교...")

        # 기존 모델 성능 (README 기준)
        baseline_r2 = 0.303
        current_r2 = self.results['cv_mean_r2']

        improvement = current_r2 - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100

        print(f"\n   기존 Ridge (31개 특성): R² = {baseline_r2:.4f}")
        print(f"   향상 Ridge (비대칭 추가): R² = {current_r2:.4f}")
        print(f"   개선폭: {improvement:+.4f} ({improvement_pct:+.1f}%)")

        if current_r2 > baseline_r2:
            print(f"   ✅ 성능 향상 성공!")
        else:
            print(f"   ⚠️  성능 향상 실패 (추가 조정 필요)")

        self.results['comparison'] = {
            'baseline_r2': baseline_r2,
            'current_r2': current_r2,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }

        return True

    def save_results(self):
        """결과 저장"""
        try:
            self.results['metadata'] = {
                'experiment': 'enhanced_volatility_phase1',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'improvements': [
                    'Leverage effect',
                    'Downside volatility',
                    'Asymmetry ratio',
                    'Jump detection',
                    'Parkinson volatility'
                ]
            }

            output_path = "data/raw/enhanced_volatility_phase1_results.json"
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)

            print(f"\n💾 결과 저장: {output_path}")
            return True

        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")
            return False

    def run_experiment(self):
        """전체 실험 실행"""
        print("="*60)
        print("🚀 변동성 예측 성능 향상 실험 - Phase 1")
        print("="*60)
        print("목표: 비대칭 특성 추가로 R² 0.30 → 0.35\n")

        if not self.load_and_prepare_data():
            return False

        if not self.train_model():
            return False

        if not self.compare_with_baseline():
            return False

        if not self.save_results():
            return False

        print("\n" + "="*60)
        print("✅ Phase 1 실험 완료!")
        print("="*60)

        # 최종 요약
        current_r2 = self.results['cv_mean_r2']
        target_r2 = 0.35

        print(f"\n📋 최종 결과:")
        print(f"   달성 R²: {current_r2:.4f}")
        print(f"   목표 R²: {target_r2:.4f}")

        if current_r2 >= target_r2:
            print(f"   ✅ Phase 1 목표 달성!")
        else:
            gap = target_r2 - current_r2
            print(f"   ⚠️  목표 미달 (부족: {gap:.4f})")
            print(f"   → Phase 2 (GARCH) 필요")

        return True

if __name__ == "__main__":
    predictor = EnhancedVolatilityPredictor(
        ticker="SPY",
        start_date="2015-01-01",
        end_date="2024-12-31"
    )

    predictor.run_experiment()
