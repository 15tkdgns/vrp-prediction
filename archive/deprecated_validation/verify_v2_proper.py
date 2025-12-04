#!/usr/bin/env python3
"""
V2 Regime-Switching 올바른 검증
- 원본 V2 방식 (80/20 split) vs 올바른 TimeSeriesSplit 비교
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔍 V2 Regime-Switching 검증: 원본 vs 올바른 방법")
print("="*70)

# 데이터 로드
spy = yf.Ticker("SPY")
df = spy.history(start="2015-01-01", end="2024-12-31")
df.index = pd.to_datetime(df.index).tz_localize(None)

df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['volatility'] = df['returns'].rolling(20).std()

# V2 타겟 (잘못된 방식 - shift(-5))
df['target_vol_5d_v2'] = df['volatility'].shift(-5)

# 올바른 타겟 (V0 방식)
targets_correct = []
for i in range(len(df)):
    if i + 5 < len(df):
        future_returns = df['returns'].iloc[i+1:i+6]
        targets_correct.append(future_returns.std())
    else:
        targets_correct.append(np.nan)
df['target_vol_5d_correct'] = targets_correct

# Lag 특성
for lag in [1, 2, 3, 5, 10]:
    df[f'vol_lag_{lag}'] = df['volatility'].shift(lag)

df = df.dropna()

print(f"\n데이터: {len(df)} 샘플")

# Regime 구분
vol_values = df['volatility'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=42)
df['regime'] = kmeans.fit_predict(vol_values)

cluster_means = df.groupby('regime')['volatility'].mean()
if cluster_means[0] > cluster_means[1]:
    df['regime'] = 1 - df['regime']

print(f"고변동 regime: {(df['regime']==1).sum()} 샘플")
print(f"저변동 regime: {(df['regime']==0).sum()} 샘플")

features = ['volatility'] + [f'vol_lag_{i}' for i in [1, 2, 3, 5, 10]]

# ============================================================
# 방법 1: V2 원본 방식 (80/20 split + 잘못된 타겟)
# ============================================================
print("\n" + "="*70)
print("1️⃣  V2 원본 방식 (80/20 split + shift(-5) 타겟)")
print("="*70)

predictions_v2 = []
actuals_v2 = []

for regime in [0, 1]:
    df_regime = df[df['regime'] == regime].copy()

    if len(df_regime) < 100:
        continue

    X = df_regime[features]
    y = df_regime['target_vol_5d_v2']

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    predictions_v2.extend(pred)
    actuals_v2.extend(y_test)

r2_v2_original = r2_score(actuals_v2, predictions_v2)
print(f"R² = {r2_v2_original:.4f}")
print(f"샘플 수: {len(predictions_v2)}")

# ============================================================
# 방법 2: V2 방식 + 올바른 타겟
# ============================================================
print("\n" + "="*70)
print("2️⃣  V2 방식 (80/20 split) + 올바른 타겟")
print("="*70)

predictions_v2_correct_target = []
actuals_v2_correct_target = []

for regime in [0, 1]:
    df_regime = df[df['regime'] == regime].copy()

    if len(df_regime) < 100:
        continue

    X = df_regime[features]
    y = df_regime['target_vol_5d_correct']

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    predictions_v2_correct_target.extend(pred)
    actuals_v2_correct_target.extend(y_test)

r2_v2_correct_target = r2_score(actuals_v2_correct_target, predictions_v2_correct_target)
print(f"R² = {r2_v2_correct_target:.4f}")
print(f"샘플 수: {len(predictions_v2_correct_target)}")

# ============================================================
# 방법 3: TimeSeriesSplit + 올바른 타겟
# ============================================================
print("\n" + "="*70)
print("3️⃣  올바른 방법 (TimeSeriesSplit + 올바른 타겟)")
print("="*70)

X = df[features]
y = df['target_vol_5d_correct']

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Regime 구분 (train만 사용)
    regime_train = df['regime'].iloc[train_idx]
    regime_test = df['regime'].iloc[test_idx]

    models = {}

    for regime in [0, 1]:
        regime_mask = (regime_train == regime)
        X_train_regime = X_train[regime_mask]
        y_train_regime = y_train[regime_mask]

        if len(X_train_regime) < 10:
            continue

        model = Ridge(alpha=1.0)
        model.fit(X_train_regime, y_train_regime)
        models[regime] = model

    # 예측
    predictions = []
    for i in range(len(X_test)):
        regime = regime_test.iloc[i]
        if regime in models:
            pred = models[regime].predict(X_test.iloc[[i]])[0]
        else:
            pred = y_train.mean()
        predictions.append(pred)

    r2 = r2_score(y_test, predictions)
    cv_scores.append(r2)
    print(f"   Fold {fold}: R² = {r2:.4f}")

r2_proper = np.mean(cv_scores)
print(f"\n평균 R² = {r2_proper:.4f} (±{np.std(cv_scores):.4f})")

# ============================================================
# 최종 비교
# ============================================================
print("\n" + "="*70)
print("📊 최종 비교")
print("="*70)

print(f"\n{'방법':<50s} {'R²':>10s} {'타겟':>15s} {'검증':>20s}")
print("-"*70)
print(f"{'V2 원본 (보고된 값)':<50s} {r2_v2_original:>10.4f} {'shift(-5)':>15s} {'80/20 split':>20s}")
print(f"{'V2 방식 + 올바른 타겟':<50s} {r2_v2_correct_target:>10.4f} {'V0 방식':>15s} {'80/20 split':>20s}")
print(f"{'올바른 검증 (TimeSeriesSplit)':<50s} {r2_proper:>10.4f} {'V0 방식':>15s} {'TimeSeriesSplit':>20s}")

print("\n" + "="*70)
print("💡 결론")
print("="*70)

if r2_v2_original > 0.32:
    print(f"✅ V2 원본 R² {r2_v2_original:.4f} 재현 성공")
else:
    print(f"❌ V2 원본 R² 재현 실패")

print(f"\n🔍 타겟 설계 영향:")
print(f"   shift(-5) 타겟: R² = {r2_v2_original:.4f}")
print(f"   올바른 타겟:   R² = {r2_v2_correct_target:.4f}")
print(f"   차이: {r2_v2_original - r2_v2_correct_target:+.4f}")

if r2_v2_original - r2_v2_correct_target > 0.1:
    print(f"   ⚠️  타겟 설계에 데이터 누출 의심 (차이 > 0.1)")

print(f"\n🔍 검증 방법 영향:")
print(f"   80/20 split:      R² = {r2_v2_correct_target:.4f}")
print(f"   TimeSeriesSplit:  R² = {r2_proper:.4f}")
print(f"   차이: {r2_v2_correct_target - r2_proper:+.4f}")

if r2_v2_correct_target - r2_proper > 0.05:
    print(f"   ⚠️  80/20 split이 과대평가 (차이 > 0.05)")

print("\n📋 최종 판단:")
if r2_proper < 0.20:
    print("   ❌ Regime-Switching 방법은 실제로 효과 없음")
    print("   ❌ V2 R² 0.328은 부적절한 검증 방법으로 인한 과대평가")
    print("   ✅ 기본 Ridge (V0 R² 0.303)가 더 신뢰 가능")
else:
    print(f"   ✅ Regime-Switching 유효 (올바른 R² = {r2_proper:.4f})")
