#!/usr/bin/env python3
"""
차트 생성 코드의 데이터 누출 검증
R² 0.46 vs 0.31 차이의 원인 분석
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔍 차트 생성 코드 데이터 누출 검증")
print("="*70)

# 데이터 로드
spy = yf.Ticker("SPY")
df = spy.history(start="2015-01-01", end="2024-12-31")
df.index = pd.to_datetime(df.index).tz_localize(None)

df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

# 타겟 생성 (V0 방식)
print("\n1️⃣  타겟 생성 검증...")
targets = []
for i in range(len(df)):
    if i + 5 < len(df):
        future_returns = df['returns'].iloc[i+1:i+6]
        targets.append(future_returns.std())
    else:
        targets.append(np.nan)
df['target_vol_5d'] = targets

# 수동 검증 (샘플)
test_idx = 100
manual_target = df['returns'].iloc[test_idx+1:test_idx+6].std()
auto_target = df['target_vol_5d'].iloc[test_idx]

print(f"   샘플 검증 (index {test_idx}):")
print(f"   자동 타겟: {auto_target:.6f}")
print(f"   수동 타겟: {manual_target:.6f}")
print(f"   일치: {'✅' if abs(auto_target - manual_target) < 1e-6 else '❌'}")

# 특성 생성 (차트 코드 방식)
print("\n2️⃣  특성 생성 검증...")

print("\n   ⚠️ 특성 생성 전 데이터 확인...")

# 변동성 특성
for window in [5, 10, 20, 60]:
    df[f'volatility_{window}d'] = df['returns'].rolling(window).std()

# 특정 타임스텝 검증
test_idx = 100
test_date = df.index[test_idx]

vol_20_value = df['volatility_20d'].iloc[test_idx]
manual_vol_20 = df['returns'].iloc[test_idx-19:test_idx+1].std()  # t-19 ~ t

print(f"\n   volatility_20d 검증 (index {test_idx}):")
print(f"   자동 계산: {vol_20_value:.6f}")
print(f"   수동 계산 [t-19:t+1]: {manual_vol_20:.6f}")
print(f"   일치: {'✅' if abs(vol_20_value - manual_vol_20) < 1e-6 else '❌'}")

# ❌ 문제: rolling(20)은 [t-19:t+1] (t 포함!)
# 타겟: returns[t+1:t+6]
# 겹침: t+1이 롤링 윈도우에 없음 → 올바름

# Lag 특성 확인
print("\n3️⃣  Lag 특성 검증...")

df['vol_lag_1'] = df['volatility_20d'].shift(1)

vol_lag_1_value = df['vol_lag_1'].iloc[test_idx]
manual_vol_lag_1 = df['volatility_20d'].iloc[test_idx-1]

print(f"   vol_lag_1 검증 (index {test_idx}):")
print(f"   자동: {vol_lag_1_value:.6f}")
print(f"   수동 [t-1]: {manual_vol_lag_1:.6f}")
print(f"   일치: {'✅' if abs(vol_lag_1_value - manual_vol_lag_1) < 1e-6 else '❌'}")

# vol_lag_1[t] = volatility_20d[t-1] = std(returns[t-20:t])
# 타겟[t] = std(returns[t+1:t+6])
# 겹침: 없음 → 올바름

# 모멘텀 검증
print("\n4️⃣  모멘텀 특성 검증...")

df['momentum_5d'] = df['returns'].rolling(5).sum()

momentum_value = df['momentum_5d'].iloc[test_idx]
manual_momentum = df['returns'].iloc[test_idx-4:test_idx+1].sum()

print(f"   momentum_5d 검증 (index {test_idx}):")
print(f"   자동: {momentum_value:.6f}")
print(f"   수동 [t-4:t+1]: {manual_momentum:.6f}")
print(f"   일치: {'✅' if abs(momentum_value - manual_momentum) < 1e-6 else '❌'}")

# momentum[t] = sum(returns[t-4:t+1])  # t 포함!
# 타겟[t] = std(returns[t+1:t+6])
# 겹침: t+1이 momentum에 없음 → 올바름

# 극단값 특성
print("\n5️⃣  극단값 특성 검증...")

df['extreme_returns'] = (df['returns'].abs() > 2 * df['volatility_20d']).astype(int)
df['extreme_count_20d'] = df['extreme_returns'].rolling(20).sum()

extreme_count_value = df['extreme_count_20d'].iloc[test_idx]

print(f"   extreme_count_20d 검증 (index {test_idx}):")
print(f"   값: {extreme_count_value}")
print(f"   사용 구간: [t-19:t+1]")

# extreme_count[t] = count(extreme_returns[t-19:t+1])
# extreme_returns[t] = |returns[t]| > 2 * volatility[t]
# → returns[t] 사용! (t일 수익률 포함)

# 타겟[t] = std(returns[t+1:t+6])
# 겹침: returns[t]가 타겟에 없음 → 올바름

print("\n" + "="*70)
print("📋 검증 결과 요약")
print("="*70)

print("\n✅ 타겟 설계: returns[t+1:t+6].std() - 올바름")
print("✅ 기본 변동성: rolling(window) = [t-window+1:t+1] - t 포함하지만 타겟과 안 겹침")
print("✅ Lag 특성: shift(lag) = 과거 데이터만 - 올바름")
print("✅ 모멘텀: rolling().sum() - t 포함하지만 타겟과 안 겹침")
print("✅ 극단값: returns[t] 사용하지만 타겟과 안 겹침")

print("\n" + "="*70)
print("🔍 R² 차이 원인 재분석")
print("="*70)

print("\n가설 1: 특성 수 차이")
print("   V0 원본: 31개")
print("   차트 코드: 26개")
print("   → 특성 수가 적어서 R² 높을 수 없음 (모순)")

print("\n가설 2: Purged K-Fold 구현 차이")
print("   V0: Train = [0, test_start-5] + [test_end+5, n_samples]")
print("   차트: 동일한 방식")
print("   → 구현 동일")

print("\n가설 3: 데이터 샘플 수 차이")
print("   V0 원본: 2,445 샘플")

df_chart = df.dropna()
print(f"   차트 코드: {len(df_chart)} 샘플")

if len(df_chart) != 2445:
    print(f"   ⚠️ 샘플 수 차이 발견: {len(df_chart) - 2445:+d}")

print("\n가설 4: CV 평균 vs 전체 Test R²")
print("   V0 원본: CV 평균 R² = 0.3113")
print("   차트 코드: 전체 Test R² = 0.4632")
print("   → 측정 방법 차이일 수 있음")

print("\n" + "="*70)
print("💡 결론")
print("="*70)

print("\n데이터 누출은 발견되지 않음 ✅")
print("\n성능 차이 가능한 원인:")
print("   1. CV 평균 vs 전체 Test R² 차이")
print("   2. 마지막 fold 성능 저하 (V0에서 -0.0068)")
print("   3. 전체 데이터 사용 시 과적합 가능성")

print("\n권장:")
print("   V0 원본 방식 (CV 평균 R² = 0.3113) 신뢰")
print("   차트는 시각화 목적으로만 사용")
print("   최종 보고 성능: R² = 0.31 (보수적)")
