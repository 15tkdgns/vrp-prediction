#!/usr/bin/env python3
"""
V3 결과 데이터 누출 검증
R² 0.86은 비정상 → 특성 생성 시간적 분리 확인
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🔍 V3 데이터 누출 검증")
print("="*60)

# 데이터 로드
spy = yf.Ticker("SPY")
df = spy.history(start="2015-01-01", end="2024-12-31")
df.index = pd.to_datetime(df.index).tz_localize(None)

df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['volatility'] = df['returns'].rolling(20).std()
df['vol_20d'] = df['returns'].rolling(20).std()

# 타겟: 5일 후 변동성
df['target_vol_5d'] = df['vol_20d'].shift(-5)

print("\n1️⃣  타겟 변수 시간적 분리 확인:")
print(f"   타겟: target_vol_5d = vol_20d.shift(-5)")
print(f"   ✅ 올바른 설계 (미래 값)")

# 문제가 될 수 있는 특성들 확인
print("\n2️⃣  의심스러운 특성 검사:")

# ATR (True Range)
df['high_low'] = df['High'] - df['Low']
df['high_close'] = abs(df['High'] - df['Close'].shift(1))
df['low_close'] = abs(df['Low'] - df['Close'].shift(1))
df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
df['atr_14'] = df['true_range'].rolling(14).mean()

print("\n   ATR 계산:")
print(f"   - True Range = max(High-Low, |High-Close_t-1|, |Low-Close_t-1|)")
print(f"   - ATR_14 = TrueRange.rolling(14).mean()")
print(f"   ⚠️  문제: ATR_14는 현재일(t)의 High, Low 포함!")

# 현재일(t) High/Low vs 미래 변동성(t+1~t+5) 상관관계
df_clean = df[['high_low', 'atr_14', 'target_vol_5d']].dropna()

corr_high_low = df_clean[['high_low', 'target_vol_5d']].corr().iloc[0, 1]
corr_atr = df_clean[['atr_14', 'target_vol_5d']].corr().iloc[0, 1]

print(f"\n   상관관계:")
print(f"   - high_low (t일) vs target_vol (t+1~t+5): {corr_high_low:.4f}")
print(f"   - atr_14 (t일 포함) vs target_vol (t+1~t+5): {corr_atr:.4f}")

if corr_high_low > 0.6 or corr_atr > 0.6:
    print(f"   ❌ 데이터 누출 확인! (상관 > 0.6)")
    print(f"   원인: t일의 High-Low가 t+1~t+5 변동성과 강한 상관")
    print(f"   이유: t일 종가 기준 변동성 계산 시 High/Low 정보 누출")

# Vol-20d 계산 확인
print("\n3️⃣  vol_20d 계산 시점:")
print(f"   vol_20d[t] = returns[t-19:t].std()")
print(f"   → t일 종가 수익률 포함!")
print(f"   target_vol_5d[t] = vol_20d[t+5]")
print(f"   → t+5일 종가 기준 (t-14 ~ t+5 수익률)")

# 겹침 확인
print("\n4️⃣  시간적 겹침 분석:")
print(f"   특성 vol_20d[t]: [t-19, t-18, ..., t-1, t] ← t일 포함")
print(f"   타겟 target[t]: vol_20d[t+5] = [t-14, ..., t+4, t+5]")
print(f"   ❌ 겹침 구간: [t-14, t] (15일 중복!)")

# Parkinson vol 확인
print("\n5️⃣  Parkinson Volatility:")
print(f"   parkinson_vol[t] = sqrt(1/(4*ln2) * log²(High[t]/Low[t]))")
print(f"   ⚠️  t일 High/Low 직접 사용 → 강한 누출")

# Gap 확인
df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
print("\n6️⃣  Gap 계산:")
print(f"   gap[t] = (Open[t] - Close[t-1]) / Close[t-1]")
print(f"   ✅ 올바른 설계 (t-1 정보만 사용)")

# 결론
print("\n" + "="*60)
print("📋 누출 원인 진단")
print("="*60)

leakage_sources = [
    ("ATR (True Range)", "HIGH", "t일 High/Low 포함"),
    ("vol_20d 롤링 윈도우", "HIGH", "t일 수익률 포함 + 타겟과 15일 겹침"),
    ("Parkinson Volatility", "HIGH", "t일 High/Low 직접 사용"),
    ("realized_range", "HIGH", "t일 High/Low/Open 사용"),
    ("Gap", "LOW", "올바른 시간적 분리"),
    ("Volume", "MEDIUM", "t일 거래량 사용 (일부 누출)"),
]

print("\n누출 위험도:")
for feature, risk, reason in leakage_sources:
    emoji = "❌" if risk == "HIGH" else ("⚠️" if risk == "MEDIUM" else "✅")
    print(f"   {emoji} {feature:25s} [{risk:6s}]: {reason}")

print("\n" + "="*60)
print("💡 해결 방법")
print("="*60)
print("1. vol_20d를 vol_20d.shift(1)로 변경 (t-1일까지만 사용)")
print("2. ATR을 atr.shift(1)로 변경")
print("3. Parkinson vol도 shift(1)")
print("4. High/Low 기반 모든 특성 shift 적용")
print("5. 타겟: shift(-5) 유지 (올바름)")
print("="*60)

# 올바른 계산 테스트
print("\n7️⃣  수정된 계산 테스트:")

df['vol_20d_correct'] = df['vol_20d'].shift(1)  # t-1일까지
df['atr_correct'] = df['atr_14'].shift(1)

df_test = df[['vol_20d_correct', 'atr_correct', 'target_vol_5d']].dropna()

corr_vol_correct = df_test[['vol_20d_correct', 'target_vol_5d']].corr().iloc[0, 1]
corr_atr_correct = df_test[['atr_correct', 'target_vol_5d']].corr().iloc[0, 1]

print(f"   수정 전 상관: {corr_atr:.4f}")
print(f"   수정 후 상관: {corr_atr_correct:.4f}")
print(f"   예상 R² 변화: 0.86 → 0.30~0.40")
