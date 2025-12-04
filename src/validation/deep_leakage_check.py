#!/usr/bin/env python3
"""
근본 문제 진단: vol_20d 롤링 윈도우 겹침
"""

import pandas as pd
import numpy as np
import yfinance as yf

print("="*70)
print("🔬 근본 문제 진단: 타겟 변수 설계 오류")
print("="*70)

spy = yf.Ticker("SPY")
df = spy.history(start="2024-01-01", end="2024-02-01")  # 샘플
df.index = pd.to_datetime(df.index).tz_localize(None)

df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

# 현재 설계
df['vol_20d'] = df['returns'].rolling(20).std()
df['target_vol_5d'] = df['vol_20d'].shift(-5)

# 예시: 2024-01-15 행
example_idx = 10 if len(df) > 10 else 0
example_date = df.index[example_idx]

print(f"\n📅 예시: {example_date.strftime('%Y-%m-%d')} (t={example_idx})")
print("="*70)

# t일의 특성
print(f"\n1️⃣  특성 vol_20d[t]:")
print(f"   vol_20d[{example_idx}] = returns[{example_idx-19}:{example_idx}].std()")
print(f"   → 사용 기간: {df.index[max(0, example_idx-19)]} ~ {example_date}")

# t일의 타겟
target_idx = example_idx + 5
if target_idx < len(df):
    print(f"\n2️⃣  타겟 target_vol_5d[t]:")
    print(f"   target_vol_5d[{example_idx}] = vol_20d[{target_idx}]")
    print(f"   = returns[{target_idx-19}:{target_idx}].std()")
    print(f"   → 사용 기간: {df.index[max(0, target_idx-19)]} ~ {df.index[target_idx]}")

    # 겹침 구간
    feature_start = max(0, example_idx - 19)
    feature_end = example_idx
    target_start = max(0, target_idx - 19)
    target_end = target_idx

    overlap_start = max(feature_start, target_start)
    overlap_end = min(feature_end, target_end)

    if overlap_end >= overlap_start:
        overlap_days = overlap_end - overlap_start + 1
        print(f"\n❌ 겹침 발견:")
        print(f"   특성 구간: [{feature_start}, {feature_end}]")
        print(f"   타겟 구간: [{target_start}, {target_end}]")
        print(f"   겹침 구간: [{overlap_start}, {overlap_end}] ({overlap_days}일)")
        print(f"   겹침 비율: {overlap_days}/20 = {overlap_days/20*100:.1f}%")

print("\n" + "="*70)
print("💡 올바른 타겟 설계")
print("="*70)
print("\n현재 (잘못됨):")
print("   target_vol_5d[t] = vol_20d[t+5]")
print("   → t일 데이터가 타겟 계산에 포함됨 (15일 겹침)")

print("\n수정안 1: 미래 구간 변동성")
print("   target_vol_5d[t] = returns[t+1:t+21].std()")
print("   → 완전 분리 (t+1부터 사용)")

print("\n수정안 2: 미래 5일 변동성 (원래 의도)")
print("   target_vol_5d[t] = returns[t+1:t+6].std()")
print("   → 5일 후 실제 변동성 (완전 분리)")

print("\n수정안 3: 미래 realized volatility")
print("   target_vol_5d[t] = sqrt(sum(returns[t+1:t+6]²))")
print("   → 5일 realized volatility (완전 분리)")

print("\n" + "="*70)
print("🔍 shift(1) 효과 없는 이유")
print("="*70)
print("\nshift(1) 적용:")
print("   vol_20d_shifted = vol_20d.shift(1)")
print("   target = vol_20d.shift(-5)")
print("\n문제:")
print("   vol_20d_shifted[t] = vol_20d[t-1] = returns[t-20:t-1].std()")
print("   target[t] = vol_20d[t+5] = returns[t-14:t+5].std()")
print("   ❌ 여전히 겹침: [t-14, t-1] (14일)")

print("\n올바른 방법: 타겟을 완전 미래로")
print("   vol_20d[t] = returns[t-19:t].std()")
print("   target[t] = returns[t+1:t+21].std()")
print("   ✅ 완전 분리: 겹침 0일")

print("\n" + "="*70)
print("📋 결론")
print("="*70)
print("\n현재 V3 Fixed도 여전히 누출:")
print("   - shift(1)만으로는 롤링 윈도우 겹침 해결 불가")
print("   - 타겟 변수 자체를 재설계해야 함")
print("\n권장:")
print("   target_vol_5d[t] = returns[t+1:t+6].std()")
print("   또는")
print("   target_vol_20d[t] = returns[t+1:t+21].std()")
