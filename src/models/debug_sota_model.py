#!/usr/bin/env python3
"""
Debug SOTA Model - 데이터 문제 진단
"""

import numpy as np
import pandas as pd
import yfinance as yf

def debug_data():
    """데이터 생성 과정 디버깅"""
    print("🔍 SOTA 모델 데이터 디버깅...")

    # 1. 기본 데이터 로드
    spy = yf.download('SPY', start='2010-01-01', end='2024-12-31', progress=False)
    spy['returns'] = spy['Close'].pct_change()
    spy = spy.dropna()

    print(f"기본 SPY 데이터: {len(spy)} 샘플")
    print(f"Returns 샘플: {spy['returns'].notna().sum()}")
    print(f"Returns 범위: {spy['returns'].min():.6f} ~ {spy['returns'].max():.6f}")

    # 2. 타겟 생성 테스트
    targets = pd.DataFrame(index=spy.index)
    returns = spy['returns']

    print(f"\n타겟 생성 테스트:")

    # 간단한 1일 타겟
    vol_1d_values = []
    for i in range(len(returns)):
        if i + 1 < len(returns):
            future_return = returns.iloc[i+1]
            vol_1d_values.append(abs(future_return))
        else:
            vol_1d_values.append(np.nan)

    targets['simple_1d'] = vol_1d_values
    print(f"Simple 1d 타겟: {targets['simple_1d'].notna().sum()} 유효 샘플")

    # 5일 변동성 타겟
    vol_5d_values = []
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_returns = returns.iloc[i+1:i+6]
            vol_5d_values.append(future_returns.std() * np.sqrt(252))  # 연율화
        else:
            vol_5d_values.append(np.nan)

    targets['vol_5d'] = vol_5d_values
    print(f"Vol 5d 타겟: {targets['vol_5d'].notna().sum()} 유효 샘플")

    # 3. 간단한 특성 생성
    features = pd.DataFrame(index=spy.index)
    features['volatility_5'] = returns.rolling(5).std()
    features['volatility_20'] = returns.rolling(20).std()

    print(f"\n특성 테스트:")
    print(f"Vol 5 특성: {features['volatility_5'].notna().sum()} 유효 샘플")
    print(f"Vol 20 특성: {features['volatility_20'].notna().sum()} 유효 샘플")

    # 4. 결합 테스트
    for target_name in ['simple_1d', 'vol_5d']:
        combined = pd.concat([features, targets[[target_name]]], axis=1).dropna()
        print(f"\n{target_name} 결합 후: {len(combined)} 샘플")

        if len(combined) > 0:
            print(f"  특성 통계:")
            print(f"    Vol 5: {combined['volatility_5'].mean():.6f} ± {combined['volatility_5'].std():.6f}")
            print(f"    Vol 20: {combined['volatility_20'].mean():.6f} ± {combined['volatility_20'].std():.6f}")
            print(f"  타겟 통계:")
            print(f"    {target_name}: {combined[target_name].mean():.6f} ± {combined[target_name].std():.6f}")

    # 5. 로그 변환 테스트
    print(f"\n로그 변환 테스트:")
    log_target = np.log(targets['vol_5d'] + 1e-8)
    print(f"Log target 유효 샘플: {log_target.notna().sum()}")
    print(f"Log target 무한대: {np.isinf(log_target).sum()}")

    return spy, features, targets

if __name__ == "__main__":
    debug_data()