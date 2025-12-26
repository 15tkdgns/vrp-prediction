#!/usr/bin/env python3
"""상관관계 매트릭스 및 특성 중요도 계산"""
import pandas as pd
import numpy as np
import json
import os

# 데이터 로드
spy = pd.read_csv('data/raw/spy_data_2020_2025.csv', index_col=0, parse_dates=True)
spy['returns'] = spy['Close'].pct_change()
spy['RV_1d'] = spy['returns'].abs() * np.sqrt(252) * 100
spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100

# VIX 데이터 로드
import yfinance as yf
vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)
spy['VIX'] = vix['Close'].reindex(spy.index).ffill()

# 특성 생성
spy['VIX_lag1'] = spy['VIX'].shift(1)
spy['VIX_lag5'] = spy['VIX'].shift(5)
spy['VIX_change'] = spy['VIX'].pct_change()
spy['VRP'] = spy['VIX'] - spy['RV_22d']
spy['VRP_lag1'] = spy['VRP'].shift(1)
spy['VRP_lag5'] = spy['VRP'].shift(5)
spy['VRP_ma5'] = spy['VRP'].rolling(5).mean()
spy['return_5d'] = spy['returns'].rolling(5).sum() * 100
spy['return_22d'] = spy['returns'].rolling(22).sum() * 100
spy['regime_high'] = (spy['VIX'] > 25).astype(int)

spy = spy.dropna()

# 특성 목록
features = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 'VIX_change', 
            'VRP_lag1', 'VRP_lag5', 'VRP_ma5', 'regime_high', 'return_5d', 'return_22d']

# 상관관계 매트릭스 계산
corr_matrix = spy[features].corr().round(2).values.tolist()

print("Correlation Matrix calculated!")
print("Shape:", len(corr_matrix), "x", len(corr_matrix[0]))

# JSON 저장
result = {
    'correlation_matrix': corr_matrix,
    'features': features,
    'n_samples': len(spy)
}

os.makedirs('data/results', exist_ok=True)
with open('data/results/correlation_matrix.json', 'w') as f:
    json.dump(result, f, indent=2)

print("Saved to data/results/correlation_matrix.json")

# 상관관계 출력
print("\nCorrelation Matrix Preview (first 6 features):")
corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)
print(corr_df.iloc[:6, :6])
