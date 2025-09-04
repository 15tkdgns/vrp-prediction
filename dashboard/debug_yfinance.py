#!/usr/bin/env python3
import yfinance as yf
import pandas as pd

# SPY 데이터 다운로드 테스트
print("SPY 데이터 컬럼 확인...")
spy_data = yf.download('SPY', start='2024-01-01', end='2024-01-10', auto_adjust=True, progress=False)
print("SPY 컬럼:", spy_data.columns.tolist())
print("SPY 데이터 타입:", type(spy_data))
print("샘플 데이터:")
print(spy_data.head())

print("\nVIX 데이터 컬럼 확인...")
vix_data = yf.download('^VIX', start='2024-01-01', end='2024-01-10', auto_adjust=True, progress=False)
print("VIX 컬럼:", vix_data.columns.tolist())
print("VIX 데이터 타입:", type(vix_data))
print("샘플 데이터:")
print(vix_data.head())