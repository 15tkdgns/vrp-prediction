#!/usr/bin/env python3
"""
SPY 데이터에서 2025년 1월-6월 데이터를 추출하고 대시보드용 파일로 저장
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def extract_2025_data():
    # 원본 데이터 로드
    df = pd.read_csv('data/raw/spy_data_2020_2025.csv')
    
    # 첫 번째 컬럼을 Date로 변환
    df['Date'] = pd.to_datetime(df['Unnamed: 0'])
    df = df.drop('Unnamed: 0', axis=1)
    
    print(f"전체 데이터 기간: {df['Date'].min()} ~ {df['Date'].max()}")
    print(f"전체 데이터 수: {len(df)}")
    
    # 2025년 1월 1일 ~ 6월 30일 필터링
    mask = (df['Date'] >= '2025-01-01') & (df['Date'] <= '2025-06-30')
    df_2025 = df[mask].copy()
    
    print(f"\n2025년 1-6월 데이터:")
    print(f"기간: {df_2025['Date'].min()} ~ {df_2025['Date'].max()}")
    print(f"데이터 수: {len(df_2025)}")
    
    if len(df_2025) == 0:
        print("2025년 1-6월 데이터가 없습니다.")
        return
    
    # 대시보드용 JSON 형태로 변환
    dashboard_data = []
    for _, row in df_2025.iterrows():
        dashboard_data.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': int(row['Volume']),
            'adj_close': float(row['Adj Close'])
        })
    
    # JSON 파일로 저장 (대시보드용)
    with open('data/raw/spy_2025_h1.json', 'w') as f:
        json.dump({
            'period': '2025-01-01 to 2025-06-30',
            'total_records': len(dashboard_data),
            'data': dashboard_data
        }, f, indent=2)
    
    # CSV 파일로도 저장
    df_2025.to_csv('data/raw/spy_2025_h1.csv', index=False)
    
    print(f"\n저장 완료:")
    print(f"- JSON: data/raw/spy_2025_h1.json")
    print(f"- CSV: data/raw/spy_2025_h1.csv")
    
    # 기본 통계 출력
    print(f"\n기본 통계:")
    print(f"- 시작가: ${df_2025['Close'].iloc[0]:.2f}")
    print(f"- 종료가: ${df_2025['Close'].iloc[-1]:.2f}")
    print(f"- 최고가: ${df_2025['High'].max():.2f}")
    print(f"- 최저가: ${df_2025['Low'].min():.2f}")
    print(f"- 평균 거래량: {df_2025['Volume'].mean():,.0f}")
    
    return dashboard_data

if __name__ == "__main__":
    extract_2025_data()