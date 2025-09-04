#!/usr/bin/env python3
"""
SPY 데이터 수집 스크립트
2020년 1월부터 2025년 8월까지의 SPY (S&P 500 ETF) 데이터를 수집하여 CSV 파일로 저장
"""

import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime
import os

def main():
    print("📊 SPY 데이터 수집 시작...")
    
    try:
        # SPY 데이터 수집 (2020-01-01 ~ 2025-08-31)
        print("🔍 SPY 데이터 다운로드 중... (2020-01-01 ~ 2025-08-31)")
        spy_data = fdr.DataReader('SPY', '2020-01-01', '2025-08-31')
        
        # 데이터 정보 출력
        print(f"📈 수집된 데이터 크기: {spy_data.shape}")
        print(f"📅 데이터 기간: {spy_data.index[0]} ~ {spy_data.index[-1]}")
        print(f"📊 컬럼: {list(spy_data.columns)}")
        
        # 데이터 미리보기
        print("\n🔍 데이터 미리보기 (첫 5행):")
        print(spy_data.head())
        
        print("\n🔍 데이터 미리보기 (마지막 5행):")
        print(spy_data.tail())
        
        # 기본 통계
        print("\n📊 기본 통계:")
        print(spy_data.describe())
        
        # 데이터 저장 경로 설정
        data_dir = "data/raw"
        os.makedirs(data_dir, exist_ok=True)
        
        # CSV 파일로 저장
        output_file = os.path.join(data_dir, "spy_data_2020_2025.csv")
        spy_data.to_csv(output_file)
        print(f"\n💾 데이터 저장 완료: {output_file}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"📁 파일 크기: {file_size:.1f} KB")
        
        # 월별 데이터 수 확인
        print("\n📅 월별 데이터 개수:")
        monthly_counts = spy_data.resample('M').size()
        print(monthly_counts.tail(12))  # 최근 12개월
        
        # 연도별 요약
        print("\n📊 연도별 가격 요약:")
        yearly_summary = spy_data.resample('Y').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean'
        })
        print(yearly_summary)
        
        return spy_data
        
    except Exception as e:
        print(f"❌ 데이터 수집 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    data = main()
    if data is not None:
        print("\n✅ SPY 데이터 수집 및 저장 완료!")
    else:
        print("\n❌ SPY 데이터 수집 실패!")