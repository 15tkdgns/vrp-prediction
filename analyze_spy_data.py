#!/usr/bin/env python3
"""
SPY 데이터 분석 스크립트
생성된 SPY 데이터를 로드하고 기본적인 분석 수행
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_analyze_spy_data():
    """SPY 데이터 로드 및 분석"""
    
    # 데이터 파일 경로
    data_file = "data/raw/spy_data_2020_2025.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_file}")
        return None
    
    print("📊 SPY 데이터 분석 시작...")
    
    # 데이터 로드
    spy_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print(f"📈 데이터 크기: {spy_data.shape}")
    print(f"📅 데이터 기간: {spy_data.index[0]} ~ {spy_data.index[-1]}")
    
    # 기본 정보
    print("\n📊 데이터 기본 정보:")
    print(spy_data.info())
    
    # 결측값 확인
    print(f"\n🔍 결측값 개수: {spy_data.isnull().sum().sum()}")
    
    # 가격 변화 계산
    spy_data['Daily_Return'] = spy_data['Close'].pct_change()
    spy_data['Price_Change'] = spy_data['Close'].diff()
    
    # 통계 정보
    print("\n📊 주요 통계:")
    print(f"평균 종가: ${spy_data['Close'].mean():.2f}")
    print(f"최고가: ${spy_data['High'].max():.2f}")
    print(f"최저가: ${spy_data['Low'].min():.2f}")
    print(f"평균 일일 수익률: {spy_data['Daily_Return'].mean()*100:.4f}%")
    print(f"일일 수익률 표준편차: {spy_data['Daily_Return'].std()*100:.4f}%")
    print(f"최대 일일 상승률: {spy_data['Daily_Return'].max()*100:.2f}%")
    print(f"최대 일일 하락률: {spy_data['Daily_Return'].min()*100:.2f}%")
    
    # 연도별 수익률
    print("\n📊 연도별 수익률:")
    yearly_returns = spy_data.groupby(spy_data.index.year)['Close'].agg(['first', 'last'])
    yearly_returns['Annual_Return'] = (yearly_returns['last'] / yearly_returns['first'] - 1) * 100
    
    for year in yearly_returns.index:
        if year < 2025:  # 2025년은 아직 끝나지 않았으므로
            print(f"{year}: {yearly_returns.loc[year, 'Annual_Return']:.2f}%")
    
    # 2025년 현재까지 수익률
    start_2025 = spy_data[spy_data.index.year == 2025]['Close'].iloc[0]
    latest_2025 = spy_data[spy_data.index.year == 2025]['Close'].iloc[-1]
    ytd_return = (latest_2025 / start_2025 - 1) * 100
    print(f"2025년 현재까지: {ytd_return:.2f}% (YTD)")
    
    # 월별 통계
    print("\n📅 월별 거래일 수:")
    monthly_trading_days = spy_data.groupby([spy_data.index.year, spy_data.index.month]).size()
    print(monthly_trading_days.tail(12))
    
    # 가장 큰 변동일
    max_gain_date = spy_data['Daily_Return'].idxmax()
    max_loss_date = spy_data['Daily_Return'].idxmin()
    
    print(f"\n📈 최대 상승일: {max_gain_date.date()} ({spy_data.loc[max_gain_date, 'Daily_Return']*100:.2f}%)")
    print(f"📉 최대 하락일: {max_loss_date.date()} ({spy_data.loc[max_loss_date, 'Daily_Return']*100:.2f}%)")
    
    # 최근 30일 통계
    recent_data = spy_data.tail(30)
    print(f"\n📊 최근 30일 통계:")
    print(f"평균 종가: ${recent_data['Close'].mean():.2f}")
    print(f"평균 일일 수익률: {recent_data['Daily_Return'].mean()*100:.4f}%")
    print(f"변동성(표준편차): {recent_data['Daily_Return'].std()*100:.4f}%")
    
    return spy_data

def save_summary_stats(spy_data):
    """요약 통계를 JSON 파일로 저장"""
    
    if spy_data is None:
        return
    
    summary_stats = {
        'data_period': {
            'start': spy_data.index[0].strftime('%Y-%m-%d'),
            'end': spy_data.index[-1].strftime('%Y-%m-%d'),
            'total_days': len(spy_data)
        },
        'price_stats': {
            'mean_close': float(spy_data['Close'].mean()),
            'max_high': float(spy_data['High'].max()),
            'min_low': float(spy_data['Low'].min()),
            'latest_close': float(spy_data['Close'].iloc[-1])
        },
        'return_stats': {
            'mean_daily_return': float(spy_data['Daily_Return'].mean()),
            'daily_volatility': float(spy_data['Daily_Return'].std()),
            'max_daily_gain': float(spy_data['Daily_Return'].max()),
            'max_daily_loss': float(spy_data['Daily_Return'].min())
        }
    }
    
    # JSON 파일로 저장
    import json
    with open('data/raw/spy_summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print("\n💾 요약 통계 저장 완료: data/raw/spy_summary_stats.json")

if __name__ == "__main__":
    data = load_and_analyze_spy_data()
    if data is not None:
        save_summary_stats(data)
        print("\n✅ SPY 데이터 분석 완료!")