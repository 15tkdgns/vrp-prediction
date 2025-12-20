#!/usr/bin/env python3
"""
기간 분리(Sub-period) 분석
==========================

기간별 모델 성능 비교:
1. 팬데믹 구간 (2020.03-2020.12)
2. 회복기 (2021-2022)
3. 정상화 (2023-2025)

실행: python src/subperiod_analysis.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    spy['returns'] = spy['Close'].pct_change()
    
    # 변동성 계산
    spy['RV_1d'] = spy['returns'].abs() * np.sqrt(252) * 100
    spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
    # 특성 생성
    spy['VIX_lag1'] = spy['VIX'].shift(1)
    spy['VIX_lag5'] = spy['VIX'].shift(5)
    spy['VIX_change'] = spy['VIX'].pct_change()
    spy['VRP_lag1'] = spy['VRP'].shift(1)
    spy['VRP_lag5'] = spy['VRP'].shift(5)
    spy['VRP_ma5'] = spy['VRP'].rolling(5).mean()
    spy['regime_high'] = (spy['VIX'] >= 25).astype(int)
    spy['regime_crisis'] = (spy['VIX'] >= 35).astype(int)
    spy['return_5d'] = spy['returns'].rolling(5).sum()
    spy['return_22d'] = spy['returns'].rolling(22).sum()
    
    spy = spy.replace([np.inf, -np.inf], np.nan).dropna()
    
    return spy


def train_and_evaluate(train_data, test_data, feature_cols):
    """모델 학습 및 평가"""
    X_train = train_data[feature_cols].values
    y_train = train_data['RV_future'].values
    X_test = test_data[feature_cols].values
    y_test_rv = test_data['RV_future'].values
    y_test_vrp = test_data['VRP_true'].values
    vix_test = test_data['VIX'].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    model.fit(X_train_s, y_train)
    
    rv_pred = model.predict(X_test_s)
    vrp_pred = vix_test - rv_pred
    
    # 성능 지표
    r2 = r2_score(y_test_vrp, vrp_pred)
    direction_acc = ((vrp_pred > vrp_pred.mean()) == (y_test_vrp > y_test_vrp.mean())).mean()
    
    # 트레이딩 성과
    signals = (vrp_pred > vrp_pred.mean()).astype(int)
    trade_returns = y_test_vrp[signals == 1]
    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
    
    return {
        'r2': float(r2),
        'direction_accuracy': float(direction_acc),
        'win_rate': float(win_rate),
        'n_samples': len(test_data),
        'avg_vix': float(test_data['VIX'].mean()),
        'avg_vrp': float(y_test_vrp.mean())
    }


def define_periods(spy):
    """기간 정의"""
    periods = {
        '전체 기간': (spy.index >= '2020-01-01'),
        '팬데믹 구간 (2020.03-2020.12)': (spy.index >= '2020-03-01') & (spy.index <= '2020-12-31'),
        '회복기 (2021-2022)': (spy.index >= '2021-01-01') & (spy.index <= '2022-12-31'),
        '정상화 (2023-2025)': (spy.index >= '2023-01-01'),
        '저변동성 (VIX<20)': spy['VIX'] < 20,
        '중변동성 (20≤VIX<25)': (spy['VIX'] >= 20) & (spy['VIX'] < 25),
        '고변동성 (VIX≥25)': spy['VIX'] >= 25
    }
    return periods


def main():
    print("\n" + "=" * 60)
    print("📊 기간 분리(Sub-period) 분석")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/3] 데이터 로드...")
    spy = load_and_prepare_data()
    print(f"     전체 샘플: {len(spy)} ({spy.index[0].strftime('%Y-%m-%d')} ~ {spy.index[-1].strftime('%Y-%m-%d')})")
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'regime_crisis', 'return_5d', 'return_22d']
    
    # 전체 데이터로 모델 학습 (80% 학습)
    split_idx = int(len(spy) * 0.8)
    train_data = spy.iloc[:split_idx]
    
    # 기간별 분석
    print("\n[2/3] 기간별 성능 분석...")
    print("-" * 60)
    
    periods = define_periods(spy)
    results = {}
    
    print(f"\n{'기간':<30} {'샘플':>8} {'R²':>10} {'방향 정확도':>12} {'승률':>10}")
    print("-" * 70)
    
    for period_name, mask in periods.items():
        test_subset = spy.loc[mask & (spy.index >= spy.iloc[split_idx].name)]
        
        if len(test_subset) < 10:
            print(f"{period_name:<30} {'N/A':>8} {'샘플 부족':>10}")
            results[period_name] = {'n_samples': len(test_subset), 'note': '샘플 부족'}
            continue
        
        result = train_and_evaluate(train_data, test_subset, feature_cols)
        results[period_name] = result
        
        print(f"{period_name:<30} {result['n_samples']:>8} {result['r2']:>10.4f} "
              f"{result['direction_accuracy']*100:>11.1f}% {result['win_rate']*100:>9.1f}%")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 분석 결과 요약")
    print("=" * 60)
    
    # 기간별 비교
    period_results = {k: v for k, v in results.items() if 'r2' in v}
    
    if period_results:
        best_period = max(period_results.items(), key=lambda x: x[1]['r2'])
        worst_period = min(period_results.items(), key=lambda x: x[1]['r2'])
        
        print(f"""
    🔹 기간별 R² 비교:
       • 최고 성능: {best_period[0]} (R² = {best_period[1]['r2']:.4f})
       • 최저 성능: {worst_period[0]} (R² = {worst_period[1]['r2']:.4f})
    
    💡 핵심 발견:
       • 팬데믹 제외 시에도 예측력 유지 여부 확인
       • VIX 수준별 성능 차이 분석
       • 저변동성 기간에서의 모델 효과성
    """)
    
    # 결과 저장
    output = {
        'period_analysis': results,
        'feature_columns': feature_cols,
        'train_period': {
            'start': train_data.index[0].strftime('%Y-%m-%d'),
            'end': train_data.index[-1].strftime('%Y-%m-%d'),
            'n_samples': len(train_data)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = Path('data/results/subperiod_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"💾 결과 저장: {output_path}")
    print("\n✅ 기간 분리 분석 완료!")


if __name__ == '__main__':
    main()
