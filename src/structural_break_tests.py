#!/usr/bin/env python3
"""
구조적 변화(Structural Break) 검정
===================================

1. Chow Test: 구조적 변화 시점 검정
2. 롤링 윈도우 회귀: 계수 안정성 분석
3. 재귀적(Recursive) 잔차 분석

실행: python src/structural_break_tests.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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
    
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    
    spy = spy.replace([np.inf, -np.inf], np.nan).dropna()
    
    return spy


def chow_test(X, y, break_point):
    """
    Chow Test for Structural Break
    
    H0: 구조적 변화 없음 (계수가 동일)
    H1: 구조적 변화 있음 (계수가 다름)
    
    Parameters:
    -----------
    X: 독립변수 (n x k)
    y: 종속변수 (n,)
    break_point: 변화 시점 인덱스
    
    Returns:
    --------
    f_stat: F-통계량
    p_value: p-value
    """
    n = len(y)
    k = X.shape[1]
    
    # 전체 모델
    model_full = LinearRegression()
    model_full.fit(X, y)
    rss_full = np.sum((y - model_full.predict(X)) ** 2)
    
    # 분할 모델 1 (변화점 이전)
    X1, y1 = X[:break_point], y[:break_point]
    n1 = len(y1)
    if n1 <= k:
        return np.nan, np.nan
    
    model1 = LinearRegression()
    model1.fit(X1, y1)
    rss1 = np.sum((y1 - model1.predict(X1)) ** 2)
    
    # 분할 모델 2 (변화점 이후)
    X2, y2 = X[break_point:], y[break_point:]
    n2 = len(y2)
    if n2 <= k:
        return np.nan, np.nan
    
    model2 = LinearRegression()
    model2.fit(X2, y2)
    rss2 = np.sum((y2 - model2.predict(X2)) ** 2)
    
    # RSS 합
    rss_pooled = rss1 + rss2
    
    # F-통계량
    f_stat = ((rss_full - rss_pooled) / k) / (rss_pooled / (n - 2 * k))
    
    # p-value
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
    
    return float(f_stat), float(p_value)


def rolling_coefficient_analysis(spy, feature_cols, window=252):
    """
    롤링 윈도우 회귀 분석
    
    시간에 따른 회귀 계수 변화 추적
    """
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    dates = spy.index
    
    scaler = StandardScaler()
    
    rolling_coefs = []
    rolling_dates = []
    rolling_r2 = []
    
    for i in range(window, len(spy)):
        X_window = X[i-window:i]
        y_window = y[i-window:i]
        
        X_scaled = scaler.fit_transform(X_window)
        
        model = LinearRegression()
        model.fit(X_scaled, y_window)
        
        rolling_coefs.append(model.coef_.tolist())
        rolling_dates.append(dates[i].strftime('%Y-%m-%d'))
        rolling_r2.append(model.score(X_scaled, y_window))
    
    return {
        'dates': rolling_dates,
        'coefficients': rolling_coefs,
        'r2': rolling_r2,
        'feature_names': feature_cols
    }


def coefficient_stability_test(rolling_coefs, feature_names):
    """계수 안정성 분석"""
    coefs_array = np.array(rolling_coefs)
    
    stability = {}
    for i, name in enumerate(feature_names):
        coef_series = coefs_array[:, i]
        stability[name] = {
            'mean': float(np.mean(coef_series)),
            'std': float(np.std(coef_series)),
            'cv': float(np.std(coef_series) / (np.abs(np.mean(coef_series)) + 1e-8)),  # 변동계수
            'min': float(np.min(coef_series)),
            'max': float(np.max(coef_series)),
            'sign_changes': int(np.sum(np.diff(np.sign(coef_series)) != 0))
        }
    
    return stability


def main():
    print("\n" + "=" * 60)
    print("🔍 구조적 변화(Structural Break) 검정")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    spy = load_and_prepare_data()
    print(f"     전체 샘플: {len(spy)} ({spy.index[0].strftime('%Y-%m-%d')} ~ {spy.index[-1].strftime('%Y-%m-%d')})")
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d']  # HAR-RV 기본 특성
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    
    # Chow Test (다양한 시점)
    print("\n[2/4] Chow Test (구조적 변화 검정)...")
    print("-" * 60)
    
    break_points = {
        '2020-03-01 (COVID 시작)': spy.index.get_indexer([pd.Timestamp('2020-03-01')], method='nearest')[0],
        '2020-12-01 (COVID 1차 종료)': spy.index.get_indexer([pd.Timestamp('2020-12-01')], method='nearest')[0],
        '2021-06-01 (회복기)': spy.index.get_indexer([pd.Timestamp('2021-06-01')], method='nearest')[0],
        '2022-01-01 (금리인상 시작)': spy.index.get_indexer([pd.Timestamp('2022-01-01')], method='nearest')[0],
        '2023-01-01 (정상화)': spy.index.get_indexer([pd.Timestamp('2023-01-01')], method='nearest')[0],
    }
    
    chow_results = {}
    
    print(f"\n{'변화 시점':<30} {'F-통계량':>12} {'p-value':>12} {'결과':>15}")
    print("-" * 70)
    
    for name, bp in break_points.items():
        f_stat, p_value = chow_test(X, y, bp)
        
        if np.isnan(f_stat):
            print(f"{name:<30} {'N/A':>12}")
            continue
        
        significance = "구조적 변화 있음" if p_value < 0.05 else "변화 없음"
        stars = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
        
        chow_results[name] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        print(f"{name:<30} {f_stat:>12.3f} {p_value:>12.4f} {significance:>15} {stars}")
    
    # 롤링 계수 분석
    print("\n[3/4] 롤링 윈도우 회귀 분석 (window=252)...")
    rolling_result = rolling_coefficient_analysis(spy, feature_cols, window=252)
    print(f"     분석 기간: {rolling_result['dates'][0]} ~ {rolling_result['dates'][-1]}")
    print(f"     R² 범위: {min(rolling_result['r2']):.4f} ~ {max(rolling_result['r2']):.4f}")
    
    # 계수 안정성 분석
    print("\n[4/4] 계수 안정성 분석...")
    stability = coefficient_stability_test(rolling_result['coefficients'], feature_cols)
    
    print(f"\n{'변수':<15} {'평균':>10} {'표준편차':>10} {'변동계수':>10} {'부호변경':>10}")
    print("-" * 55)
    for name, stats_dict in stability.items():
        print(f"{name:<15} {stats_dict['mean']:>10.3f} {stats_dict['std']:>10.3f} "
              f"{stats_dict['cv']:>10.3f} {stats_dict['sign_changes']:>10}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 검정 결과 요약")
    print("=" * 60)
    
    sig_breaks = [k for k, v in chow_results.items() if v.get('significant', False)]
    
    print(f"""
    🔹 Chow Test 결과:
       • 유의한 구조적 변화: {len(sig_breaks)}개 시점
       {'• ' + ', '.join(sig_breaks) if sig_breaks else '• 유의한 변화 없음'}
    
    🔹 롤링 R² 분석:
       • 평균 R²: {np.mean(rolling_result['r2']):.4f}
       • R² 표준편차: {np.std(rolling_result['r2']):.4f}
       • 모델 예측력이 시간에 따라 {'안정적' if np.std(rolling_result['r2']) < 0.1 else '변동적'}
    
    💡 핵심 발견:
       • {'구조적 변화가 감지되어 팬데믹의 영향이 있음' if sig_breaks else '구조적 변화가 없어 모델이 안정적'}
       • 롤링 회귀 계수가 {'안정적' if max(s['cv'] for s in stability.values()) < 1 else '변동적'}임
    """)
    
    # 결과 저장
    output = {
        'chow_test': chow_results,
        'rolling_analysis': {
            'window_size': 252,
            'n_windows': len(rolling_result['dates']),
            'r2_mean': float(np.mean(rolling_result['r2'])),
            'r2_std': float(np.std(rolling_result['r2'])),
            'r2_min': float(min(rolling_result['r2'])),
            'r2_max': float(max(rolling_result['r2']))
        },
        'coefficient_stability': stability,
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = Path('data/results/structural_breaks.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 결과 저장: {output_path}")
    print("\n✅ 구조적 변화 검정 완료!")


if __name__ == '__main__':
    main()
