#!/usr/bin/env python3
"""
통계적 강건성 검증 실험
======================

1. Bootstrap 신뢰구간
2. Regime별 성능 분석
3. Rolling Window 검증
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime
from scipy import stats

SEED = 42
np.random.seed(SEED)


def create_features():
    """특성 생성 (기존과 동일)"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start=spy.index[0], end=spy.index[-1], progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    
    spy['returns'] = spy['Close'].pct_change()
    
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
        spy[f'realized_vol_{window}'] = spy[f'volatility_{window}'] * np.sqrt(252)
    
    for window in [5, 10, 20]:
        spy[f'mean_return_{window}'] = spy['returns'].rolling(window).mean()
        spy[f'skew_{window}'] = spy['returns'].rolling(window).skew()
        spy[f'kurt_{window}'] = spy['returns'].rolling(window).kurt()
    
    for lag in [1, 2, 3, 5]:
        spy[f'return_lag_{lag}'] = spy['returns'].shift(lag)
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    for window in [5, 10, 20]:
        spy[f'momentum_{window}'] = spy['returns'].rolling(window).sum()
    
    spy['vol_ratio_5_20'] = spy['volatility_5'] / (spy['volatility_20'] + 1e-8)
    spy['vol_ratio_10_50'] = spy['volatility_10'] / (spy['volatility_50'] + 1e-8)
    spy['zscore_20'] = (spy['returns'] - spy['returns'].rolling(20).mean()) / (spy['returns'].rolling(20).std() + 1e-8)
    
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_lag_5'] = spy['VIX'].shift(5)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['regime_crisis'] = (vix_lagged >= 35).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vol_in_crisis'] = spy['regime_crisis'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    spy['vix_excess_35'] = np.maximum(vix_lagged - 35, 0)
    
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            vol_values.append(pd.Series(returns[i+1:i+6]).std())
        else:
            vol_values.append(np.nan)
    spy['target_vol_5d'] = vol_values
    
    spy = spy.dropna()
    
    feature_cols = [c for c in spy.columns if c.startswith((
        'volatility_', 'realized_vol_', 'mean_return_', 'skew_', 'kurt_',
        'return_lag_', 'vol_lag_', 'vol_ratio_', 'zscore_', 'momentum_',
        'vix_', 'regime_', 'vol_in_', 'vix_excess_'
    ))]
    
    return spy, feature_cols


def bootstrap_confidence_interval(spy, feature_cols, n_bootstrap=1000):
    """1. Bootstrap 신뢰구간"""
    print("\n" + "=" * 60)
    print("[1/3] Bootstrap 신뢰구간 (n=1000)")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['target_vol_5d'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 최적 모델
    model = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    original_r2 = r2_score(y_test, y_pred)
    original_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Bootstrap
    r2_scores = []
    rmse_scores = []
    n_test = len(y_test)
    
    for i in range(n_bootstrap):
        idx = np.random.choice(n_test, size=n_test, replace=True)
        r2 = r2_score(y_test[idx], y_pred[idx])
        rmse = np.sqrt(mean_squared_error(y_test[idx], y_pred[idx]))
        r2_scores.append(r2)
        rmse_scores.append(rmse)
    
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    r2_ci_lower = np.percentile(r2_scores, 2.5)
    r2_ci_upper = np.percentile(r2_scores, 97.5)
    
    rmse_mean = np.mean(rmse_scores)
    rmse_ci_lower = np.percentile(rmse_scores, 2.5)
    rmse_ci_upper = np.percentile(rmse_scores, 97.5)
    
    print(f"\n  📊 R² 통계:")
    print(f"     Point estimate: {original_r2:.4f}")
    print(f"     Bootstrap mean: {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"     95% CI: [{r2_ci_lower:.4f}, {r2_ci_upper:.4f}]")
    
    print(f"\n  📊 RMSE 통계:")
    print(f"     Point estimate: {original_rmse:.6f}")
    print(f"     Bootstrap mean: {rmse_mean:.6f}")
    print(f"     95% CI: [{rmse_ci_lower:.6f}, {rmse_ci_upper:.6f}]")
    
    return {
        'r2': {
            'point_estimate': float(original_r2),
            'bootstrap_mean': float(r2_mean),
            'bootstrap_std': float(r2_std),
            'ci_lower': float(r2_ci_lower),
            'ci_upper': float(r2_ci_upper)
        },
        'rmse': {
            'point_estimate': float(original_rmse),
            'bootstrap_mean': float(rmse_mean),
            'ci_lower': float(rmse_ci_lower),
            'ci_upper': float(rmse_ci_upper)
        }
    }


def regime_analysis(spy, feature_cols):
    """2. Regime별 성능 분석"""
    print("\n" + "=" * 60)
    print("[2/3] Regime별 성능 분석")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['target_vol_5d'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 테스트 기간 VIX
    test_vix = spy['VIX'].iloc[split_idx:].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    # Regime 분류
    low_vol_mask = test_vix < 20
    normal_mask = (test_vix >= 20) & (test_vix < 25)
    high_vol_mask = (test_vix >= 25) & (test_vix < 35)
    crisis_mask = test_vix >= 35
    
    regimes = {
        'Low Vol (VIX<20)': low_vol_mask,
        'Normal (20≤VIX<25)': normal_mask,
        'High Vol (25≤VIX<35)': high_vol_mask,
        'Crisis (VIX≥35)': crisis_mask
    }
    
    regime_results = {}
    
    print("\n  📊 Regime별 성능:")
    print(f"     {'Regime':25s} | {'샘플':6s} | {'R²':8s} | {'RMSE':10s} | {'MAE':10s}")
    print("     " + "-" * 70)
    
    for regime_name, mask in regimes.items():
        n_samples = mask.sum()
        if n_samples >= 10:
            r2 = r2_score(y_test[mask], y_pred[mask])
            rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            mae = mean_absolute_error(y_test[mask], y_pred[mask])
            regime_results[regime_name] = {
                'n_samples': int(n_samples),
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae)
            }
            print(f"     {regime_name:25s} | {n_samples:6d} | {r2:8.4f} | {rmse:10.6f} | {mae:10.6f}")
        else:
            regime_results[regime_name] = {'n_samples': int(n_samples), 'r2': None, 'rmse': None, 'mae': None}
            print(f"     {regime_name:25s} | {n_samples:6d} | {'N/A':8s} | {'N/A':10s} | {'N/A':10s}")
    
    # 전체
    r2_all = r2_score(y_test, y_pred)
    rmse_all = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_all = mean_absolute_error(y_test, y_pred)
    print("     " + "-" * 70)
    print(f"     {'전체':25s} | {len(y_test):6d} | {r2_all:8.4f} | {rmse_all:10.6f} | {mae_all:10.6f}")
    
    return regime_results


def rolling_window_validation(spy, feature_cols, window_size=252, step_size=63):
    """3. Rolling Window 검증"""
    print("\n" + "=" * 60)
    print(f"[3/3] Rolling Window 검증 (window={window_size}일, step={step_size}일)")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['target_vol_5d'].values
    dates = spy.index
    
    scaler = StandardScaler()
    
    results = []
    min_train_size = 500  # 최소 학습 데이터
    
    start_idx = min_train_size
    
    while start_idx + window_size <= len(X):
        train_end = start_idx
        test_start = start_idx
        test_end = min(start_idx + step_size, len(X))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({
            'period_start': str(dates[test_start].date()),
            'period_end': str(dates[test_end-1].date()),
            'train_samples': train_end,
            'test_samples': test_end - test_start,
            'r2': float(r2),
            'rmse': float(rmse)
        })
        
        start_idx += step_size
    
    # 결과 요약
    r2_values = [r['r2'] for r in results]
    
    print(f"\n  📊 Rolling Window 결과 ({len(results)}개 윈도우):")
    print(f"     R² 평균: {np.mean(r2_values):.4f}")
    print(f"     R² 표준편차: {np.std(r2_values):.4f}")
    print(f"     R² 최소: {np.min(r2_values):.4f}")
    print(f"     R² 최대: {np.max(r2_values):.4f}")
    
    print(f"\n  📊 기간별 상세:")
    print(f"     {'Period':25s} | {'Train':6s} | {'Test':5s} | {'R²':8s}")
    print("     " + "-" * 55)
    for r in results:
        print(f"     {r['period_start']} ~ {r['period_end'][:4]} | {r['train_samples']:6d} | {r['test_samples']:5d} | {r['r2']:8.4f}")
    
    return {
        'windows': results,
        'summary': {
            'mean_r2': float(np.mean(r2_values)),
            'std_r2': float(np.std(r2_values)),
            'min_r2': float(np.min(r2_values)),
            'max_r2': float(np.max(r2_values)),
            'n_windows': len(results)
        }
    }


def main():
    print("\n" + "🔬" * 30)
    print("통계적 강건성 검증 실험")
    print("🔬" * 30)
    
    # 데이터 준비
    print("\n데이터 준비 중...")
    spy, feature_cols = create_features()
    print(f"  ✓ 데이터: {len(spy)} 행, {len(feature_cols)} 특성")
    
    # 1. Bootstrap
    bootstrap_results = bootstrap_confidence_interval(spy, feature_cols)
    
    # 2. Regime 분석
    regime_results = regime_analysis(spy, feature_cols)
    
    # 3. Rolling Window
    rolling_results = rolling_window_validation(spy, feature_cols)
    
    # 결과 저장
    output = {
        'bootstrap': bootstrap_results,
        'regime_analysis': regime_results,
        'rolling_window': rolling_results,
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = Path('paper/robustness_validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 결과 저장: {output_path}")
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("📊 최종 요약")
    print("=" * 60)
    
    print(f"""
    🎯 모델: ElasticNet (alpha=0.0003, l1_ratio=0.6)
    
    📈 성능:
       • R² = {bootstrap_results['r2']['point_estimate']:.4f}
       • 95% CI: [{bootstrap_results['r2']['ci_lower']:.4f}, {bootstrap_results['r2']['ci_upper']:.4f}]
    
    🔄 Rolling Window 안정성:
       • 평균 R²: {rolling_results['summary']['mean_r2']:.4f} ± {rolling_results['summary']['std_r2']:.4f}
    
    📉 Regime별 성능:
       • 저변동성(VIX<20): R² = {regime_results.get('Low Vol (VIX<20)', {}).get('r2', 'N/A')}
       • 고변동성(VIX≥25): 샘플 부족으로 제한적
    """)


if __name__ == '__main__':
    main()
