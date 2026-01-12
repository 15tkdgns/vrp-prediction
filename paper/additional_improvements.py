#!/usr/bin/env python3
"""
추가 개선 실험
==============

1. 10년 데이터 (2014-2024)
2. 로그 변환 타겟
3. 변동성 방향 예측 (분류)
4. 상위 특성만 사용
5. 다른 예측 기간 (1일, 10일, 22일)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, f1_score
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def experiment_1_longer_data():
    """실험 1: 10년 데이터 (2014-2024)"""
    print("\n" + "=" * 60)
    print("[1/5] 10년 데이터 (2014-2024)")
    print("=" * 60)
    
    # 10년 데이터 다운로드
    print("  데이터 다운로드 중...")
    spy = yf.download('SPY', start='2014-01-01', end='2025-01-01', progress=False)
    vix = yf.download('^VIX', start='2014-01-01', end='2025-01-01', progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    
    # 특성 생성
    spy['returns'] = spy['Close'].pct_change()
    
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
    
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    
    for lag in [1, 2, 3, 5]:
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    # 타겟
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            vol_values.append(pd.Series(returns[i+1:i+6]).std())
        else:
            vol_values.append(np.nan)
    spy['target'] = vol_values
    
    spy = spy.dropna()
    
    feature_cols = ['volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
                   'vix_lag_1', 'vix_change', 'vix_zscore', 'regime_high_vol',
                   'vol_in_high_regime', 'vix_excess_25', 'vol_lag_1', 'vol_lag_2', 
                   'vol_lag_3', 'vol_lag_5']
    
    X = spy[feature_cols].values
    y = spy['target'].values
    
    # 80-20 분할
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    
    # Bootstrap 신뢰구간
    n_bootstrap = 500
    r2_scores = []
    for i in range(n_bootstrap):
        idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
        r2_scores.append(r2_score(y_test[idx], y_pred[idx]))
    
    ci_lower = np.percentile(r2_scores, 2.5)
    ci_upper = np.percentile(r2_scores, 97.5)
    
    print(f"\n  📊 결과:")
    print(f"     데이터: {len(spy)} 행 (10년)")
    print(f"     Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"     R² = {r2:.4f}")
    print(f"     95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"     CI 폭: {ci_upper - ci_lower:.4f}")
    
    return {
        'n_samples': len(spy),
        'r2': float(r2),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_width': float(ci_upper - ci_lower)
    }


def experiment_2_log_transform():
    """실험 2: 로그 변환 타겟"""
    print("\n" + "=" * 60)
    print("[2/5] 로그 변환 타겟")
    print("=" * 60)
    
    # 기존 데이터 사용
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start=spy.index[0], end=spy.index[-1], progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    
    spy['returns'] = spy['Close'].pct_change()
    
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
    
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    
    for lag in [1, 2, 3, 5]:
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    # 타겟
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            vol_values.append(pd.Series(returns[i+1:i+6]).std())
        else:
            vol_values.append(np.nan)
    spy['target'] = vol_values
    spy['target_log'] = np.log(spy['target'] + 1e-8)
    
    spy = spy.dropna()
    
    feature_cols = ['volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
                   'vix_lag_1', 'vix_change', 'vix_zscore', 'regime_high_vol',
                   'vol_in_high_regime', 'vix_excess_25', 'vol_lag_1', 'vol_lag_2', 
                   'vol_lag_3', 'vol_lag_5']
    
    X = spy[feature_cols].values
    y_original = spy['target'].values
    y_log = spy['target_log'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_orig, y_test_orig = y_original[:split_idx], y_original[split_idx:]
    y_train_log, y_test_log = y_log[:split_idx], y_log[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 원본 타겟
    model_orig = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
    model_orig.fit(X_train_s, y_train_orig)
    y_pred_orig = model_orig.predict(X_test_s)
    r2_orig = r2_score(y_test_orig, y_pred_orig)
    
    # 로그 타겟
    model_log = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
    model_log.fit(X_train_s, y_train_log)
    y_pred_log = model_log.predict(X_test_s)
    y_pred_exp = np.exp(y_pred_log) - 1e-8  # 역변환
    r2_log = r2_score(y_test_orig, y_pred_exp)
    
    print(f"\n  📊 결과:")
    print(f"     원본 타겟 R²:  {r2_orig:.4f}")
    print(f"     로그 타겟 R²:  {r2_log:.4f}")
    print(f"     개선:          {r2_log - r2_orig:+.4f}")
    
    return {
        'original_r2': float(r2_orig),
        'log_r2': float(r2_log),
        'improvement': float(r2_log - r2_orig)
    }


def experiment_3_direction_prediction():
    """실험 3: 변동성 방향 예측 (분류)"""
    print("\n" + "=" * 60)
    print("[3/5] 변동성 방향 예측 (분류)")
    print("=" * 60)
    
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start=spy.index[0], end=spy.index[-1], progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    
    spy['returns'] = spy['Close'].pct_change()
    
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
    
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    
    for lag in [1, 2, 3, 5]:
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    # 미래 변동성
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            vol_values.append(pd.Series(returns[i+1:i+6]).std())
        else:
            vol_values.append(np.nan)
    spy['future_vol'] = vol_values
    
    # 방향 타겟: 미래 변동성 > 현재 변동성
    spy['direction'] = (spy['future_vol'] > spy['volatility_5']).astype(int)
    
    spy = spy.dropna()
    
    feature_cols = ['volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
                   'vix_lag_1', 'vix_change', 'vix_zscore', 'regime_high_vol',
                   'vol_in_high_regime', 'vix_excess_25', 'vol_lag_1', 'vol_lag_2', 
                   'vol_lag_3', 'vol_lag_5']
    
    X = spy[feature_cols].values
    y = spy['direction'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(C=0.1, random_state=SEED, max_iter=10000)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())  # 다수 클래스 예측
    
    print(f"\n  📊 결과:")
    print(f"     정확도:        {accuracy:.4f}")
    print(f"     F1 Score:      {f1:.4f}")
    print(f"     기준선 (다수):  {baseline_acc:.4f}")
    print(f"     개선:          {accuracy - baseline_acc:+.4f}")
    
    return {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'baseline': float(baseline_acc),
        'improvement': float(accuracy - baseline_acc)
    }


def experiment_4_top_features():
    """실험 4: 상위 특성만 사용"""
    print("\n" + "=" * 60)
    print("[4/5] 상위 특성만 사용")
    print("=" * 60)
    
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start=spy.index[0], end=spy.index[-1], progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    
    spy['returns'] = spy['Close'].pct_change()
    
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
    
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    
    for lag in [1, 2, 3, 5]:
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            vol_values.append(pd.Series(returns[i+1:i+6]).std())
        else:
            vol_values.append(np.nan)
    spy['target'] = vol_values
    
    spy = spy.dropna()
    
    all_features = ['volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
                   'vix_lag_1', 'vix_change', 'vix_zscore', 'regime_high_vol',
                   'vol_in_high_regime', 'vix_excess_25', 'vol_lag_1', 'vol_lag_2', 
                   'vol_lag_3', 'vol_lag_5']
    
    # 상위 3개만 (vix_lag_1, volatility_20, volatility_5)
    top_3 = ['vix_lag_1', 'volatility_20', 'volatility_5']
    
    # 상위 5개
    top_5 = ['vix_lag_1', 'volatility_20', 'volatility_5', 'vix_change', 'regime_high_vol']
    
    y = spy['target'].values
    split_idx = int(len(spy) * 0.8)
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    results = {}
    
    for name, features in [('전체 (14)', all_features), ('상위 5', top_5), ('상위 3', top_3)]:
        X = spy[features].values
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = float(r2)
        print(f"     {name:15s}: R² = {r2:.4f}")
    
    return results


def experiment_5_different_horizons():
    """실험 5: 다른 예측 기간"""
    print("\n" + "=" * 60)
    print("[5/5] 다른 예측 기간 (1일, 5일, 10일, 22일)")
    print("=" * 60)
    
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start=spy.index[0], end=spy.index[-1], progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    
    spy['returns'] = spy['Close'].pct_change()
    
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
    
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    
    for lag in [1, 2, 3, 5]:
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    feature_cols = ['volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
                   'vix_lag_1', 'vix_change', 'vix_zscore', 'regime_high_vol',
                   'vol_in_high_regime', 'vix_excess_25', 'vol_lag_1', 'vol_lag_2', 
                   'vol_lag_3', 'vol_lag_5']
    
    results = {}
    
    for horizon in [1, 5, 10, 22]:
        # 각 horizon에 대한 타겟 생성
        vol_values = []
        returns = spy['returns'].values
        for i in range(len(returns)):
            if i + horizon < len(returns):
                vol_values.append(pd.Series(returns[i+1:i+1+horizon]).std())
            else:
                vol_values.append(np.nan)
        spy[f'target_{horizon}d'] = vol_values
        
        data = spy.dropna(subset=[f'target_{horizon}d'] + feature_cols)
        
        X = data[feature_cols].values
        y = data[f'target_{horizon}d'].values
        
        split_idx = int(len(data) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        
        results[f'{horizon}d'] = float(r2)
        print(f"     {horizon:2d}일 예측: R² = {r2:.4f}")
    
    return results


def main():
    print("\n" + "🔬" * 30)
    print("추가 개선 실험")
    print("🔬" * 30)
    
    # 실험 실행
    result_1 = experiment_1_longer_data()
    result_2 = experiment_2_log_transform()
    result_3 = experiment_3_direction_prediction()
    result_4 = experiment_4_top_features()
    result_5 = experiment_5_different_horizons()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 최종 요약")
    print("=" * 60)
    
    print(f"""
    기준 성능: R² = 0.2608
    
    1️⃣ 10년 데이터
       • R² = {result_1['r2']:.4f}
       • 95% CI: [{result_1['ci_lower']:.4f}, {result_1['ci_upper']:.4f}]
       • CI 폭: {result_1['ci_width']:.4f} (축소 효과 확인)
    
    2️⃣ 로그 변환 타겟
       • 원본: R² = {result_2['original_r2']:.4f}
       • 로그: R² = {result_2['log_r2']:.4f}
       • 개선: {result_2['improvement']:+.4f}
    
    3️⃣ 변동성 방향 예측 (분류)
       • 정확도: {result_3['accuracy']:.4f}
       • 기준선 대비: {result_3['improvement']:+.4f}
    
    4️⃣ 상위 특성만 사용
       • 전체: R² = {result_4.get('전체 (14)', 0):.4f}
       • 상위 5: R² = {result_4.get('상위 5', 0):.4f}
       • 상위 3: R² = {result_4.get('상위 3', 0):.4f}
    
    5️⃣ 예측 기간별 성능
       • 1일: R² = {result_5.get('1d', 0):.4f}
       • 5일: R² = {result_5.get('5d', 0):.4f}
       • 10일: R² = {result_5.get('10d', 0):.4f}
       • 22일: R² = {result_5.get('22d', 0):.4f}
    """)
    
    # 저장
    output = {
        'longer_data': result_1,
        'log_transform': result_2,
        'direction_prediction': result_3,
        'top_features': result_4,
        'different_horizons': result_5,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/additional_improvements_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 결과 저장: paper/additional_improvements_results.json")


if __name__ == '__main__':
    main()
