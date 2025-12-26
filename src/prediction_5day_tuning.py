"""
5일 예측 튜닝 실험
==================
문제: 5일 예측 R² = -0.20 (과적합)
목표: 양수 R² 달성

전략:
1. 극강 정규화 (alpha=10, 100)
2. 특성 축소 (핵심 3개만)
3. Walk-Forward CV
4. 단순 모델 (앙상블 없이)
5. Naive 벤치마크 비교
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Lasso, Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_minimal_features_5d(ticker):
    """최소 특성 (과적합 방지)"""
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=data.index)
    
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    # 최소 특성 (3개만)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_ratio_lag1'] = (rv_5d / rv_22d.clip(lower=1e-8)).shift(1)
    
    # 타겟
    features['RV_5d_future'] = rv_5d.shift(-5)
    
    return features.dropna()

# ============================================================================
# 실험 1: 극강 정규화로 단순 모델
# ============================================================================

def experiment_extreme_regularization(ticker):
    """극강 정규화 실험"""
    print(f"\n[Exp 1] Extreme Regularization: {ticker}")
    
    features = prepare_minimal_features_5d(ticker)
    if features is None or len(features) < 500:
        return None
    
    X = features[['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1']]
    y = features['RV_5d_future']
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    y_train_log = np.log(y_train + 1)
    
    results = {}
    
    # Naive 벤치마크
    naive_pred = np.full(len(y_test), X_test['RV_5d_lag1'].mean())
    results['naive'] = r2_score(y_test, naive_pred)
    
    # Persistence (어제 RV로 예측)
    persist_pred = X_test['RV_5d_lag1'].values
    results['persistence'] = r2_score(y_test, persist_pred)
    
    # 다양한 정규화 강도
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train_log)
        pred = np.exp(model.predict(X_test_s)) - 1
        results[f'Ridge_a{alpha}'] = r2_score(y_test, pred)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    for name, r2 in results.items():
        print(f"    {name}: R2={r2:.4f}")
    
    best = max(results.items(), key=lambda x: x[1])
    return {'asset': ticker, 'best_model': best[0], 'best_r2': best[1], 'all_results': results}

# ============================================================================
# 실험 2: Walk-Forward CV
# ============================================================================

def experiment_walk_forward_5d(ticker):
    """Walk-Forward CV"""
    print(f"\n[Exp 2] Walk-Forward CV: {ticker}")
    
    features = prepare_minimal_features_5d(ticker)
    if features is None or len(features) < 500:
        return None
    
    X = features[['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1']]
    y = features['RV_5d_future']
    
    tscv = TimeSeriesSplit(n_splits=5)
    gap = 5
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        train_idx = train_idx[:-gap]
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        if len(X_train) < 200 or len(X_test) < 50:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        y_train_log = np.log(y_train + 1)
        
        model = Ridge(alpha=100.0)  # 강한 정규화
        model.fit(X_train_s, y_train_log)
        
        pred = np.exp(model.predict(X_test_s)) - 1
        r2 = r2_score(y_test, pred)
        
        # Persistence 비교
        persist_r2 = r2_score(y_test, X_test['RV_5d_lag1'].values)
        
        fold_results.append({
            'fold': fold,
            'r2': r2,
            'persistence_r2': persist_r2,
            'improvement': r2 - persist_r2
        })
        
        print(f"    Fold {fold}: Model R2={r2:.4f}, Persistence R2={persist_r2:.4f}")
    
    if fold_results:
        avg_r2 = np.mean([f['r2'] for f in fold_results])
        avg_persist = np.mean([f['persistence_r2'] for f in fold_results])
        print(f"  Avg: Model R2={avg_r2:.4f}, Persistence R2={avg_persist:.4f}")
        return {'asset': ticker, 'avg_r2': avg_r2, 'avg_persistence': avg_persist, 'folds': fold_results}
    
    return None

# ============================================================================
# 실험 3: AR(1) 스타일 단순 예측
# ============================================================================

def experiment_ar1_style(ticker):
    """AR(1) 스타일: RV_t+1 = a + b * RV_t"""
    print(f"\n[Exp 3] AR(1) Style: {ticker}")
    
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_5d = calculate_rv(returns, 5)
    
    df = pd.DataFrame(index=data.index)
    df['RV_lag1'] = rv_5d.shift(1)
    df['RV_future'] = rv_5d.shift(-5)
    df = df.dropna()
    
    # 단순 분할
    n = len(df)
    train_end = int(n * 0.7)
    
    train = df.iloc[:train_end]
    test = df.iloc[train_end:]
    
    # 단순 선형 회귀 (수동)
    X_train = train['RV_lag1'].values
    y_train = train['RV_future'].values
    
    # OLS coefficients
    mean_x = X_train.mean()
    mean_y = y_train.mean()
    
    beta = np.sum((X_train - mean_x) * (y_train - mean_y)) / np.sum((X_train - mean_x) ** 2)
    alpha = mean_y - beta * mean_x
    
    # 예측
    X_test = test['RV_lag1'].values
    y_test = test['RV_future'].values
    
    pred = alpha + beta * X_test
    r2 = r2_score(y_test, pred)
    
    # Persistence
    persist_r2 = r2_score(y_test, X_test)
    
    print(f"  AR(1): RV_future = {alpha:.4f} + {beta:.4f} * RV_lag1")
    print(f"  Test R2: {r2:.4f}")
    print(f"  Persistence R2: {persist_r2:.4f}")
    
    return {'asset': ticker, 'alpha': alpha, 'beta': beta, 'r2': r2, 'persistence_r2': persist_r2}

# ============================================================================
# 실험 4: 로그 변환 없이
# ============================================================================

def experiment_no_log_transform(ticker):
    """로그 변환 없이 직접 예측"""
    print(f"\n[Exp 4] No Log Transform: {ticker}")
    
    features = prepare_minimal_features_5d(ticker)
    if features is None or len(features) < 500:
        return None
    
    X = features[['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1']]
    y = features['RV_5d_future']
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    
    # 로그 변환 없이 Ridge
    for alpha in [1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)  # 로그 변환 없음
        pred = model.predict(X_test_s)
        pred = np.maximum(pred, 0)
        results[f'Ridge_a{alpha}_noLog'] = r2_score(y_test, pred)
    
    # Huber (이상치 제거)
    model = HuberRegressor(epsilon=1.35, alpha=10.0, max_iter=500)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    pred = np.maximum(pred, 0)
    results['Huber_noLog'] = r2_score(y_test, pred)
    
    for name, r2 in results.items():
        print(f"    {name}: R2={r2:.4f}")
    
    best = max(results.items(), key=lambda x: x[1])
    return {'asset': ticker, 'best_model': best[0], 'best_r2': best[1], 'all_results': results}

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("5-Day Prediction Tuning Experiment")
    print("="*80)
    
    assets = ['SPY', 'QQQ', 'EEM', 'GLD', 'TLT']
    
    all_results = {}
    
    for asset in assets:
        all_results[asset] = {}
        
        # 실험 1: 극강 정규화
        r1 = experiment_extreme_regularization(asset)
        if r1:
            all_results[asset]['extreme_reg'] = r1
        
        # 실험 2: Walk-Forward CV
        r2 = experiment_walk_forward_5d(asset)
        if r2:
            all_results[asset]['walk_forward'] = r2
        
        # 실험 3: AR(1) 스타일
        r3 = experiment_ar1_style(asset)
        if r3:
            all_results[asset]['ar1'] = r3
        
        # 실험 4: 로그 변환 없이
        r4 = experiment_no_log_transform(asset)
        if r4:
            all_results[asset]['no_log'] = r4
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': '5-Day Prediction Tuning',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/5day_tuning_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    for asset in assets:
        if asset in all_results:
            results = all_results[asset]
            
            best_r2 = -999
            best_method = None
            
            if 'extreme_reg' in results and results['extreme_reg']:
                r2 = results['extreme_reg']['best_r2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_method = f"Extreme Reg ({results['extreme_reg']['best_model']})"
            
            if 'ar1' in results and results['ar1']:
                r2 = results['ar1']['r2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_method = "AR(1)"
            
            if 'no_log' in results and results['no_log']:
                r2 = results['no_log']['best_r2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_method = f"No Log ({results['no_log']['best_model']})"
            
            print(f"  {asset}: Best R2={best_r2:.4f} ({best_method})")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
