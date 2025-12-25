"""
5일 예측 종합 실험
==================
기준: 5일 예측
실험:
1. 하이브리드 (AR(1) + 핵심 고급 특성)
2. 추가 자산 (IWM, XLE, FXI, EWZ)
3. 비선형 모델 (Gradient Boosting, XGBoost)
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import nnls
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

# ============================================================================
# 실험 1: 하이브리드 (AR(1) + 핵심 고급 특성)
# ============================================================================

def prepare_hybrid_features(ticker):
    """하이브리드: AR(1) 기반 + 핵심 특성만"""
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    if len(data) < 500:
        return None
    
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=data.index)
    
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    # AR(1) 핵심 (가장 중요)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    
    # 핵심 추가 특성 (6개만)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_ratio_lag1'] = (rv_5d / rv_22d.clip(lower=1e-8)).shift(1)
    
    # VIX
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(data.index).ffill().shift(1)
    
    # VIX 변화율
    features['VIX_change_lag1'] = features['VIX_lag1'].pct_change()
    
    # 방향성 (줌바흐 핵심)
    features['direction_5d_lag1'] = returns.rolling(5).apply(lambda x: np.mean(x > 0)).shift(1)
    
    # 타겟
    features['RV_5d_future'] = rv_5d.shift(-5)
    
    return features.dropna()

def experiment_hybrid(ticker):
    """하이브리드 모델 실험"""
    print(f"\n[Hybrid] {ticker}")
    
    features = prepare_hybrid_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
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
    
    # Ridge (No Log) - 다양한 alpha
    for alpha in [1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)
        pred = np.maximum(model.predict(X_test_s), 0)
        results[f'Ridge_a{alpha}'] = r2_score(y_test, pred)
    
    # Huber
    model = HuberRegressor(epsilon=1.35, alpha=10.0, max_iter=500)
    model.fit(X_train_s, y_train)
    pred = np.maximum(model.predict(X_test_s), 0)
    results['Huber'] = r2_score(y_test, pred)
    
    # Persistence
    results['Persistence'] = r2_score(y_test, X_test['RV_5d_lag1'].values)
    
    best = max([(k, v) for k, v in results.items() if k != 'Persistence'], key=lambda x: x[1])
    print(f"  Best: {best[0]} R² = {best[1]:.4f}")
    print(f"  Persistence: {results['Persistence']:.4f}")
    
    return {'asset': ticker, 'best_model': best[0], 'best_r2': best[1], 'all': results}

# ============================================================================
# 실험 2: 추가 자산
# ============================================================================

def prepare_simple_features(ticker):
    """단순 특성 (검증된 최적 구조)"""
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    if len(data) < 500:
        return None
    
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=data.index)
    
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_ratio_lag1'] = (rv_5d / rv_22d.clip(lower=1e-8)).shift(1)
    
    features['RV_5d_future'] = rv_5d.shift(-5)
    
    return features.dropna()

def experiment_new_assets():
    """새로운 자산 실험"""
    print("\n" + "="*70)
    print("[Experiment 2] New Assets")
    print("="*70)
    
    assets = ['IWM', 'XLE', 'FXI', 'EWZ', 'XLF', 'XLK']
    
    all_results = {}
    
    for ticker in assets:
        print(f"\n  {ticker}:")
        features = prepare_simple_features(ticker)
        if features is None:
            print(f"    No data available")
            continue
        
        feature_cols = [c for c in features.columns if c != 'RV_5d_future']
        X = features[feature_cols]
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
        
        # Ridge α=100 (검증된 최적)
        model = Ridge(alpha=100.0)
        model.fit(X_train_s, y_train)
        pred = np.maximum(model.predict(X_test_s), 0)
        r2 = r2_score(y_test, pred)
        
        # AR(1)
        X_train_ar = X_train[['RV_5d_lag1']].values
        X_test_ar = X_test[['RV_5d_lag1']].values
        
        mean_x = X_train_ar.mean()
        mean_y = y_train.mean()
        beta = np.sum((X_train_ar.flatten() - mean_x) * (y_train.values - mean_y)) / np.sum((X_train_ar.flatten() - mean_x) ** 2)
        alpha_ar = mean_y - beta * mean_x
        ar1_pred = alpha_ar + beta * X_test_ar.flatten()
        ar1_r2 = r2_score(y_test, ar1_pred)
        
        # Persistence
        persist_r2 = r2_score(y_test, X_test['RV_5d_lag1'].values)
        
        best_r2 = max(r2, ar1_r2)
        best_model = 'Ridge' if r2 > ar1_r2 else 'AR(1)'
        
        all_results[ticker] = {
            'ridge_r2': r2,
            'ar1_r2': ar1_r2,
            'best_r2': best_r2,
            'best_model': best_model,
            'persistence': persist_r2
        }
        
        print(f"    Ridge R²: {r2:.4f}, AR(1) R²: {ar1_r2:.4f}")
        print(f"    Best: {best_model} ({best_r2:.4f})")
    
    return all_results

# ============================================================================
# 실험 3: 비선형 모델
# ============================================================================

def experiment_nonlinear(ticker):
    """비선형 모델 실험"""
    print(f"\n[Non-linear] {ticker}")
    
    features = prepare_hybrid_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
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
    
    # Gradient Boosting (다양한 설정)
    for depth in [2, 3]:
        for n_est in [50, 100]:
            model = GradientBoostingRegressor(
                n_estimators=n_est, 
                max_depth=depth, 
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=42
            )
            model.fit(X_train_s, y_train)
            pred = np.maximum(model.predict(X_test_s), 0)
            results[f'GB_d{depth}_n{n_est}'] = r2_score(y_test, pred)
    
    # Random Forest
    for depth in [3, 4]:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=depth,
            min_samples_leaf=30,
            random_state=42
        )
        model.fit(X_train_s, y_train)
        pred = np.maximum(model.predict(X_test_s), 0)
        results[f'RF_d{depth}'] = r2_score(y_test, pred)
    
    # XGBoost 스타일 (sklearn GB with higher regularization)
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.01,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_s, y_train)
    pred = np.maximum(model.predict(X_test_s), 0)
    results['GB_regularized'] = r2_score(y_test, pred)
    
    # 비교: Ridge (baseline)
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, y_train)
    pred = np.maximum(model.predict(X_test_s), 0)
    results['Ridge_baseline'] = r2_score(y_test, pred)
    
    best = max(results.items(), key=lambda x: x[1])
    print(f"  Best: {best[0]} R² = {best[1]:.4f}")
    
    return {'asset': ticker, 'best_model': best[0], 'best_r2': best[1], 'all': results}

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("5-Day Comprehensive Experiments")
    print("="*80)
    print("Baseline: 5-day prediction")
    
    all_results = {}
    
    # ========================================
    # 실험 1: 하이브리드
    # ========================================
    print("\n" + "="*70)
    print("[Experiment 1] Hybrid (AR(1) + Key Features)")
    print("="*70)
    
    all_results['hybrid'] = {}
    for ticker in ['SPY', 'QQQ', 'EEM', 'GLD', 'TLT']:
        result = experiment_hybrid(ticker)
        if result:
            all_results['hybrid'][ticker] = result
    
    # ========================================
    # 실험 2: 추가 자산
    # ========================================
    all_results['new_assets'] = experiment_new_assets()
    
    # ========================================
    # 실험 3: 비선형 모델
    # ========================================
    print("\n" + "="*70)
    print("[Experiment 3] Non-linear Models")
    print("="*70)
    
    all_results['nonlinear'] = {}
    for ticker in ['SPY', 'QQQ', 'EEM', 'GLD', 'TLT']:
        result = experiment_nonlinear(ticker)
        if result:
            all_results['nonlinear'][ticker] = result
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': '5-Day Comprehensive Experiments',
            'baseline': '5-day prediction',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/5day_comprehensive_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    print("\n[1] Hybrid Results:")
    for ticker, result in all_results['hybrid'].items():
        print(f"    {ticker}: {result['best_r2']:.4f} ({result['best_model']})")
    
    print("\n[2] New Assets Results:")
    for ticker, result in all_results['new_assets'].items():
        print(f"    {ticker}: {result['best_r2']:.4f} ({result['best_model']})")
    
    print("\n[3] Non-linear Results:")
    for ticker, result in all_results['nonlinear'].items():
        print(f"    {ticker}: {result['best_r2']:.4f} ({result['best_model']})")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
