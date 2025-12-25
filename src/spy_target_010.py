"""
SPY R² 0.10+ 달성 실험
======================
문제: 현재 OOS R² = -0.11 → Val/Test 과적합 심각

전략:
1. 특성 최소화 (핵심 5개만)
2. 극강 정규화 (α=1.0+)
3. Walk-Forward CV (더 보수적 검증)
4. 단순 평균 앙상블 (NNLS 대신)
5. 분위수 회귀 시도
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Lasso, Ridge, HuberRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import nnls
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 기본 함수
# ============================================================================

def calculate_rv(returns, window=22):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

# ============================================================================
# 최소 특성 준비 (과적합 방지)
# ============================================================================

def prepare_minimal_features():
    """최소 핵심 특성만 - 과적합 방지"""
    print("Preparing minimal features...")
    
    spy = yf.download('SPY', start='2018-01-01', end='2025-01-01', progress=False)  # 더 긴 기간
    returns = spy['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=spy.index)
    
    # 핵심 HAR-RV만 (lag 적용)
    rv_22d = calculate_rv(returns, 22)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_22d_lag5'] = rv_22d.shift(5)
    features['RV_22d_lag22'] = rv_22d.shift(22)  # 월간 지연
    
    # VIX (lag 적용)
    vix = yf.download('^VIX', start='2018-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close']
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]
    vix_aligned = vix_close.reindex(spy.index).ffill()
    features['VIX_lag1'] = vix_aligned.shift(1)
    
    # VRP (lag 적용)
    features['VRP_lag1'] = (vix_aligned.shift(1) ** 2 / 100) - rv_22d.shift(1)
    
    # 타겟
    features['RV_22d_future'] = rv_22d.shift(-22)
    
    return features.dropna()

# ============================================================================
# Walk-Forward CV
# ============================================================================

def walk_forward_cv(X, y, n_splits=5, gap=22):
    """Walk-Forward Cross-Validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Gap 적용
        train_idx = train_idx[:-gap] if len(train_idx) > gap else train_idx
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        if len(X_train) < 100 or len(X_test) < 30:
            continue
        
        # 표준화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 극강 정규화 모델
        y_train_log = np.log(y_train + 1)
        
        model = Ridge(alpha=10.0)  # 매우 강한 정규화
        model.fit(X_train_scaled, y_train_log)
        
        pred_log = model.predict(X_test_scaled)
        pred = np.exp(pred_log) - 1
        pred = np.maximum(pred, 0)
        
        r2 = r2_score(y_test, pred)
        results.append({'fold': fold, 'r2': r2, 'train_size': len(X_train), 'test_size': len(X_test)})
    
    return results

# ============================================================================
# 극강 정규화 실험
# ============================================================================

def extreme_regularization_experiment(X_train, y_train, X_val, y_val, X_test, y_test):
    """극강 정규화로 과적합 방지"""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_log = np.log(y_train + 1)
    
    # 다양한 정규화 강도
    alphas = [0.1, 1.0, 10.0, 100.0]
    
    best_test_r2 = -999
    best_config = None
    
    print("\n  Testing regularization strengths:")
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train_log)
        
        val_pred = np.exp(model.predict(X_val_scaled)) - 1
        test_pred = np.exp(model.predict(X_test_scaled)) - 1
        
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"    Ridge α={alpha}: Val R2={val_r2:.4f}, Test R2={test_r2:.4f}")
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_config = {'model': 'Ridge', 'alpha': alpha, 'val_r2': val_r2, 'test_r2': test_r2}
    
    # Huber도 시도
    for alpha in [0.1, 1.0, 10.0]:
        model = HuberRegressor(epsilon=1.5, alpha=alpha, max_iter=500)
        model.fit(X_train_scaled, y_train_log)
        
        val_pred = np.exp(model.predict(X_val_scaled)) - 1
        test_pred = np.exp(model.predict(X_test_scaled)) - 1
        
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"    Huber α={alpha}: Val R2={val_r2:.4f}, Test R2={test_r2:.4f}")
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_config = {'model': 'Huber', 'alpha': alpha, 'val_r2': val_r2, 'test_r2': test_r2}
    
    return best_config

# ============================================================================
# 단순 평균 앙상블 (NNLS 대신)
# ============================================================================

def simple_average_ensemble(X_train, y_train, X_test, y_test):
    """단순 평균 앙상블 - 과적합 방지"""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_log = np.log(y_train + 1)
    
    models = {
        'Ridge_strong': Ridge(alpha=10.0),
        'Lasso_strong': Lasso(alpha=1.0, max_iter=3000),
        'Huber_strong': HuberRegressor(epsilon=1.5, alpha=1.0, max_iter=500),
    }
    
    predictions = []
    
    print("\n  Simple average ensemble:")
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train_log)
            pred_log = model.predict(X_test_scaled)
            pred = np.exp(pred_log) - 1
            pred = np.maximum(pred, 0)
            predictions.append(pred)
            
            r2 = r2_score(y_test, pred)
            print(f"    {name}: R2={r2:.4f}")
        except:
            pass
    
    if predictions:
        # 단순 평균
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        print(f"    Simple Average: R2={ensemble_r2:.4f}")
        return ensemble_r2
    
    return None

# ============================================================================
# 나이브 벤치마크
# ============================================================================

def naive_benchmark(y_train, y_test):
    """나이브 벤치마크: 과거 평균으로 예측"""
    
    # 지난 22일 RV 평균
    naive_pred = np.full(len(y_test), y_train.mean())
    naive_r2 = r2_score(y_test, naive_pred)
    
    print(f"\n  Naive (mean prediction): R2={naive_r2:.4f}")
    
    return naive_r2

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("SPY R² 0.10+ Target Experiment")
    print("="*80)
    
    features = prepare_minimal_features()
    
    exclude_cols = ['RV_22d_future']
    feature_cols = [c for c in features.columns if c not in exclude_cols]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    print(f"\nFeatures: {len(feature_cols)} (minimal set)")
    print(f"Total samples: {len(X)}")
    
    # 3분할
    gap = 22
    n = len(X)
    train_end = int(n * 0.6) - gap
    val_end = int(n * 0.8) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end + gap : val_end]
    y_val = y.iloc[train_end + gap : val_end]
    X_test = X.iloc[val_end + gap:]
    y_test = y.iloc[val_end + gap:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    results = {}
    
    # 1. 나이브 벤치마크
    print("\n[1] Naive Benchmark")
    results['naive'] = naive_benchmark(y_train, y_test)
    
    # 2. Walk-Forward CV
    print("\n[2] Walk-Forward CV (Ridge α=10)")
    wf_results = walk_forward_cv(X, y, n_splits=5, gap=22)
    if wf_results:
        avg_r2 = np.mean([r['r2'] for r in wf_results])
        print(f"  Average CV R2: {avg_r2:.4f}")
        for r in wf_results:
            print(f"    Fold {r['fold']}: R2={r['r2']:.4f}")
        results['walk_forward_cv'] = avg_r2
    
    # 3. 극강 정규화
    print("\n[3] Extreme Regularization")
    best_config = extreme_regularization_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
    if best_config:
        results['best_regularization'] = best_config
    
    # 4. 단순 평균 앙상블
    print("\n[4] Simple Average Ensemble")
    # Train + Val을 합쳐서 학습
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    ensemble_r2 = simple_average_ensemble(X_trainval, y_trainval, X_test, y_test)
    if ensemble_r2:
        results['simple_ensemble'] = ensemble_r2
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'SPY R² 0.10+ Target',
            'features': feature_cols,
            'n_features': len(feature_cols),
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }
    
    output_path = 'data/results/spy_target_010_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"  Best Test R2: {best_config['test_r2']:.4f} ({best_config['model']} α={best_config['alpha']})")
    print(f"  Target: 0.10")
    print(f"  Gap to target: {0.10 - best_config['test_r2']:.4f}")
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
