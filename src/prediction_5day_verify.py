"""
5일 예측 누출 검증 및 튜닝
===========================
목표:
1. Walk-Forward CV로 R² 검증
2. 기간별 성능 안정성 확인
3. 하이퍼파라미터 최적화
4. 최종 Robust R² 확정

상위 성능 자산: SPY, QQQ, XLE, XLF
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_features(ticker):
    """하이브리드 특성"""
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
    
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(data.index).ffill().shift(1)
    features['VIX_change_lag1'] = features['VIX_lag1'].pct_change()
    features['direction_5d_lag1'] = returns.rolling(5).apply(lambda x: np.mean(x > 0)).shift(1)
    
    features['RV_5d_future'] = rv_5d.shift(-5)
    
    return features.dropna()

# ============================================================================
# 검증 1: Walk-Forward CV
# ============================================================================

def walk_forward_cv(ticker, transform='raw', n_splits=5):
    """Walk-Forward CV로 검증"""
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gap = 5
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        train_idx = train_idx[:-gap] if len(train_idx) > gap else train_idx
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        if len(X_train) < 200 or len(X_test) < 50:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # 타겟 변환
        if transform == 'log':
            y_train_t = np.log(y_train + 1)
        elif transform == 'boxcox':
            lam = 0.25
            y_train_t = (y_train ** lam - 1) / lam
        else:
            y_train_t = y_train.values
        
        model = Ridge(alpha=100.0)
        model.fit(X_train_s, y_train_t)
        
        pred_t = model.predict(X_test_s)
        
        # 역변환
        if transform == 'log':
            pred = np.exp(pred_t) - 1
        elif transform == 'boxcox':
            lam = 0.25
            pred = ((pred_t * lam) + 1) ** (1/lam)
        else:
            pred = pred_t
        
        pred = np.maximum(pred, 0)
        pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
        
        r2 = r2_score(y_test, pred)
        
        test_start = X_test.index[0].strftime('%Y-%m')
        test_end = X_test.index[-1].strftime('%Y-%m')
        
        fold_results.append({
            'fold': fold,
            'period': f'{test_start}~{test_end}',
            'r2': r2,
            'train_size': len(X_train),
            'test_size': len(X_test)
        })
    
    return fold_results

# ============================================================================
# 검증 2: 기간별 성능
# ============================================================================

def period_analysis(ticker, transform='log'):
    """연도별 성능 분석"""
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    results = {}
    
    # 각 연도별로 테스트
    for test_year in [2021, 2022, 2023, 2024]:
        train_end = f'{test_year}-01-01'
        test_start = f'{test_year}-01-01'
        test_end = f'{test_year}-12-31'
        
        train_mask = X.index < train_end
        test_mask = (X.index >= test_start) & (X.index <= test_end)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        if len(X_train) < 200 or len(X_test) < 20:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        if transform == 'log':
            y_train_t = np.log(y_train + 1)
        elif transform == 'boxcox':
            lam = 0.25
            y_train_t = (y_train ** lam - 1) / lam
        else:
            y_train_t = y_train.values
        
        model = Ridge(alpha=100.0)
        model.fit(X_train_s, y_train_t)
        
        pred_t = model.predict(X_test_s)
        
        if transform == 'log':
            pred = np.exp(pred_t) - 1
        elif transform == 'boxcox':
            lam = 0.25
            pred = ((pred_t * lam) + 1) ** (1/lam)
        else:
            pred = pred_t
        
        pred = np.maximum(pred, 0)
        pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
        
        r2 = r2_score(y_test, pred)
        results[test_year] = r2
    
    return results

# ============================================================================
# 튜닝: alpha 최적화
# ============================================================================

def tune_alpha(ticker, transform='log'):
    """Alpha 하이퍼파라미터 튜닝"""
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.6) - gap
    val_end = int(n * 0.8) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end + gap : val_end]
    y_val = y.iloc[train_end + gap : val_end]
    X_test = X.iloc[val_end + gap:]
    y_test = y.iloc[val_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    if transform == 'log':
        y_train_t = np.log(y_train + 1)
    elif transform == 'boxcox':
        lam = 0.25
        y_train_t = (y_train ** lam - 1) / lam
    else:
        y_train_t = y_train.values
    
    results = {}
    
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train_t)
        
        pred_val_t = model.predict(X_val_s)
        pred_test_t = model.predict(X_test_s)
        
        if transform == 'log':
            pred_val = np.exp(pred_val_t) - 1
            pred_test = np.exp(pred_test_t) - 1
        elif transform == 'boxcox':
            lam = 0.25
            pred_val = ((pred_val_t * lam) + 1) ** (1/lam)
            pred_test = ((pred_test_t * lam) + 1) ** (1/lam)
        else:
            pred_val = pred_val_t
            pred_test = pred_test_t
        
        pred_val = np.maximum(pred_val, 0)
        pred_test = np.maximum(pred_test, 0)
        pred_val = np.nan_to_num(pred_val, nan=0, posinf=0, neginf=0)
        pred_test = np.nan_to_num(pred_test, nan=0, posinf=0, neginf=0)
        
        results[alpha] = {
            'val_r2': r2_score(y_val, pred_val),
            'test_r2': r2_score(y_test, pred_test)
        }
    
    # 최적 alpha (Val 기준)
    best_alpha = max(results.items(), key=lambda x: x[1]['val_r2'])
    
    return {
        'all_results': results,
        'best_alpha': best_alpha[0],
        'best_val_r2': best_alpha[1]['val_r2'],
        'best_test_r2': best_alpha[1]['test_r2']
    }

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("5-Day Leakage Check & Tuning")
    print("="*80)
    
    # 상위 성능 자산과 최적 transform
    assets = {
        'SPY': 'log',
        'QQQ': 'raw',  # Stacking이었지만 검증용으로 raw
        'XLE': 'boxcox',
        'XLF': 'boxcox',
        'XLK': 'log',
        'EEM': 'boxcox'
    }
    
    all_results = {}
    
    for ticker, transform in assets.items():
        print(f"\n{'='*60}")
        print(f"{ticker} (transform={transform})")
        print(f"{'='*60}")
        
        all_results[ticker] = {'transform': transform}
        
        # 1. Walk-Forward CV
        print("\n[1] Walk-Forward CV (5-fold):")
        wf_results = walk_forward_cv(ticker, transform)
        if wf_results:
            for r in wf_results:
                print(f"    Fold {r['fold']}: {r['period']} | R²={r['r2']:.4f}")
            
            avg_r2 = np.mean([r['r2'] for r in wf_results])
            std_r2 = np.std([r['r2'] for r in wf_results])
            print(f"    Average: {avg_r2:.4f} (±{std_r2:.4f})")
            
            all_results[ticker]['wf_cv'] = {
                'avg_r2': avg_r2,
                'std_r2': std_r2,
                'folds': wf_results
            }
        
        # 2. 연도별 분석
        print("\n[2] Yearly Performance:")
        yearly = period_analysis(ticker, transform)
        if yearly:
            for year, r2 in yearly.items():
                print(f"    {year}: R²={r2:.4f}")
            all_results[ticker]['yearly'] = yearly
        
        # 3. Alpha 튜닝
        print("\n[3] Alpha Tuning:")
        tune_results = tune_alpha(ticker, transform)
        if tune_results:
            print(f"    Best alpha: {tune_results['best_alpha']}")
            print(f"    Val R²: {tune_results['best_val_r2']:.4f}")
            print(f"    Test R²: {tune_results['best_test_r2']:.4f}")
            all_results[ticker]['tuning'] = tune_results
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': '5-Day Leakage Check & Tuning',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/5day_verification_tuning.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("Summary: Verified R² (Walk-Forward CV)")
    print("="*80)
    
    print(f"{'Asset':<8} {'Original':>10} {'CV Avg':>10} {'CV Std':>10} {'Best Alpha':>12}")
    print("-"*55)
    
    original_r2 = {'SPY': 0.40, 'QQQ': 0.40, 'XLE': 0.38, 'XLF': 0.31, 'XLK': 0.22, 'EEM': 0.22}
    
    for ticker in assets.keys():
        if ticker in all_results and 'wf_cv' in all_results[ticker]:
            orig = original_r2.get(ticker, 0)
            avg = all_results[ticker]['wf_cv']['avg_r2']
            std = all_results[ticker]['wf_cv']['std_r2']
            best_alpha = all_results[ticker].get('tuning', {}).get('best_alpha', 'N/A')
            print(f"{ticker:<8} {orig:>10.2f} {avg:>10.4f} {std:>10.4f} {best_alpha:>12}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
