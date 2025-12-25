"""
5일 예측 실험
=============
기존 22일 예측 → 5일 예측으로 변경
단기 예측은 더 쉬움 → 높은 R² 예상

모든 자산: SPY, QQQ, EEM, GLD, TLT
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Lasso, Ridge, HuberRegressor, ElasticNet
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

def prepare_features_5d(ticker):
    """5일 예측용 특성 준비"""
    print(f"  Preparing features for {ticker}...")
    
    data = yf.download(ticker, start='2018-01-01', end='2025-01-01', progress=False)
    if len(data) < 200:
        return None
    
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=data.index)
    
    # HAR-RV (lag 적용) - 5일 예측에 맞게 조정
    rv_1d = (returns ** 2) * 252
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    features['RV_1d_lag1'] = rv_1d.shift(1)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    
    # VIX (lag 적용)
    vix = yf.download('^VIX', start='2018-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    vix_aligned = vix_close.reindex(data.index).ffill()
    
    features['VIX_lag1'] = vix_aligned.shift(1)
    features['VIX_change_lag1'] = vix_aligned.pct_change().shift(1)
    
    # VRP (lag 적용)
    features['VRP_lag1'] = (vix_aligned.shift(1) ** 2 / 100) - rv_22d.shift(1)
    
    # 수익률 (lag 적용)
    features['return_1d_lag1'] = returns.shift(1)
    features['return_5d_lag1'] = returns.rolling(5).sum().shift(1)
    
    # 타겟: 5일 후 RV
    features['RV_5d_future'] = rv_5d.shift(-5)
    
    return features.dropna()

def run_nnls_ensemble_5d(X_train, y_train, X_val, y_val, X_test, y_test):
    """NNLS 앙상블 (5일 예측용)"""
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    y_train_log = np.log(y_train + 1)
    
    models = {
        'Lasso': Lasso(alpha=0.01, max_iter=3000, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Huber': HuberRegressor(epsilon=1.35, alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000, random_state=42),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42),
    }
    
    val_preds = {}
    test_preds = {}
    individual_r2 = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train_log)
            
            val_pred = np.exp(model.predict(X_val_s)) - 1
            test_pred = np.exp(model.predict(X_test_s)) - 1
            
            val_preds[name] = np.maximum(val_pred, 0)
            test_preds[name] = np.maximum(test_pred, 0)
            
            individual_r2[name] = {
                'val': r2_score(y_val, val_pred),
                'test': r2_score(y_test, test_pred)
            }
        except:
            pass
    
    if len(val_preds) < 2:
        return None
    
    # NNLS 가중치: Val로 학습
    model_names = list(val_preds.keys())
    val_matrix = np.column_stack([val_preds[m] for m in model_names])
    weights, _ = nnls(val_matrix, y_val.values)
    
    weights_norm = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
    
    # Test 앙상블
    test_matrix = np.column_stack([test_preds[m] for m in model_names])
    ensemble_test = test_matrix @ weights
    
    val_ensemble = val_matrix @ weights
    
    return {
        'val_r2': r2_score(y_val, val_ensemble),
        'test_r2': r2_score(y_test, ensemble_test),
        'test_mae': mean_absolute_error(y_test, ensemble_test),
        'individual_r2': individual_r2,
        'weights': dict(zip(model_names, weights_norm.tolist()))
    }

def run_experiment_5d(ticker):
    """5일 예측 실험"""
    print(f"\n{'='*60}")
    print(f"5-Day Prediction: {ticker}")
    print(f"{'='*60}")
    
    features = prepare_features_5d(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    # 3분할 (gap=5)
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
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    result = run_nnls_ensemble_5d(X_train, y_train, X_val, y_val, X_test, y_test)
    
    if result:
        print(f"\n  Results:")
        print(f"    Val R2: {result['val_r2']:.4f}")
        print(f"    Test R2: {result['test_r2']:.4f}")
        print(f"    Test MAE: {result['test_mae']:.4f}")
        print(f"    Weights: {result['weights']}")
        
        result['asset'] = ticker
        result['horizon'] = '5-day'
    
    return result

def main():
    print("="*80)
    print("5-Day RV Prediction Experiment")
    print("="*80)
    print("Prediction Horizon: 5 days (vs 22 days previously)")
    print("Gap: 5 days")
    
    assets = ['SPY', 'QQQ', 'EEM', 'GLD', 'TLT']
    
    all_results = {}
    
    for asset in assets:
        result = run_experiment_5d(asset)
        if result:
            all_results[asset] = result
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': '5-Day RV Prediction',
            'horizon': 5,
            'gap': 5,
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/5day_prediction_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("Summary: 5-Day Prediction Results")
    print("="*80)
    print(f"{'Asset':<8} {'Test R²':>10} {'22-Day R² (prev)':>18}")
    print("-"*40)
    
    prev_r2 = {'SPY': 0.085, 'QQQ': 0.127, 'EEM': 0.185, 'GLD': 0.030, 'TLT': 0.005}
    
    for asset, result in all_results.items():
        prev = prev_r2.get(asset, 0)
        print(f"{asset:<8} {result['test_r2']:>10.4f} {prev:>18.3f}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
