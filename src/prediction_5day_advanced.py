"""
5일 예측 고급 전략 실험
========================
22일에서 사용한 모든 전략을 5일 예측에 적용

전략:
1. 줌바흐 효과 (경로 의존성)
2. HAR-REQ (분위수 기반)
3. 시간 가변적 Hurst 지수
4. VIX 고급 특성
5. Jump 성분 분리
6. 최적 모델: No Log + AR(1) 조합
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

def calculate_bipower_variation(returns, window):
    abs_returns = returns.abs()
    bv = (np.pi / 2) * (abs_returns * abs_returns.shift(1)).rolling(window).sum() * 252
    return bv.iloc[:, 0] if isinstance(bv, pd.DataFrame) else bv

def calculate_rolling_hurst(series, window=60):
    """시간 가변적 Hurst 지수"""
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(window, len(series), 5):
        subseries = series.iloc[i-window:i].values
        try:
            lags = range(2, min(15, window//4))
            tau = [np.std(subseries[lag:] - subseries[:-lag]) for lag in lags]
            if all(t > 0 for t in tau):
                poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                result.iloc[i] = poly[0]
        except:
            pass
    
    return result.ffill().bfill()

# ============================================================================
# 고급 특성 준비 (5일 예측용)
# ============================================================================

def prepare_advanced_features_5d(ticker):
    """5일 예측용 고급 특성"""
    print(f"  Preparing advanced features for {ticker}...")
    
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    if len(data) < 500:
        return None
    
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=data.index)
    
    # === 1. 기본 HAR-RV (lag 적용) ===
    rv_1d = (returns ** 2) * 252
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    features['RV_1d_lag1'] = rv_1d.shift(1)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_ratio_5_22_lag1'] = (rv_5d / rv_22d.clip(lower=1e-8)).shift(1)
    
    # === 2. Jump 성분 (HAR-RV-CJ) ===
    bv_5d = calculate_bipower_variation(returns, 5)
    jump_5d = (rv_5d - bv_5d).clip(lower=0)
    features['BV_5d_lag1'] = bv_5d.shift(1)
    features['Jump_5d_lag1'] = jump_5d.shift(1)
    features['Jump_ratio_lag1'] = (jump_5d / rv_5d.clip(lower=1e-8)).shift(1)
    
    # === 3. VIX 특성 (lag 적용) ===
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    vix_aligned = vix_close.reindex(data.index).ffill()
    
    features['VIX_lag1'] = vix_aligned.shift(1)
    features['VIX_change_lag1'] = vix_aligned.pct_change().shift(1)
    features['VIX_ma5_lag1'] = vix_aligned.rolling(5).mean().shift(1)
    features['VRP_lag1'] = (vix_aligned.shift(1) ** 2 / 100) - rv_22d.shift(1)
    
    # VIX 레짐
    vix_lag = vix_aligned.shift(1)
    features['VIX_high_lag1'] = (vix_lag >= 20).astype(int)
    
    # === 4. 줌바흐 효과 (경로 의존성) ===
    for w in [5, 10]:
        weights = np.exp(np.linspace(-1, 0, w))
        weights /= weights.sum()
        
        weighted_ret = returns.rolling(w).apply(
            lambda x: np.sum(x * weights[-len(x):]) if len(x) == w else np.nan
        )
        features[f'weighted_return_{w}d_lag1'] = weighted_ret.shift(1)
        
        direction = returns.rolling(w).apply(lambda x: np.mean(x > 0))
        features[f'direction_{w}d_lag1'] = direction.shift(1)
        
        # 궤적-변동성 상호작용
        features[f'traj_vol_{w}d_lag1'] = (weighted_ret * rv_5d).shift(1)
    
    # === 5. HAR-REQ (분위수 기반) ===
    q25 = rv_5d.rolling(252, min_periods=60).quantile(0.25)
    q50 = rv_5d.rolling(252, min_periods=60).quantile(0.50)
    q75 = rv_5d.rolling(252, min_periods=60).quantile(0.75)
    
    features['rv_above_q50_lag1'] = (rv_5d.shift(1) > q50.shift(1)).astype(int)
    features['rv_above_q75_lag1'] = (rv_5d.shift(1) > q75.shift(1)).astype(int)
    features['rv_iqr_position_lag1'] = ((rv_5d - q25) / (q75 - q25 + 1e-8)).shift(1)
    
    # === 6. 시간 가변적 Hurst ===
    features['hurst_60d_lag1'] = calculate_rolling_hurst(rv_5d, 60).shift(1)
    
    # === 7. 수익률 특성 ===
    features['return_1d_lag1'] = returns.shift(1)
    features['return_5d_lag1'] = returns.rolling(5).sum().shift(1)
    
    # === 타겟 ===
    features['RV_5d_future'] = rv_5d.shift(-5)
    
    return features.dropna()

# ============================================================================
# NNLS 앙상블 (No Log 버전)
# ============================================================================

def nnls_ensemble_5d(X_train, y_train, X_val, y_val, X_test, y_test, use_log=False):
    """NNLS 앙상블 (No Log 옵션)"""
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    if use_log:
        y_train_t = np.log(y_train + 1)
    else:
        y_train_t = y_train.values
    
    models = {
        'Ridge_a1': Ridge(alpha=1.0, random_state=42),
        'Ridge_a10': Ridge(alpha=10.0, random_state=42),
        'Ridge_a100': Ridge(alpha=100.0, random_state=42),
        'Huber': HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000, random_state=42),
        'RF': RandomForestRegressor(n_estimators=50, max_depth=3, min_samples_leaf=30, random_state=42),
    }
    
    val_preds = {}
    test_preds = {}
    individual_r2 = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train_t)
            
            val_pred = model.predict(X_val_s)
            test_pred = model.predict(X_test_s)
            
            if use_log:
                val_pred = np.exp(val_pred) - 1
                test_pred = np.exp(test_pred) - 1
            
            val_pred = np.maximum(val_pred, 0)
            test_pred = np.maximum(test_pred, 0)
            
            val_preds[name] = val_pred
            test_preds[name] = test_pred
            
            individual_r2[name] = {
                'val': r2_score(y_val, val_pred),
                'test': r2_score(y_test, test_pred)
            }
        except Exception as e:
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
    
    val_matrix_pred = val_matrix @ weights
    
    return {
        'val_r2': r2_score(y_val, val_matrix_pred),
        'test_r2': r2_score(y_test, ensemble_test),
        'test_mae': mean_absolute_error(y_test, ensemble_test),
        'individual_r2': individual_r2,
        'weights': dict(zip(model_names, weights_norm.tolist()))
    }

# ============================================================================
# 자산별 실험
# ============================================================================

def run_experiment_5d_advanced(ticker):
    """5일 고급 전략 실험"""
    print(f"\n{'='*70}")
    print(f"5-Day Advanced Strategies: {ticker}")
    print(f"{'='*70}")
    
    features = prepare_advanced_features_5d(ticker)
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
    
    results = {}
    
    # 1. NNLS (No Log)
    print("\n  [1] NNLS Ensemble (No Log):")
    result_nolog = nnls_ensemble_5d(X_train, y_train, X_val, y_val, X_test, y_test, use_log=False)
    if result_nolog:
        results['nnls_nolog'] = result_nolog
        print(f"      Val R2: {result_nolog['val_r2']:.4f}, Test R2: {result_nolog['test_r2']:.4f}")
    
    # 2. NNLS (Log)
    print("\n  [2] NNLS Ensemble (Log):")
    result_log = nnls_ensemble_5d(X_train, y_train, X_val, y_val, X_test, y_test, use_log=True)
    if result_log:
        results['nnls_log'] = result_log
        print(f"      Val R2: {result_log['val_r2']:.4f}, Test R2: {result_log['test_r2']:.4f}")
    
    # 3. 단순 AR(1) + 고급 특성
    print("\n  [3] Simple Ridge (No Log):")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, y_train)
    pred = np.maximum(model.predict(X_test_s), 0)
    simple_r2 = r2_score(y_test, pred)
    results['simple_ridge'] = {'test_r2': simple_r2}
    print(f"      Test R2: {simple_r2:.4f}")
    
    # 4. Persistence 비교
    persist_r2 = r2_score(y_test, X_test['RV_5d_lag1'].values)
    results['persistence'] = {'test_r2': persist_r2}
    print(f"\n  [Benchmark] Persistence R2: {persist_r2:.4f}")
    
    # 최고 결과
    best_r2 = max([r.get('test_r2', r.get('test_r2', -999)) for r in results.values()])
    print(f"\n  Best Test R2: {best_r2:.4f}")
    
    return {
        'asset': ticker,
        'n_features': len(feature_cols),
        'results': results,
        'best_r2': best_r2
    }

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("5-Day Advanced Strategies Experiment")
    print("="*80)
    print("Applying all 22-day strategies to 5-day prediction")
    print("Strategies: Zumbach, HAR-REQ, Hurst, VIX, Jump, NNLS")
    
    assets = ['SPY', 'QQQ', 'EEM', 'GLD', 'TLT']
    
    all_results = {}
    
    for asset in assets:
        result = run_experiment_5d_advanced(asset)
        if result:
            all_results[asset] = result
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': '5-Day Advanced Strategies',
            'strategies': ['Zumbach Effect', 'HAR-REQ', 'Hurst', 'VIX', 'Jump', 'NNLS'],
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/5day_advanced_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("Summary: 5-Day Advanced Strategies")
    print("="*80)
    
    print(f"{'Asset':<8} {'Best R²':>10} {'Persistence':>12} {'Improvement':>12}")
    print("-"*45)
    
    for asset, result in all_results.items():
        best_r2 = result['best_r2']
        persist = result['results']['persistence']['test_r2']
        improvement = best_r2 - persist
        print(f"{asset:<8} {best_r2:>10.4f} {persist:>12.4f} {improvement:>12.4f}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
