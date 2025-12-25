"""
SPY R² 개선 실험 (누출 방지 버전)
================================
목표: R² 0.073 → 0.10+

핵심 원칙:
1. 모든 특성에 shift(1) 적용
2. Train/Val/Test 3분할 (60/20/20)
3. NNLS 가중치는 Val로만 학습
4. gap=22 적용
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

# ============================================================================
# 기본 지표
# ============================================================================

def calculate_rv(returns, window=22):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def calculate_bipower_variation(returns, window=22):
    """Bipower Variation (연속 변동성)"""
    abs_returns = returns.abs()
    bv = (np.pi / 2) * (abs_returns * abs_returns.shift(1)).rolling(window).sum() * 252
    return bv.iloc[:, 0] if isinstance(bv, pd.DataFrame) else bv

def calculate_jump(rv, bv):
    """Jump 성분 = max(RV - BV, 0)"""
    return (rv - bv).clip(lower=0)

# ============================================================================
# 특성 준비 (모든 특성 lag 적용)
# ============================================================================

def prepare_spy_features_leakage_free():
    """SPY 특성 - 모든 특성에 lag 적용"""
    print("Preparing SPY features (leakage-free)...")
    
    # SPY 데이터
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    returns = spy['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=spy.index)
    
    # === 1. HAR-RV (모두 lag1 적용) ===
    print("  [1] HAR-RV features (lagged)...")
    rv_1d = (returns ** 2) * 252
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    rv_66d = calculate_rv(returns, 66)
    
    features['RV_1d_lag1'] = rv_1d.shift(1)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_66d_lag1'] = rv_66d.shift(1)
    
    # RV 변화율
    features['RV_22d_change_lag1'] = rv_22d.pct_change().shift(1)
    features['RV_22d_ma5_lag1'] = rv_22d.rolling(5).mean().shift(1)
    
    # === 2. HAR-RV-CJ Jump 성분 (lag1) ===
    print("  [2] Jump components (lagged)...")
    bv_22d = calculate_bipower_variation(returns, 22)
    jump_22d = calculate_jump(rv_22d, bv_22d)
    
    features['BV_22d_lag1'] = bv_22d.shift(1)
    features['Jump_22d_lag1'] = jump_22d.shift(1)
    features['Jump_ratio_lag1'] = (jump_22d / rv_22d.clip(lower=1e-8)).shift(1)
    
    # === 3. VIX 특성 (모두 lag1) ===
    print("  [3] VIX features (lagged)...")
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close']
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]
    vix_aligned = vix_close.reindex(spy.index).ffill()
    
    features['VIX_lag1'] = vix_aligned.shift(1)
    features['VIX_lag5'] = vix_aligned.shift(5)
    features['VIX_ma5_lag1'] = vix_aligned.rolling(5).mean().shift(1)
    features['VIX_ma22_lag1'] = vix_aligned.rolling(22).mean().shift(1)
    features['VIX_change_lag1'] = vix_aligned.pct_change().shift(1)
    
    # VIX 레짐 (lag1)
    vix_lag = vix_aligned.shift(1)
    features['VIX_high_lag1'] = (vix_lag >= 25).astype(int)
    features['VIX_extreme_lag1'] = (vix_lag >= 35).astype(int)
    
    # VIX 텀 스트럭처 (lag1)
    features['VIX_term_lag1'] = (vix_aligned / vix_aligned.rolling(22).mean()).shift(1)
    
    # === 4. VRP (lag1) ===
    print("  [4] VRP features (lagged)...")
    vrp = (vix_aligned ** 2 / 100) - rv_22d
    features['VRP_lag1'] = vrp.shift(1)
    features['VRP_ma5_lag1'] = vrp.rolling(5).mean().shift(1)
    features['VRP_std5_lag1'] = vrp.rolling(5).std().shift(1)
    
    # === 5. 줌바흐 효과 (경로 의존성, lag1) ===
    print("  [5] Zumbach effect (lagged)...")
    for w in [5, 22]:
        # 가중 수익률
        weights = np.exp(np.linspace(-1, 0, w))
        weights /= weights.sum()
        weighted_ret = returns.rolling(w).apply(
            lambda x: np.sum(x * weights[-len(x):]) if len(x) == w else np.nan
        )
        features[f'weighted_return_{w}d_lag1'] = weighted_ret.shift(1)
        
        # 방향성 (양수 비율)
        direction = returns.rolling(w).apply(lambda x: np.mean(x > 0))
        features[f'direction_{w}d_lag1'] = direction.shift(1)
    
    # === 6. 수익률 특성 (lag1) ===
    print("  [6] Return features (lagged)...")
    features['return_1d_lag1'] = returns.shift(1)
    features['return_5d_lag1'] = returns.rolling(5).sum().shift(1)
    features['return_22d_lag1'] = returns.rolling(22).sum().shift(1)
    
    # === 타겟 ===
    features['RV_22d_future'] = rv_22d.shift(-22)
    
    return features.dropna()

# ============================================================================
# NNLS 앙상블 (누출 방지)
# ============================================================================

def leakage_free_nnls_ensemble(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    누출 방지 NNLS 앙상블:
    - Train: 모델 학습
    - Val: NNLS 가중치 학습
    - Test: 순수 OOS 평가
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_log = np.log(y_train + 1)
    
    models = {
        'Lasso_light': Lasso(alpha=0.001, max_iter=3000, random_state=42),
        'Lasso_medium': Lasso(alpha=0.01, max_iter=3000, random_state=42),
        'Lasso_strong': Lasso(alpha=0.1, max_iter=3000, random_state=42),
        'Ridge': Ridge(alpha=0.1, random_state=42),
        'Huber': HuberRegressor(epsilon=1.35, alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000, random_state=42),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    }
    
    val_predictions = {}
    test_predictions = {}
    results = {'individual': {}}
    
    print("\n  Individual model performance:")
    for name, model in models.items():
        try:
            # Train으로 학습
            model.fit(X_train_scaled, y_train_log)
            
            # Val 예측
            val_pred_log = model.predict(X_val_scaled)
            val_pred = np.exp(val_pred_log) - 1
            val_pred = np.maximum(val_pred, 0)
            val_predictions[name] = val_pred
            val_r2 = r2_score(y_val, val_pred)
            
            # Test 예측
            test_pred_log = model.predict(X_test_scaled)
            test_pred = np.exp(test_pred_log) - 1
            test_pred = np.maximum(test_pred, 0)
            test_predictions[name] = test_pred
            test_r2 = r2_score(y_test, test_pred)
            
            results['individual'][name] = {'val_r2': val_r2, 'test_r2': test_r2}
            print(f"    {name}: Val R2={val_r2:.4f}, Test R2={test_r2:.4f}")
            
        except Exception as e:
            print(f"    {name} failed: {e}")
    
    if len(val_predictions) < 2:
        return None
    
    # NNLS 가중치: Val로만 학습!
    model_names = list(val_predictions.keys())
    val_pred_matrix = np.column_stack([val_predictions[m] for m in model_names])
    weights, _ = nnls(val_pred_matrix, y_val.values)
    
    # 가중치 정규화
    if weights.sum() > 0:
        weights_norm = weights / weights.sum()
    else:
        weights_norm = np.ones(len(weights)) / len(weights)
    
    # Val 앙상블 성능
    ensemble_val = val_pred_matrix @ weights
    val_r2 = r2_score(y_val, ensemble_val)
    
    # Test 앙상블 성능 (순수 OOS)
    test_pred_matrix = np.column_stack([test_predictions[m] for m in model_names])
    ensemble_test = test_pred_matrix @ weights
    test_r2 = r2_score(y_test, ensemble_test)
    test_mae = mean_absolute_error(y_test, ensemble_test)
    
    results['ensemble'] = {
        'val_r2': val_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'weights': dict(zip(model_names, weights_norm.tolist()))
    }
    
    return results

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("SPY R² Optimization (Leakage-Free)")
    print("="*80)
    
    features = prepare_spy_features_leakage_free()
    
    # 특성/타겟 분리
    exclude_cols = ['RV_22d_future']
    feature_cols = [c for c in features.columns if c not in exclude_cols]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    # 3분할: Train(60%) / Val(20%) / Test(20%)
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
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Gap: {gap} days")
    
    # 앙상블 실행
    results = leakage_free_nnls_ensemble(X_train, y_train, X_val, y_val, X_test, y_test)
    
    if results:
        print(f"\n{'='*60}")
        print("NNLS Ensemble Results (Leakage-Free)")
        print(f"{'='*60}")
        print(f"  Validation R2: {results['ensemble']['val_r2']:.4f}")
        print(f"  Test R2 (OOS): {results['ensemble']['test_r2']:.4f}")
        print(f"  Test MAE: {results['ensemble']['test_mae']:.4f}")
        print(f"  Weights: {results['ensemble']['weights']}")
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'SPY Leakage-Free Optimization',
            'features': len(feature_cols),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'gap': gap,
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }
    
    output_path = 'data/results/spy_leakage_free_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
