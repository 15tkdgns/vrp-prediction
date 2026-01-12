#!/usr/bin/env python3
"""
성능 향상 실험
==============

현재 최고 성능(R² 0.2572)을 넘어설 수 있는 모델 탐색
- XGBoost, LightGBM, CatBoost
- Random Forest
- 앙상블 (Stacking, Voting)
- 하이퍼파라미터 최적화
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

# 추가 모델
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

SEED = 42
np.random.seed(SEED)


def load_and_prepare_data():
    """데이터 로드 및 특성 생성"""
    print("\n" + "=" * 60)
    print("[1/4] 데이터 준비")
    print("=" * 60)
    
    # SPY 데이터 로드
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    if csv_path.exists():
        spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
    
    # VIX 로드
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    spy['VIX'] = vix['Close']
    spy = spy.ffill().dropna()
    
    # 특성 생성
    spy['returns'] = spy['Close'].pct_change()
    
    # 변동성 특성
    for w in [5, 10, 20, 50]:
        spy[f'volatility_{w}'] = spy['returns'].rolling(w).std()
        spy[f'realized_vol_{w}'] = spy[f'volatility_{w}'] * np.sqrt(252)
    
    # VIX 특성
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_lag_5'] = spy['VIX'].shift(5)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_change_5'] = spy['VIX'].pct_change(5)
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    spy['vix_ma_ratio'] = spy['VIX'] / (spy['VIX'].rolling(20).mean() + 1e-8)
    
    # Regime 특성
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['regime_crisis'] = (vix_lagged >= 35).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vol_in_crisis'] = spy['regime_crisis'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    spy['vix_excess_35'] = np.maximum(vix_lagged - 35, 0)
    
    # 수익률 통계
    for w in [5, 10, 20]:
        spy[f'mean_return_{w}'] = spy['returns'].rolling(w).mean()
        spy[f'skew_{w}'] = spy['returns'].rolling(w).skew()
        spy[f'kurt_{w}'] = spy['returns'].rolling(w).kurt()
    
    # 래그 변수
    for lag in [1, 2, 3, 5]:
        spy[f'return_lag_{lag}'] = spy['returns'].shift(lag)
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    # 모멘텀
    for w in [5, 10, 20]:
        spy[f'momentum_{w}'] = spy['returns'].rolling(w).sum()
    
    # 비율
    spy['vol_ratio_5_20'] = spy['volatility_5'] / (spy['volatility_20'] + 1e-8)
    spy['vol_ratio_10_50'] = spy['volatility_10'] / (spy['volatility_50'] + 1e-8)
    
    # Z-score
    spy['zscore_20'] = (spy['returns'] - spy['returns'].rolling(20).mean()) / (spy['returns'].rolling(20).std() + 1e-8)
    
    # HAR 특성
    returns_sq = spy['returns'] ** 2
    spy['har_rv_d'] = returns_sq.shift(1)
    spy['har_rv_w'] = returns_sq.rolling(5).mean().shift(1)
    spy['har_rv_m'] = returns_sq.rolling(22).mean().shift(1)
    
    # 추가 VIX 상호작용
    spy['vix_vol_interaction'] = spy['vix_lag_1'] * spy['volatility_5']
    spy['vix_momentum_interaction'] = spy['vix_lag_1'] * spy['momentum_5']
    
    # 타겟: 5일 미래 변동성
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_window = returns[i+1:i+6]
            vol_values.append(pd.Series(future_window).std())
        else:
            vol_values.append(np.nan)
    spy['target'] = vol_values
    
    spy = spy.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 특성 컬럼
    feature_cols = [c for c in spy.columns if c.startswith((
        'volatility_', 'realized_vol_', 'vix_', 'regime_', 'vol_in_',
        'mean_return_', 'skew_', 'kurt_', 'return_lag_', 'vol_lag_',
        'momentum_', 'vol_ratio_', 'zscore_', 'har_', 'vix_excess_'
    ))]
    
    print(f"  ✓ 데이터: {len(spy)} 행, {len(feature_cols)} 특성")
    
    return spy, feature_cols


def run_experiments(spy, feature_cols):
    """모델 실험"""
    print("\n" + "=" * 60)
    print("[2/4] 모델 실험")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['target'].values
    
    # 분할
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    best_r2 = 0
    best_model_name = ""
    
    # 1. 기준 모델: ElasticNet + VIX + Regime
    print("\n  🔹 ElasticNet (baseline)")
    en = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_train)
    y_pred = en.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['ElasticNet'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))}
    print(f"     R² = {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = 'ElasticNet'
    
    # 2. Ridge (다양한 alpha)
    print("\n  🔹 Ridge (GridSearch)")
    ridge = Ridge(random_state=SEED)
    ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge_cv = GridSearchCV(ridge, ridge_params, cv=TimeSeriesSplit(n_splits=5), scoring='r2')
    ridge_cv.fit(X_train_s, y_train)
    y_pred = ridge_cv.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['Ridge'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred)), 
                        'best_params': ridge_cv.best_params_}
    print(f"     R² = {r2:.4f}, best alpha = {ridge_cv.best_params_['alpha']}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = 'Ridge'
    
    # 3. Lasso
    print("\n  🔹 Lasso (GridSearch)")
    lasso = Lasso(random_state=SEED, max_iter=10000)
    lasso_params = {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
    lasso_cv = GridSearchCV(lasso, lasso_params, cv=TimeSeriesSplit(n_splits=5), scoring='r2')
    lasso_cv.fit(X_train_s, y_train)
    y_pred = lasso_cv.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['Lasso'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'best_params': lasso_cv.best_params_}
    print(f"     R² = {r2:.4f}, best alpha = {lasso_cv.best_params_['alpha']}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = 'Lasso'
    
    # 4. Random Forest
    print("\n  🔹 Random Forest")
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, 
                               random_state=SEED, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['RandomForest'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))}
    print(f"     R² = {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = 'RandomForest'
    
    # 5. Gradient Boosting (제한적)
    print("\n  🔹 GradientBoosting (제한적)")
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05,
                                   min_samples_leaf=10, random_state=SEED)
    gb.fit(X_train_s, y_train)
    y_pred = gb.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['GradientBoosting'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))}
    print(f"     R² = {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = 'GradientBoosting'
    
    # 6. XGBoost
    if HAS_XGB:
        print("\n  🔹 XGBoost")
        xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                          reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, verbosity=0)
        xgb.fit(X_train_s, y_train)
        y_pred = xgb.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        results['XGBoost'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))}
        print(f"     R² = {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = 'XGBoost'
    
    # 7. LightGBM
    if HAS_LGBM:
        print("\n  🔹 LightGBM")
        lgbm = LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                            reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, verbose=-1)
        lgbm.fit(X_train_s, y_train)
        y_pred = lgbm.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        results['LightGBM'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))}
        print(f"     R² = {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = 'LightGBM'
    
    # 8. CatBoost
    if HAS_CATBOOST:
        print("\n  🔹 CatBoost")
        cat = CatBoostRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                               random_state=SEED, verbose=0)
        cat.fit(X_train_s, y_train)
        y_pred = cat.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        results['CatBoost'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))}
        print(f"     R² = {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = 'CatBoost'
    
    # 9. Stacking Ensemble
    print("\n  🔹 Stacking Ensemble (ElasticNet + Ridge + RF)")
    estimators = [
        ('en', ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=SEED, max_iter=10000)),
        ('ridge', Ridge(alpha=1.0, random_state=SEED)),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=6, random_state=SEED, n_jobs=-1))
    ]
    stacking = StackingRegressor(estimators=estimators, 
                                 final_estimator=Ridge(alpha=1.0),
                                 cv=5)
    stacking.fit(X_train_s, y_train)
    y_pred = stacking.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['Stacking'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred))}
    print(f"     R² = {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = 'Stacking'
    
    # 10. Simple Average Ensemble
    print("\n  🔹 Simple Average Ensemble")
    en_model = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    ridge_model = Ridge(alpha=1.0, random_state=SEED)
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=SEED, n_jobs=-1)
    
    en_model.fit(X_train_s, y_train)
    ridge_model.fit(X_train_s, y_train)
    rf_model.fit(X_train_s, y_train)
    
    y_pred_en = en_model.predict(X_test_s)
    y_pred_ridge = ridge_model.predict(X_test_s)
    y_pred_rf = rf_model.predict(X_test_s)
    
    y_pred_avg = (y_pred_en + y_pred_ridge + y_pred_rf) / 3
    r2 = r2_score(y_test, y_pred_avg)
    results['SimpleAverage'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred_avg))}
    print(f"     R² = {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = 'SimpleAverage'
    
    # 11. Weighted Ensemble (ElasticNet 가중치 높음)
    print("\n  🔹 Weighted Ensemble (EN:0.5, Ridge:0.3, RF:0.2)")
    y_pred_weighted = 0.5 * y_pred_en + 0.3 * y_pred_ridge + 0.2 * y_pred_rf
    r2 = r2_score(y_test, y_pred_weighted)
    results['WeightedEnsemble'] = {'r2': r2, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred_weighted))}
    print(f"     R² = {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = 'WeightedEnsemble'
    
    return results, best_model_name, best_r2, y_test


def optimize_best_model(spy, feature_cols, best_model_name):
    """최적 모델 하이퍼파라미터 튜닝"""
    print("\n" + "=" * 60)
    print(f"[3/4] 최적 모델 하이퍼파라미터 튜닝: {best_model_name}")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['target'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    best_r2 = 0
    best_params = {}
    
    if best_model_name in ['ElasticNet', 'Stacking', 'SimpleAverage', 'WeightedEnsemble']:
        # ElasticNet 상세 튜닝
        print("  → ElasticNet 상세 하이퍼파라미터 탐색")
        param_grid = {
            'alpha': [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.002],
            'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        
        en = ElasticNet(random_state=SEED, max_iter=10000)
        cv = GridSearchCV(en, param_grid, cv=TimeSeriesSplit(n_splits=5), 
                         scoring='r2', n_jobs=-1, verbose=1)
        cv.fit(X_train_s, y_train)
        
        y_pred = cv.predict(X_test_s)
        best_r2 = r2_score(y_test, y_pred)
        best_params = cv.best_params_
        
        print(f"\n  ✓ 최적 파라미터: {best_params}")
        print(f"  ✓ 최고 Test R²: {best_r2:.4f}")
    
    return best_r2, best_params


def main():
    """메인 함수"""
    print("\n" + "🚀" * 30)
    print("성능 향상 실험")
    print("🚀" * 30)
    
    # 1. 데이터 준비
    spy, feature_cols = load_and_prepare_data()
    
    # 2. 모델 실험
    results, best_model_name, best_r2, y_test = run_experiments(spy, feature_cols)
    
    # 3. 최적 모델 튜닝
    final_r2, best_params = optimize_best_model(spy, feature_cols, best_model_name)
    
    # 4. 결과 요약
    print("\n" + "=" * 60)
    print("[4/4] 실험 결과 요약")
    print("=" * 60)
    
    print("\n📊 모델별 성능 (Test R² 기준 정렬):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    baseline_r2 = 0.2572  # 기존 최고 성능
    
    for i, (model, metrics) in enumerate(sorted_results, 1):
        r2 = metrics['r2']
        diff = r2 - baseline_r2
        marker = "⭐" if r2 > baseline_r2 else "  "
        improvement = f"(+{diff:.4f})" if diff > 0 else f"({diff:.4f})"
        print(f"  {i:2d}. {marker} {model:20s}: R² = {r2:.4f} {improvement}")
    
    print(f"\n🏆 최고 성능 모델: {sorted_results[0][0]}")
    print(f"  • Test R²: {sorted_results[0][1]['r2']:.4f}")
    print(f"  • 기존 대비: {sorted_results[0][1]['r2'] - baseline_r2:+.4f}")
    
    if final_r2 > baseline_r2:
        print(f"\n✅ 성능 향상 성공!")
        print(f"  • 기존: R² = {baseline_r2:.4f}")
        print(f"  • 개선: R² = {final_r2:.4f}")
        print(f"  • 향상: +{(final_r2 - baseline_r2) / baseline_r2 * 100:.1f}%")
    else:
        print(f"\n⚠️ 기존 성능(R² = {baseline_r2:.4f})이 여전히 최고")
    
    # 결과 저장
    output = {
        'experiment_results': {m: {'r2': float(v['r2']), 'rmse': float(v['rmse'])} 
                               for m, v in results.items()},
        'best_model': best_model_name,
        'best_r2': float(best_r2),
        'baseline_r2': baseline_r2,
        'improvement': float(best_r2 - baseline_r2),
        'tuned_r2': float(final_r2),
        'tuned_params': best_params,
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = Path('paper/performance_improvement_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  💾 결과 저장: {output_path}")
    
    return output


if __name__ == '__main__':
    results = main()
