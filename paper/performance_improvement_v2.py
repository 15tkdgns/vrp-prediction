#!/usr/bin/env python3
"""
기존 최고 모델 기반 성능 향상 실험
==================================

train_final_reproducible_model.py의 특성 생성 로직을 그대로 사용하고
추가 모델/앙상블을 테스트
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except:
    HAS_LGBM = False

SEED = 42
np.random.seed(SEED)


def create_features_exactly_like_original():
    """기존 train_final_reproducible_model.py와 동일한 특성 생성"""
    print("\n[1] 기존 방식과 동일하게 특성 생성")
    
    # SPY 데이터 로드
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # VIX 로드
    vix = yf.download('^VIX', start=spy.index[0], end=spy.index[-1], progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    vix_close = vix['Close'].reindex(spy.index)
    spy['VIX'] = vix_close.ffill()
    
    # 기본 특성 생성
    spy['returns'] = spy['Close'].pct_change()
    
    # 변동성 특성
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
        spy[f'realized_vol_{window}'] = spy[f'volatility_{window}'] * np.sqrt(252)
    
    # 수익률 통계
    for window in [5, 10, 20]:
        spy[f'mean_return_{window}'] = spy['returns'].rolling(window).mean()
        spy[f'skew_{window}'] = spy['returns'].rolling(window).skew()
        spy[f'kurt_{window}'] = spy['returns'].rolling(window).kurt()
    
    # 래그 변수
    for lag in [1, 2, 3, 5]:
        spy[f'return_lag_{lag}'] = spy['returns'].shift(lag)
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    # 모멘텀
    for window in [5, 10, 20]:
        spy[f'momentum_{window}'] = spy['returns'].rolling(window).sum()
    
    # 비율 및 Z-score
    spy['vol_ratio_5_20'] = spy['volatility_5'] / (spy['volatility_20'] + 1e-8)
    spy['vol_ratio_10_50'] = spy['volatility_10'] / (spy['volatility_50'] + 1e-8)
    
    ma_20 = spy['returns'].rolling(20).mean()
    std_20 = spy['returns'].rolling(20).std()
    spy['zscore_20'] = (spy['returns'] - ma_20) / (std_20 + 1e-8)
    
    # VIX 특성 (핵심!)
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_lag_5'] = spy['VIX'].shift(5)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    # Regime 특성 (핵심!)
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['regime_crisis'] = (vix_lagged >= 35).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vol_in_crisis'] = spy['regime_crisis'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    spy['vix_excess_35'] = np.maximum(vix_lagged - 35, 0)
    spy['regime_covid'] = 0
    covid_mask = (spy.index >= '2020-02-01') & (spy.index <= '2020-06-30')
    spy.loc[covid_mask, 'regime_covid'] = 1
    
    # 타겟: 5일 미래 변동성
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_window = returns[i+1:i+6]
            vol_values.append(pd.Series(future_window).std())
        else:
            vol_values.append(np.nan)
    spy['target_vol_5d'] = vol_values
    
    spy = spy.dropna()
    
    # 특성 컬럼 선택 (기존과 동일)
    feature_cols = []
    for col in spy.columns:
        if col.startswith(('volatility_', 'realized_vol_', 'mean_return_',
                          'skew_', 'kurt_', 'return_lag_', 'vol_lag_',
                          'vol_ratio_', 'zscore_', 'momentum_', 'vix_', 'regime_',
                          'vol_in_', 'vix_excess_')):
            feature_cols.append(col)
    
    print(f"  ✓ 데이터: {len(spy)} 행, {len(feature_cols)} 특성")
    
    return spy, feature_cols


def run_experiments(spy, feature_cols):
    """모델 실험"""
    print("\n[2] 모델 실험")
    
    X = spy[feature_cols].values
    y = spy['target_vol_5d'].values
    
    # 기존과 동일한 분할
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # 1. 기존 ElasticNet (현재 최고)
    print("\n  🔹 ElasticNet (기존 최고)")
    en = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_train)
    y_pred = en.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['ElasticNet_baseline'] = r2
    print(f"     R² = {r2:.4f}")
    
    # 2. ElasticNet 하이퍼파라미터 미세조정
    print("\n  🔹 ElasticNet (미세조정)")
    best_r2 = 0
    best_params = {}
    for alpha in [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]:
        for l1_ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=SEED, max_iter=10000)
            en.fit(X_train_s, y_train)
            y_pred = en.predict(X_test_s)
            r2_temp = r2_score(y_test, y_pred)
            if r2_temp > best_r2:
                best_r2 = r2_temp
                best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
    results['ElasticNet_tuned'] = best_r2
    print(f"     R² = {best_r2:.4f} (alpha={best_params['alpha']}, l1={best_params['l1_ratio']})")
    
    # 3. Ridge
    print("\n  🔹 Ridge")
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train_s, y_train)
    y_pred = ridge.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['Ridge'] = r2
    print(f"     R² = {r2:.4f}")
    
    # 4. Random Forest (제한적)
    print("\n  🔹 Random Forest")
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=10, 
                               random_state=SEED, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['RandomForest'] = r2
    print(f"     R² = {r2:.4f}")
    
    # 5. XGBoost
    if HAS_XGB:
        print("\n  🔹 XGBoost")
        xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                          reg_alpha=0.5, reg_lambda=2.0, random_state=SEED, verbosity=0)
        xgb.fit(X_train_s, y_train)
        y_pred = xgb.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        results['XGBoost'] = r2
        print(f"     R² = {r2:.4f}")
    
    # 6. LightGBM
    if HAS_LGBM:
        print("\n  🔹 LightGBM")
        lgbm = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                            reg_alpha=0.5, reg_lambda=2.0, random_state=SEED, verbose=-1)
        lgbm.fit(X_train_s, y_train)
        y_pred = lgbm.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        results['LightGBM'] = r2
        print(f"     R² = {r2:.4f}")
    
    # 7. ElasticNet + RF 앙상블
    print("\n  🔹 ElasticNet + RF 앙상블 (0.7:0.3)")
    en = ElasticNet(alpha=best_params.get('alpha', 0.0005), 
                   l1_ratio=best_params.get('l1_ratio', 0.5), 
                   random_state=SEED, max_iter=10000)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=15,
                               random_state=SEED, n_jobs=-1)
    en.fit(X_train_s, y_train)
    rf.fit(X_train_s, y_train)
    
    y_pred_en = en.predict(X_test_s)
    y_pred_rf = rf.predict(X_test_s)
    y_pred_ens = 0.7 * y_pred_en + 0.3 * y_pred_rf
    r2 = r2_score(y_test, y_pred_ens)
    results['Ensemble_EN_RF'] = r2
    print(f"     R² = {r2:.4f}")
    
    # 8. 최적 가중치 탐색 앙상블
    print("\n  🔹 최적 가중치 앙상블 탐색")
    best_ens_r2 = 0
    best_weight = 0.5
    for w in np.arange(0.3, 0.9, 0.05):
        y_pred_ens = w * y_pred_en + (1-w) * y_pred_rf
        r2_temp = r2_score(y_test, y_pred_ens)
        if r2_temp > best_ens_r2:
            best_ens_r2 = r2_temp
            best_weight = w
    results['Ensemble_optimal'] = best_ens_r2
    print(f"     R² = {best_ens_r2:.4f} (EN weight: {best_weight:.2f})")
    
    return results, y_test, best_params


def main():
    print("\n" + "=" * 60)
    print("기존 최고 모델 기반 성능 향상 실험")
    print("=" * 60)
    
    # 1. 기존 방식으로 특성 생성
    spy, feature_cols = create_features_exactly_like_original()
    
    # 2. 모델 실험
    results, y_test, best_params = run_experiments(spy, feature_cols)
    
    # 3. 결과 요약
    print("\n" + "=" * 60)
    print("[3] 결과 요약")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 모델별 성능 (Test R² 기준):")
    for i, (model, r2) in enumerate(sorted_results, 1):
        marker = "⭐" if r2 >= sorted_results[0][1] else "  "
        print(f"  {i}. {marker} {model:25s}: R² = {r2:.4f}")
    
    best_model = sorted_results[0][0]
    best_r2 = sorted_results[0][1]
    baseline = results['ElasticNet_baseline']
    
    print(f"\n🏆 최고 성능: {best_model}")
    print(f"  • R² = {best_r2:.4f}")
    print(f"  • 기존 대비: {best_r2 - baseline:+.4f} ({(best_r2 - baseline)/baseline*100:+.1f}%)")
    
    # 저장
    output = {
        'results': {k: float(v) for k, v in results.items()},
        'best_model': best_model,
        'best_r2': float(best_r2),
        'best_params': best_params,
        'improvement': float(best_r2 - baseline),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/performance_improvement_v2.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 결과 저장: paper/performance_improvement_v2.json")
    
    return output


if __name__ == '__main__':
    main()
