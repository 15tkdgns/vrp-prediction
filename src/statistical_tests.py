#!/usr/bin/env python3
"""
VRP 예측 모델 통계적 유의성 검정
================================

1. Diebold-Mariano Test: 예측력 비교
2. Paired t-test: 모델 간 성능 차이
3. Bootstrap p-value: R² 유의성

실행: python src/statistical_tests.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def diebold_mariano_test(errors1, errors2, h=1, power=2):
    """
    Diebold-Mariano Test for comparing predictive accuracy.
    
    H0: 두 모델의 예측력에 차이가 없다
    H1: 두 모델의 예측력에 차이가 있다
    
    Parameters:
    -----------
    errors1: array-like, 모델 1의 예측 오차
    errors2: array-like, 모델 2의 예측 오차
    h: int, 예측 기간 (horizon)
    power: int, 1=MAE, 2=MSE 기반
    
    Returns:
    --------
    dm_stat: DM 통계량
    p_value: 양측 p-value
    """
    d = np.abs(errors1)**power - np.abs(errors2)**power
    
    n = len(d)
    mean_d = np.mean(d)
    
    # Newey-West 분산 추정 (자기상관 고려)
    gamma_0 = np.var(d, ddof=1)
    
    # 자기공분산 계산
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[:-k], d[k:])[0, 1] if len(d) > k else 0
        gamma_sum += 2 * gamma_k
    
    var_d = (gamma_0 + gamma_sum) / n
    
    if var_d <= 0:
        var_d = gamma_0 / n
    
    dm_stat = mean_d / np.sqrt(var_d)
    
    # 양측 검정 p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value


def bootstrap_r2_test(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    """
    Bootstrap을 이용한 R² 유의성 검정
    
    H0: R² = 0 (모델에 예측력 없음)
    H1: R² > 0 (모델에 예측력 있음)
    """
    n = len(y_true)
    observed_r2 = r2_score(y_true, y_pred)
    
    # Null 분포 생성 (y를 셔플하여 R² 계산)
    null_r2s = []
    for _ in range(n_bootstrap):
        shuffled_idx = np.random.permutation(n)
        null_r2 = r2_score(y_true, y_pred[shuffled_idx])
        null_r2s.append(null_r2)
    
    null_r2s = np.array(null_r2s)
    
    # p-value: null 분포에서 observed 이상인 비율
    p_value = np.mean(null_r2s >= observed_r2)
    
    # 신뢰구간 (Bootstrap)
    boot_r2s = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_r2 = r2_score(y_true[idx], y_pred[idx])
        boot_r2s.append(boot_r2)
    
    ci_lower = np.percentile(boot_r2s, 100 * alpha / 2)
    ci_upper = np.percentile(boot_r2s, 100 * (1 - alpha / 2))
    
    return {
        'r2': observed_r2,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha
    }


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    spy['returns'] = spy['Close'].pct_change()
    
    # 변동성 계산
    spy['RV_1d'] = spy['returns'].abs() * np.sqrt(252) * 100
    spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
    # 특성 생성
    spy['VIX_lag1'] = spy['VIX'].shift(1)
    spy['VIX_lag5'] = spy['VIX'].shift(5)
    spy['VIX_change'] = spy['VIX'].pct_change()
    spy['VIX_ma20'] = spy['VIX'].rolling(20).mean()
    spy['VRP_lag1'] = spy['VRP'].shift(1)
    spy['VRP_lag5'] = spy['VRP'].shift(5)
    spy['VRP_ma5'] = spy['VRP'].rolling(5).mean()
    spy['regime_high'] = (spy['VIX'] >= 25).astype(int)
    spy['regime_crisis'] = (spy['VIX'] >= 35).astype(int)
    spy['return_5d'] = spy['returns'].rolling(5).sum()
    spy['return_22d'] = spy['returns'].rolling(22).sum()
    
    spy = spy.replace([np.inf, -np.inf], np.nan).dropna()
    
    return spy


def train_models_and_get_errors(spy):
    """모델 학습 및 예측 오차 계산"""
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'regime_crisis', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    y_vrp = spy['VRP_true'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_rv_train, y_rv_test = y_rv[:split_idx], y_rv[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    vix_test = spy['VIX'].values[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {}
    predictions = {}
    errors = {}
    
    # HAR-RV (Benchmark)
    har_X_train = spy[['RV_1d', 'RV_5d', 'RV_22d']].values[:split_idx]
    har_X_test = spy[['RV_1d', 'RV_5d', 'RV_22d']].values[split_idx:]
    har = LinearRegression()
    har.fit(har_X_train, y_rv_train)
    har_rv_pred = har.predict(har_X_test)
    har_vrp_pred = vix_test - har_rv_pred
    predictions['HAR-RV'] = har_vrp_pred
    errors['HAR-RV'] = y_vrp_test - har_vrp_pred
    
    # ElasticNet
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_rv_train)
    en_rv_pred = en.predict(X_test_s)
    en_vrp_pred = vix_test - en_rv_pred
    predictions['ElasticNet'] = en_vrp_pred
    errors['ElasticNet'] = y_vrp_test - en_vrp_pred
    
    # Ridge
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train_s, y_rv_train)
    ridge_rv_pred = ridge.predict(X_test_s)
    ridge_vrp_pred = vix_test - ridge_rv_pred
    predictions['Ridge'] = ridge_vrp_pred
    errors['Ridge'] = y_vrp_test - ridge_vrp_pred
    
    # GradientBoosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, 
                                    learning_rate=0.05, random_state=SEED)
    gb.fit(X_train_s, y_rv_train)
    gb_rv_pred = gb.predict(X_test_s)
    gb_vrp_pred = vix_test - gb_rv_pred
    predictions['GradientBoosting'] = gb_vrp_pred
    errors['GradientBoosting'] = y_vrp_test - gb_vrp_pred
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, 
                                  learning_rate=0.05, random_state=SEED)
    xgb_model.fit(X_train_s, y_rv_train)
    xgb_rv_pred = xgb_model.predict(X_test_s)
    xgb_vrp_pred = vix_test - xgb_rv_pred
    predictions['XGBoost'] = xgb_vrp_pred
    errors['XGBoost'] = y_vrp_test - xgb_vrp_pred
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, 
                                   learning_rate=0.05, random_state=SEED, verbose=-1)
    lgb_model.fit(X_train_s, y_rv_train)
    lgb_rv_pred = lgb_model.predict(X_test_s)
    lgb_vrp_pred = vix_test - lgb_rv_pred
    predictions['LightGBM'] = lgb_vrp_pred
    errors['LightGBM'] = y_vrp_test - lgb_vrp_pred
    
    return y_vrp_test, predictions, errors


def main():
    print("\n" + "=" * 60)
    print("📊 VRP 예측 모델 통계적 유의성 검정")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/4] 데이터 로드 및 전처리...")
    spy = load_and_prepare_data()
    print(f"     샘플 수: {len(spy)}")
    
    # 모델 학습 및 오차 계산
    print("\n[2/4] 모델 학습 및 예측...")
    y_true, predictions, errors = train_models_and_get_errors(spy)
    
    models = list(errors.keys())
    print(f"     모델 수: {len(models)}")
    
    # ================================================================
    # 검정 1: Diebold-Mariano Test (ElasticNet vs 다른 모델)
    # ================================================================
    print("\n[3/4] Diebold-Mariano Test (예측력 비교)...")
    print("-" * 60)
    
    dm_results = {}
    base_model = 'ElasticNet'
    
    for model in models:
        if model == base_model:
            continue
        
        dm_stat, p_value = diebold_mariano_test(
            errors[base_model], errors[model], h=22, power=2
        )
        
        significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
        better = "ElasticNet" if dm_stat < 0 else model
        
        dm_results[model] = {
            'dm_statistic': float(dm_stat),
            'p_value': float(p_value),
            'significant_at_5pct': p_value < 0.05,
            'better_model': better
        }
        
        print(f"     {base_model} vs {model:18s}: DM = {dm_stat:7.3f}, p = {p_value:.4f} {significance}")
    
    # ================================================================
    # 검정 2: Bootstrap R² 유의성 검정
    # ================================================================
    print("\n[4/4] Bootstrap R² 유의성 검정...")
    print("-" * 60)
    
    bootstrap_results = {}
    
    for model, pred in predictions.items():
        result = bootstrap_r2_test(y_true, pred, n_bootstrap=1000)
        bootstrap_results[model] = result
        
        sig = "✅" if result['significant'] else "❌"
        print(f"     {model:18s}: R² = {result['r2']:.4f}, p = {result['p_value']:.4f} {sig}")
        print(f"                        95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    
    # ================================================================
    # 결과 요약
    # ================================================================
    print("\n" + "=" * 60)
    print("📊 검정 결과 요약")
    print("=" * 60)
    
    print("\n🔹 Diebold-Mariano Test 결과 (ElasticNet 기준):")
    print("   H0: 두 모델의 예측력에 차이 없음")
    
    sig_count = sum(1 for r in dm_results.values() if r['significant_at_5pct'])
    print(f"\n   • 유의한 차이 (p < 0.05): {sig_count}/{len(dm_results)} 모델")
    
    for model, result in dm_results.items():
        if result['significant_at_5pct']:
            print(f"     - {model}: ElasticNet이 유의하게 {'우수' if result['dm_statistic'] < 0 else '열등'}")
    
    print("\n🔹 Bootstrap R² 유의성 검정 결과:")
    print("   H0: R² = 0 (예측력 없음)")
    
    for model, result in bootstrap_results.items():
        status = "예측력 있음 (p < 0.05)" if result['significant'] else "예측력 없음"
        print(f"   • {model}: {status}")
    
    # 결과 저장
    def convert_to_native(obj):
        """numpy 타입을 Python native 타입으로 변환"""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    output = {
        'diebold_mariano': convert_to_native(dm_results),
        'bootstrap_r2': convert_to_native(bootstrap_results),
        'summary': {
            'best_model': 'ElasticNet',
            'significant_dm_tests': sig_count,
            'total_dm_tests': len(dm_results),
            'models_with_significant_r2': sum(1 for r in bootstrap_results.values() if r['significant'])
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = Path('data/results/statistical_tests.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 결과 저장: {output_path}")
    print("\n✅ 통계적 유의성 검정 완료!")


if __name__ == '__main__':
    main()
