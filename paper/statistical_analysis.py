#!/usr/bin/env python3
"""
논문용 통계 분석 및 표 생성
===========================

Diebold-Mariano 검정, 기술통계량 등 논문에 필요한 통계 분석 수행
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yfinance as yf
from pathlib import Path
import json

SEED = 42
np.random.seed(SEED)


def load_data():
    """데이터 로드"""
    print("=" * 60)
    print("[1] 데이터 로드")
    print("=" * 60)
    
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
    
    print(f"  ✓ SPY 데이터: {len(spy)} 행")
    return spy


def compute_descriptive_stats(spy):
    """기술통계량 계산"""
    print("\n" + "=" * 60)
    print("[2] 기술통계량")
    print("=" * 60)
    
    spy['returns'] = spy['Close'].pct_change() * 100  # 퍼센트
    spy['volatility_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252)
    
    spy = spy.dropna()
    
    # 기술통계량 테이블
    stats_df = pd.DataFrame({
        '변수': ['일별 수익률 (%)', '5일 실현변동성 (연율화)', 'VIX'],
        '평균': [
            spy['returns'].mean(),
            spy['volatility_5d'].mean(),
            spy['VIX'].mean()
        ],
        '표준편차': [
            spy['returns'].std(),
            spy['volatility_5d'].std(),
            spy['VIX'].std()
        ],
        '최소값': [
            spy['returns'].min(),
            spy['volatility_5d'].min(),
            spy['VIX'].min()
        ],
        '최대값': [
            spy['returns'].max(),
            spy['volatility_5d'].max(),
            spy['VIX'].max()
        ],
        '왜도': [
            stats.skew(spy['returns'].dropna()),
            stats.skew(spy['volatility_5d'].dropna()),
            stats.skew(spy['VIX'].dropna())
        ],
        '첨도': [
            stats.kurtosis(spy['returns'].dropna()),
            stats.kurtosis(spy['volatility_5d'].dropna()),
            stats.kurtosis(spy['VIX'].dropna())
        ]
    }).round(4)
    
    print("\n표 1: SPY 데이터 기술통계량 (2020-2024)")
    print(stats_df.to_string(index=False))
    
    return spy, stats_df


def create_features_and_target(spy):
    """특성 및 타겟 생성"""
    print("\n" + "=" * 60)
    print("[3] 특성 및 타겟 생성")
    print("=" * 60)
    
    spy['returns_raw'] = spy['Close'].pct_change()
    
    # 변동성 특성
    for w in [5, 10, 20, 50]:
        spy[f'volatility_{w}'] = spy['returns_raw'].rolling(w).std()
        spy[f'realized_vol_{w}'] = spy[f'volatility_{w}'] * np.sqrt(252)
    
    # VIX 특성
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_lag_5'] = spy['VIX'].shift(5)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
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
        spy[f'mean_return_{w}'] = spy['returns_raw'].rolling(w).mean()
        spy[f'skew_{w}'] = spy['returns_raw'].rolling(w).skew()
        spy[f'kurt_{w}'] = spy['returns_raw'].rolling(w).kurt()
    
    # 래그 변수
    for lag in [1, 2, 3, 5]:
        spy[f'return_lag_{lag}'] = spy['returns_raw'].shift(lag)
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    # 모멘텀
    for w in [5, 10, 20]:
        spy[f'momentum_{w}'] = spy['returns_raw'].rolling(w).sum()
    
    # 비율 및 Z-score
    spy['vol_ratio_5_20'] = spy['volatility_5'] / (spy['volatility_20'] + 1e-8)
    spy['zscore_20'] = (spy['returns_raw'] - spy['returns_raw'].rolling(20).mean()) / (spy['returns_raw'].rolling(20).std() + 1e-8)
    
    # HAR 특성
    returns_sq = spy['returns_raw'] ** 2
    spy['har_rv_d'] = returns_sq.shift(1)
    spy['har_rv_w'] = returns_sq.rolling(5).mean().shift(1)
    spy['har_rv_m'] = returns_sq.rolling(22).mean().shift(1)
    
    # 타겟: 5일 미래 변동성
    vol_values = []
    returns = spy['returns_raw'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_window = returns[i+1:i+6]
            vol_values.append(pd.Series(future_window).std())
        else:
            vol_values.append(np.nan)
    spy['target'] = vol_values
    
    spy = spy.dropna()
    print(f"  ✓ 최종 데이터: {len(spy)} 행")
    
    return spy


def train_all_models(spy):
    """모든 모델 학습 및 평가"""
    print("\n" + "=" * 60)
    print("[4] 모델 학습 및 평가")
    print("=" * 60)
    
    # 특성 컬럼
    feature_cols = [c for c in spy.columns if c.startswith((
        'volatility_', 'realized_vol_', 'vix_', 'regime_', 'vol_in_',
        'mean_return_', 'skew_', 'kurt_', 'return_lag_', 'vol_lag_',
        'momentum_', 'vol_ratio_', 'zscore_', 'har_'
    ))]
    
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
    predictions = {}
    
    # 1. ElasticNet + VIX + Regime
    print("\n  → ElasticNet + VIX + Regime")
    en = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_train)
    y_pred_en = en.predict(X_test_s)
    predictions['ElasticNet+VIX+Regime'] = y_pred_en
    results['ElasticNet+VIX+Regime'] = {
        'r2': r2_score(y_test, y_pred_en),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_en)),
        'mae': mean_absolute_error(y_test, y_pred_en),
        'params': len([c for c in en.coef_ if abs(c) > 1e-8])
    }
    print(f"    R² = {results['ElasticNet+VIX+Regime']['r2']:.4f}")
    
    # 2. HAR-RV
    print("  → HAR-RV")
    har_cols = ['har_rv_d', 'har_rv_w', 'har_rv_m']
    X_har = spy[[c for c in har_cols if c in spy.columns]].values
    X_har_train, X_har_test = X_har[:split_idx], X_har[split_idx:]
    X_har_train_s = scaler.fit_transform(X_har_train)
    X_har_test_s = scaler.transform(X_har_test)
    
    har = Ridge(alpha=1.0, random_state=SEED)
    har.fit(X_har_train_s, y_train)
    y_pred_har = har.predict(X_har_test_s)
    predictions['HAR-RV'] = y_pred_har
    results['HAR-RV'] = {
        'r2': r2_score(y_test, y_pred_har),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_har)),
        'mae': mean_absolute_error(y_test, y_pred_har),
        'params': 4
    }
    print(f"    R² = {results['HAR-RV']['r2']:.4f}")
    
    # 3. Ridge
    print("  → Ridge")
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train_s, y_train)
    y_pred_ridge = ridge.predict(X_test_s)
    predictions['Ridge'] = y_pred_ridge
    results['Ridge'] = {
        'r2': r2_score(y_test, y_pred_ridge),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'mae': mean_absolute_error(y_test, y_pred_ridge),
        'params': len(feature_cols)
    }
    print(f"    R² = {results['Ridge']['r2']:.4f}")
    
    # 4. GradientBoosting
    print("  → GradientBoosting")
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED)
    gb.fit(X_train_s, y_train)
    y_pred_gb = gb.predict(X_test_s)
    predictions['GradientBoosting'] = y_pred_gb
    results['GradientBoosting'] = {
        'r2': r2_score(y_test, y_pred_gb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'mae': mean_absolute_error(y_test, y_pred_gb),
        'params': 1000
    }
    print(f"    R² = {results['GradientBoosting']['r2']:.4f}")
    
    # 5. 기본 ElasticNet (VIX/Regime 없이)
    print("  → ElasticNet (baseline)")
    baseline_cols = [c for c in feature_cols if not c.startswith(('vix_', 'regime_', 'vol_in_'))]
    X_base = spy[baseline_cols].values
    X_base_train, X_base_test = X_base[:split_idx], X_base[split_idx:]
    X_base_train_s = scaler.fit_transform(X_base_train)
    X_base_test_s = scaler.transform(X_base_test)
    
    en_base = ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=SEED, max_iter=10000)
    en_base.fit(X_base_train_s, y_train)
    y_pred_base = en_base.predict(X_base_test_s)
    predictions['ElasticNet_baseline'] = y_pred_base
    results['ElasticNet_baseline'] = {
        'r2': r2_score(y_test, y_pred_base),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_base)),
        'mae': mean_absolute_error(y_test, y_pred_base),
        'params': 31
    }
    print(f"    R² = {results['ElasticNet_baseline']['r2']:.4f}")
    
    return results, predictions, y_test, feature_cols, en.coef_


def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano 검정
    H0: 두 모델의 예측 정확도가 동일하다
    """
    d = e1**2 - e2**2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    # Newey-West 표준오차 (autocorrelation correction)
    n = len(d)
    gamma0 = np.sum((d - d_mean)**2) / n
    
    gamma = 0
    for k in range(1, h):
        gamma += 2 * np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n
    
    var_d = (gamma0 + gamma) / n
    
    if var_d <= 0:
        var_d = d_var / n
    
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def perform_statistical_tests(predictions, y_test):
    """통계적 검정 수행"""
    print("\n" + "=" * 60)
    print("[5] Diebold-Mariano 검정")
    print("=" * 60)
    
    base_model = 'ElasticNet+VIX+Regime'
    base_pred = predictions[base_model]
    base_error = y_test - base_pred
    
    dm_results = []
    
    for model_name, pred in predictions.items():
        if model_name == base_model:
            continue
        
        error = y_test - pred
        dm_stat, p_value = diebold_mariano_test(base_error, error)
        
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = ''
        
        conclusion = '유의하게 우수' if dm_stat > 0 and p_value < 0.05 else '유의하지 않음'
        
        dm_results.append({
            'Model': model_name,
            'DM_stat': dm_stat,
            'p_value': p_value,
            'sig': sig,
            'conclusion': conclusion
        })
        
        print(f"  vs {model_name}: DM = {dm_stat:.2f}, p = {p_value:.4f} {sig}")
    
    return dm_results


def analyze_feature_importance(feature_cols, coef):
    """특성 중요도 분석"""
    print("\n" + "=" * 60)
    print("[6] 특성 중요도 (ElasticNet 계수)")
    print("=" * 60)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': np.abs(coef)
    }).sort_values('coefficient', ascending=False)
    
    print("\n상위 10개 특성:")
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['coefficient']:.6f}")
    
    return importance


def save_results(results, dm_results, stats_df):
    """결과 저장"""
    print("\n" + "=" * 60)
    print("[7] 결과 저장")
    print("=" * 60)
    
    output = {
        'model_performance': results,
        'dm_test': dm_results,
        'descriptive_stats': stats_df.to_dict('records')
    }
    
    output_path = Path('paper/statistical_analysis_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"  ✓ 결과 저장: {output_path}")


def main():
    """메인 함수"""
    print("\n" + "🔬" * 30)
    print("논문용 통계 분석")
    print("🔬" * 30)
    
    # 1. 데이터 로드
    spy = load_data()
    
    # 2. 기술통계량
    spy, stats_df = compute_descriptive_stats(spy)
    
    # 3. 특성 생성
    spy = create_features_and_target(spy)
    
    # 4. 모델 학습
    results, predictions, y_test, feature_cols, coef = train_all_models(spy)
    
    # 5. 통계적 검정
    dm_results = perform_statistical_tests(predictions, y_test)
    
    # 6. 특성 중요도
    importance = analyze_feature_importance(feature_cols, coef)
    
    # 7. 결과 저장
    save_results(results, dm_results, stats_df)
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("📊 최종 결과 요약")
    print("=" * 60)
    
    print("\n모델 성능:")
    for model, perf in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
        print(f"  {model:25s}: R² = {perf['r2']:.4f}, RMSE = {perf['rmse']:.6f}")
    
    print("\n✅ 통계 분석 완료!")


if __name__ == '__main__':
    main()
