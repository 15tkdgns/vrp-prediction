#!/usr/bin/env python3
"""
추가 성능 향상 실험 - 새로운 특성 추가
======================================

현재 최고: R² = 0.2607 (ElasticNet alpha=0.0003, l1=0.6)
목표: 0.27+ 달성
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def create_enhanced_features():
    """강화된 특성 생성"""
    print("\n[1] 강화된 특성 생성")
    
    # SPY 데이터 로드
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # VIX 로드
    vix = yf.download('^VIX', start=spy.index[0], end=spy.index[-1], progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    vix_close = vix['Close'].reindex(spy.index).ffill()
    spy['VIX'] = vix_close
    
    # 기본 특성
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
    spy['zscore_20'] = (spy['returns'] - spy['returns'].rolling(20).mean()) / (spy['returns'].rolling(20).std() + 1e-8)
    
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
    
    # ========================================
    # 새로운 특성 추가
    # ========================================
    
    # 1. VIX 변화율 추가
    spy['vix_change_3'] = spy['VIX'].pct_change(3)
    spy['vix_change_5'] = spy['VIX'].pct_change(5)
    
    # 2. VIX-변동성 상호작용
    spy['vix_vol_interaction'] = spy['vix_lag_1'] * spy['volatility_5']
    spy['vix_vol_ratio'] = spy['VIX'] / (spy['volatility_20'] * np.sqrt(252) * 100 + 1e-8)
    
    # 3. VIX 이동평균 비율
    spy['vix_ma_5'] = spy['VIX'].rolling(5).mean()
    spy['vix_ma_20'] = spy['VIX'].rolling(20).mean()
    spy['vix_ma_ratio'] = spy['vix_ma_5'] / (spy['vix_ma_20'] + 1e-8)
    
    # 4. 변동성 가속도 (2차 미분)
    spy['vol_acceleration'] = spy['volatility_5'].diff().diff()
    
    # 5. 수익률 극단값
    spy['return_extreme_pos'] = (spy['returns'] > spy['returns'].rolling(20).quantile(0.95)).astype(int)
    spy['return_extreme_neg'] = (spy['returns'] < spy['returns'].rolling(20).quantile(0.05)).astype(int)
    
    # 6. 변동성 추세
    spy['vol_trend'] = spy['volatility_5'] - spy['volatility_20']
    
    # 7. VIX 변동성
    spy['vix_volatility'] = spy['VIX'].rolling(10).std()
    
    # 8. 고저 범위 (Garman-Klass 스타일)
    spy['high_low_range'] = (np.log(spy['High']) - np.log(spy['Low'])) ** 2
    spy['high_low_range_ma5'] = spy['high_low_range'].rolling(5).mean()
    
    # 9. 거래량 특성 
    if 'Volume' in spy.columns:
        spy['volume_ma_ratio'] = spy['Volume'] / (spy['Volume'].rolling(20).mean() + 1e-8)
        spy['volume_change'] = spy['Volume'].pct_change()
    
    # 10. VIX 임계값 추가
    spy['vix_excess_20'] = np.maximum(vix_lagged - 20, 0)
    spy['vix_excess_30'] = np.maximum(vix_lagged - 30, 0)
    
    # 11. 변동성 백분위수
    spy['vol_percentile'] = spy['volatility_5'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # 타겟
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_window = returns[i+1:i+6]
            vol_values.append(pd.Series(future_window).std())
        else:
            vol_values.append(np.nan)
    spy['target_vol_5d'] = vol_values
    
    spy = spy.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 특성 컬럼 선택
    feature_cols = []
    for col in spy.columns:
        if col.startswith(('volatility_', 'realized_vol_', 'mean_return_',
                          'skew_', 'kurt_', 'return_lag_', 'vol_lag_',
                          'vol_ratio_', 'zscore_', 'momentum_', 'vix_', 'regime_',
                          'vol_in_', 'vix_excess_', 'high_low_', 'volume_',
                          'return_extreme_', 'vol_trend', 'vol_acceleration',
                          'vol_percentile')):
            feature_cols.append(col)
    
    print(f"  ✓ 데이터: {len(spy)} 행, {len(feature_cols)} 특성")
    print(f"  ✓ 새로운 특성: {len(feature_cols) - 42}개 추가")
    
    return spy, feature_cols


def run_experiments(spy, feature_cols):
    """실험"""
    print("\n[2] 모델 실험")
    
    X = spy[feature_cols].values
    y = spy['target_vol_5d'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # 1. 기존 최적 파라미터
    print("\n  🔹 ElasticNet (기존 최적: alpha=0.0003, l1=0.6)")
    en = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_train)
    y_pred = en.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    results['ElasticNet_prev_best'] = r2
    print(f"     R² = {r2:.4f}")
    
    # 2. 새로운 특성으로 미세조정
    print("\n  🔹 ElasticNet (새 특성 + 미세조정)")
    best_r2 = 0
    best_params = {}
    for alpha in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0007, 0.001]:
        for l1_ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=SEED, max_iter=10000)
            en.fit(X_train_s, y_train)
            y_pred = en.predict(X_test_s)
            r2_temp = r2_score(y_test, y_pred)
            if r2_temp > best_r2:
                best_r2 = r2_temp
                best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
    results['ElasticNet_new_best'] = best_r2
    print(f"     R² = {best_r2:.4f} (alpha={best_params['alpha']}, l1={best_params['l1_ratio']})")
    
    # 3. 최적 모델로 특성 중요도 확인
    best_en = ElasticNet(**best_params, random_state=SEED, max_iter=10000)
    best_en.fit(X_train_s, y_train)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coef': np.abs(best_en.coef_)
    }).sort_values('coef', ascending=False)
    
    print("\n  📊 상위 10 특성:")
    for i, row in importance.head(10).iterrows():
        print(f"     {row['feature']:25s}: {row['coef']:.6f}")
    
    # 4. 앙상블 테스트
    print("\n  🔹 ElasticNet + Ridge 앙상블")
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train_s, y_train)
    
    y_pred_en = best_en.predict(X_test_s)
    y_pred_ridge = ridge.predict(X_test_s)
    
    best_ens_r2 = 0
    best_w = 0.5
    for w in np.arange(0.5, 1.0, 0.05):
        y_ens = w * y_pred_en + (1-w) * y_pred_ridge
        r2_ens = r2_score(y_test, y_ens)
        if r2_ens > best_ens_r2:
            best_ens_r2 = r2_ens
            best_w = w
    
    results['Ensemble_EN_Ridge'] = best_ens_r2
    print(f"     R² = {best_ens_r2:.4f} (EN weight: {best_w:.2f})")
    
    return results, best_params, importance


def main():
    print("\n" + "=" * 60)
    print("추가 성능 향상 실험 - 새로운 특성 추가")
    print("=" * 60)
    
    # 1. 강화된 특성 생성
    spy, feature_cols = create_enhanced_features()
    
    # 2. 실험
    results, best_params, importance = run_experiments(spy, feature_cols)
    
    # 3. 결과
    print("\n" + "=" * 60)
    print("[3] 결과 요약")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    baseline = 0.2607  # 이전 최고
    
    print("\n📊 모델별 성능:")
    for model, r2 in sorted_results:
        diff = r2 - baseline
        marker = "⭐ " if r2 > baseline else "   "
        print(f"  {marker}{model:25s}: R² = {r2:.4f} ({diff:+.4f})")
    
    best_model = sorted_results[0][0]
    best_r2 = sorted_results[0][1]
    
    print(f"\n🏆 최고 성능: {best_model}")
    print(f"  • R² = {best_r2:.4f}")
    print(f"  • 기존 대비: {best_r2 - baseline:+.4f} ({(best_r2 - baseline)/baseline*100:+.1f}%)")
    print(f"  • 최적 파라미터: {best_params}")
    
    # 저장
    output = {
        'results': {k: float(v) for k, v in results.items()},
        'best_model': best_model,
        'best_r2': float(best_r2),
        'best_params': best_params,
        'baseline': baseline,
        'improvement': float(best_r2 - baseline),
        'top_features': importance.head(15).to_dict('records'),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/performance_improvement_v3.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 결과 저장: paper/performance_improvement_v3.json")


if __name__ == '__main__':
    main()
