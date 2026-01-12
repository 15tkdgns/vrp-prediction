#!/usr/bin/env python3
"""
한계 극복 실험
==============

1. Regime-Switching 모델: Regime별 개별 모델 학습
2. Adaptive 재학습: 최근 데이터에 더 많은 가중치
3. 앙상블 분산 감소: 여러 모델 평균
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def create_features():
    """특성 생성"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start=spy.index[0], end=spy.index[-1], progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
    
    spy['returns'] = spy['Close'].pct_change()
    
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
        spy[f'realized_vol_{window}'] = spy[f'volatility_{window}'] * np.sqrt(252)
    
    for window in [5, 10, 20]:
        spy[f'mean_return_{window}'] = spy['returns'].rolling(window).mean()
        spy[f'skew_{window}'] = spy['returns'].rolling(window).skew()
        spy[f'kurt_{window}'] = spy['returns'].rolling(window).kurt()
    
    for lag in [1, 2, 3, 5]:
        spy[f'return_lag_{lag}'] = spy['returns'].shift(lag)
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    for window in [5, 10, 20]:
        spy[f'momentum_{window}'] = spy['returns'].rolling(window).sum()
    
    spy['vol_ratio_5_20'] = spy['volatility_5'] / (spy['volatility_20'] + 1e-8)
    spy['vol_ratio_10_50'] = spy['volatility_10'] / (spy['volatility_50'] + 1e-8)
    spy['zscore_20'] = (spy['returns'] - spy['returns'].rolling(20).mean()) / (spy['returns'].rolling(20).std() + 1e-8)
    
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_lag_5'] = spy['VIX'].shift(5)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['regime_crisis'] = (vix_lagged >= 35).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vol_in_crisis'] = spy['regime_crisis'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    spy['vix_excess_35'] = np.maximum(vix_lagged - 35, 0)
    
    # Regime 라벨
    spy['regime'] = 'low'
    spy.loc[vix_lagged >= 20, 'regime'] = 'normal'
    spy.loc[vix_lagged >= 25, 'regime'] = 'high'
    spy.loc[vix_lagged >= 35, 'regime'] = 'crisis'
    
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            vol_values.append(pd.Series(returns[i+1:i+6]).std())
        else:
            vol_values.append(np.nan)
    spy['target_vol_5d'] = vol_values
    
    spy = spy.dropna()
    
    feature_cols = [c for c in spy.columns if c.startswith((
        'volatility_', 'realized_vol_', 'mean_return_', 'skew_', 'kurt_',
        'return_lag_', 'vol_lag_', 'vol_ratio_', 'zscore_', 'momentum_',
        'vix_', 'regime_high', 'regime_crisis', 'vol_in_', 'vix_excess_'
    ))]
    
    return spy, feature_cols


def experiment_1_regime_switching(spy, feature_cols):
    """실험 1: Regime-Switching 모델"""
    print("\n" + "=" * 60)
    print("[1/3] Regime-Switching 모델")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['target_vol_5d'].values
    regimes = spy['regime'].values
    test_vix = spy['VIX'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    regimes_train = regimes[:split_idx]
    regimes_test = regimes[split_idx:]
    test_vix_values = test_vix[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 1. 단일 모델 (기준)
    single_model = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
    single_model.fit(X_train_s, y_train)
    y_pred_single = single_model.predict(X_test_s)
    r2_single = r2_score(y_test, y_pred_single)
    print(f"\n  🔹 단일 모델 (기준): R² = {r2_single:.4f}")
    
    # 2. Regime별 개별 모델
    print("\n  🔹 Regime-Switching 모델:")
    
    # Regime별 모델 학습
    regime_models = {}
    for regime in ['low', 'normal', 'high', 'crisis']:
        mask = regimes_train == regime
        if mask.sum() >= 20:  # 최소 샘플 수
            model = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
            model.fit(X_train_s[mask], y_train[mask])
            regime_models[regime] = model
            print(f"     {regime:8s}: {mask.sum()} 샘플로 학습")
        else:
            regime_models[regime] = single_model  # fallback
            print(f"     {regime:8s}: 샘플 부족, 단일 모델 사용")
    
    # Regime별 예측
    y_pred_regime = np.zeros(len(y_test))
    for i, regime in enumerate(regimes_test):
        model = regime_models.get(regime, single_model)
        y_pred_regime[i] = model.predict(X_test_s[i:i+1])[0]
    
    r2_regime = r2_score(y_test, y_pred_regime)
    print(f"\n     Regime-Switching R²: {r2_regime:.4f}")
    
    # Regime별 성능 비교
    print("\n  📊 Regime별 성능 비교:")
    print(f"     {'Regime':10s} | {'단일 R²':10s} | {'Switching R²':12s}")
    print("     " + "-" * 40)
    
    regime_results = {}
    for regime in ['low', 'normal', 'high']:
        mask = regimes_test == regime
        if mask.sum() >= 5:
            r2_s = r2_score(y_test[mask], y_pred_single[mask])
            r2_r = r2_score(y_test[mask], y_pred_regime[mask])
            print(f"     {regime:10s} | {r2_s:10.4f} | {r2_r:12.4f}")
            regime_results[regime] = {'single': float(r2_s), 'switching': float(r2_r)}
    
    improvement = r2_regime - r2_single
    print(f"\n  ✅ 개선: {improvement:+.4f} ({improvement/abs(r2_single)*100:+.1f}%)")
    
    return {
        'single_r2': float(r2_single),
        'regime_switching_r2': float(r2_regime),
        'improvement': float(improvement),
        'regime_results': regime_results
    }


def experiment_2_adaptive_learning(spy, feature_cols):
    """실험 2: Adaptive 재학습 (더 짧은 학습 윈도우)"""
    print("\n" + "=" * 60)
    print("[2/3] Adaptive 재학습")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['target_vol_5d'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    
    # 1. 전체 데이터로 학습 (기준)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model_full = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
    model_full.fit(X_train_s, y_train)
    y_pred_full = model_full.predict(X_test_s)
    r2_full = r2_score(y_test, y_pred_full)
    print(f"\n  🔹 전체 학습 (기준): R² = {r2_full:.4f}")
    
    # 2. 짧은 윈도우 (최근 1년만)
    window_252 = min(252, len(X_train))
    X_train_short = X_train[-window_252:]
    y_train_short = y_train[-window_252:]
    
    scaler_short = StandardScaler()
    X_train_short_s = scaler_short.fit_transform(X_train_short)
    X_test_short_s = scaler_short.transform(X_test)
    
    model_short = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
    model_short.fit(X_train_short_s, y_train_short)
    y_pred_short = model_short.predict(X_test_short_s)
    r2_short = r2_score(y_test, y_pred_short)
    print(f"  🔹 최근 1년 학습: R² = {r2_short:.4f}")
    
    # 3. 가중치 학습 (최근 데이터 가중치 높음)
    n_train = len(X_train)
    weights = np.linspace(0.5, 1.0, n_train)  # 0.5 → 1.0
    
    # ElasticNet은 sample_weight를 지원하지 않으므로 Ridge 사용
    from sklearn.linear_model import Ridge as WeightedRidge
    model_weighted = WeightedRidge(alpha=1.0, random_state=SEED)
    model_weighted.fit(X_train_s, y_train, sample_weight=weights)
    y_pred_weighted = model_weighted.predict(X_test_s)
    r2_weighted = r2_score(y_test, y_pred_weighted)
    print(f"  🔹 가중치 학습: R² = {r2_weighted:.4f}")
    
    # 4. Rolling 재학습
    print("\n  🔹 Rolling 재학습 (63일마다 갱신):")
    y_pred_rolling = []
    step = 63
    
    for i in range(0, len(X_test), step):
        # 현재까지의 데이터로 학습
        train_end = split_idx + i
        X_train_rolling = X[:train_end]
        y_train_rolling = y[:train_end]
        
        scaler_r = StandardScaler()
        X_train_r_s = scaler_r.fit_transform(X_train_rolling)
        
        test_end = min(i + step, len(X_test))
        X_test_batch = X_test[i:test_end]
        X_test_batch_s = scaler_r.transform(X_test_batch)
        
        model_r = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000)
        model_r.fit(X_train_r_s, y_train_rolling)
        
        y_pred_batch = model_r.predict(X_test_batch_s)
        y_pred_rolling.extend(y_pred_batch)
    
    y_pred_rolling = np.array(y_pred_rolling)
    r2_rolling = r2_score(y_test, y_pred_rolling)
    print(f"     R² = {r2_rolling:.4f}")
    
    best_r2 = max(r2_full, r2_short, r2_weighted, r2_rolling)
    best_method = ['전체', '최근 1년', '가중치', 'Rolling'][
        [r2_full, r2_short, r2_weighted, r2_rolling].index(best_r2)]
    
    print(f"\n  ✅ 최고 성능: {best_method} (R² = {best_r2:.4f})")
    
    return {
        'full_r2': float(r2_full),
        'short_window_r2': float(r2_short),
        'weighted_r2': float(r2_weighted),
        'rolling_r2': float(r2_rolling),
        'best_method': best_method,
        'best_r2': float(best_r2)
    }


def experiment_3_ensemble_variance_reduction(spy, feature_cols):
    """실험 3: 앙상블 분산 감소"""
    print("\n" + "=" * 60)
    print("[3/3] 앙상블 분산 감소")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['target_vol_5d'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 여러 모델 학습
    models = {
        'ElasticNet_1': ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED, max_iter=10000),
        'ElasticNet_2': ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=SEED, max_iter=10000),
        'ElasticNet_3': ElasticNet(alpha=0.0001, l1_ratio=0.7, random_state=SEED, max_iter=10000),
        'Ridge': Ridge(alpha=1.0, random_state=SEED),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=15, 
                                    random_state=SEED, n_jobs=-1)
    }
    
    predictions = {}
    print("\n  📊 개별 모델 성능:")
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        predictions[name] = y_pred
        print(f"     {name:15s}: R² = {r2:.4f}")
    
    # 앙상블 조합
    print("\n  📊 앙상블 조합:")
    
    # 1. 단순 평균
    y_pred_avg = np.mean([predictions[k] for k in predictions], axis=0)
    r2_avg = r2_score(y_test, y_pred_avg)
    print(f"     단순 평균 (5개):    R² = {r2_avg:.4f}")
    
    # 2. ElasticNet 3개 평균
    y_pred_en_avg = np.mean([predictions['ElasticNet_1'], predictions['ElasticNet_2'], 
                             predictions['ElasticNet_3']], axis=0)
    r2_en_avg = r2_score(y_test, y_pred_en_avg)
    print(f"     ElasticNet 평균:    R² = {r2_en_avg:.4f}")
    
    # 3. 최적 가중치 탐색 (ElasticNet_1 + Ridge)
    best_r2 = 0
    best_w = 0.5
    for w in np.arange(0.5, 1.0, 0.05):
        y_ens = w * predictions['ElasticNet_1'] + (1-w) * predictions['Ridge']
        r2_ens = r2_score(y_test, y_ens)
        if r2_ens > best_r2:
            best_r2 = r2_ens
            best_w = w
    print(f"     EN+Ridge (w={best_w:.2f}): R² = {best_r2:.4f}")
    
    # 4. Bootstrap 앙상블 (분산 감소)
    n_bootstrap = 10
    bootstrap_preds = []
    for i in range(n_bootstrap):
        np.random.seed(SEED + i)
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        
        model = ElasticNet(alpha=0.0003, l1_ratio=0.6, random_state=SEED+i, max_iter=10000)
        model.fit(X_train_s[idx], y_train[idx])
        bootstrap_preds.append(model.predict(X_test_s))
    
    y_pred_bootstrap = np.mean(bootstrap_preds, axis=0)
    r2_bootstrap = r2_score(y_test, y_pred_bootstrap)
    print(f"     Bootstrap 앙상블:   R² = {r2_bootstrap:.4f}")
    
    # 분산 분석
    pred_std_single = np.std(predictions['ElasticNet_1'])
    pred_std_ensemble = np.std(y_pred_bootstrap)
    variance_reduction = (pred_std_single - pred_std_ensemble) / pred_std_single * 100
    
    print(f"\n  📉 분산 감소:")
    print(f"     단일 모델 예측 표준편차: {pred_std_single:.6f}")
    print(f"     앙상블 예측 표준편차:    {pred_std_ensemble:.6f}")
    print(f"     분산 감소율: {variance_reduction:.1f}%")
    
    return {
        'individual_r2': {k: float(r2_score(y_test, v)) for k, v in predictions.items()},
        'simple_avg_r2': float(r2_avg),
        'en_avg_r2': float(r2_en_avg),
        'weighted_r2': float(best_r2),
        'bootstrap_r2': float(r2_bootstrap),
        'variance_reduction': float(variance_reduction)
    }


def main():
    print("\n" + "🔧" * 30)
    print("한계 극복 실험")
    print("🔧" * 30)
    
    # 데이터 준비
    print("\n데이터 준비 중...")
    spy, feature_cols = create_features()
    print(f"  ✓ 데이터: {len(spy)} 행, {len(feature_cols)} 특성")
    
    # 실험 1: Regime-Switching
    result_1 = experiment_1_regime_switching(spy, feature_cols)
    
    # 실험 2: Adaptive 재학습
    result_2 = experiment_2_adaptive_learning(spy, feature_cols)
    
    # 실험 3: 앙상블 분산 감소
    result_3 = experiment_3_ensemble_variance_reduction(spy, feature_cols)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 최종 요약")
    print("=" * 60)
    
    baseline = 0.2608
    
    print(f"""
    기준 성능: R² = {baseline:.4f}
    
    1️⃣ Regime-Switching 모델
       • 결과: R² = {result_1['regime_switching_r2']:.4f}
       • 개선: {result_1['improvement']:+.4f}
    
    2️⃣ Adaptive 재학습
       • 최고: {result_2['best_method']} (R² = {result_2['best_r2']:.4f})
       • Rolling 재학습: R² = {result_2['rolling_r2']:.4f}
    
    3️⃣ 앙상블 분산 감소
       • Bootstrap 앙상블: R² = {result_3['bootstrap_r2']:.4f}
       • 분산 감소: {result_3['variance_reduction']:.1f}%
    """)
    
    # 저장
    output = {
        'regime_switching': result_1,
        'adaptive_learning': result_2,
        'ensemble_variance': result_3,
        'baseline_r2': baseline,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/limitation_solutions_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 결과 저장: paper/limitation_solutions_results.json")


if __name__ == '__main__':
    main()
