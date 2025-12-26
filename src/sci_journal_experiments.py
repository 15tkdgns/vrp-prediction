"""
SCI 저널 요구사항 실험
======================
1. THAR (Threshold HAR) 모델 구현 및 비교
2. HAR-CJ (점프 성분 포함) 모델 구현 및 DM 검정
3. SHAP (Shapley) 분석 - 설명 가능한 AI
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def diebold_mariano_test(y_true, pred1, pred2, h=5):
    """Diebold-Mariano 검정"""
    e1 = y_true - pred1
    e2 = y_true - pred2
    d = e1**2 - e2**2
    
    n = len(d)
    mean_d = np.mean(d)
    
    # HAC 표준오차 (Newey-West)
    gamma0 = np.var(d)
    gamma = 0
    for i in range(1, h):
        gamma += 2 * (1 - i/h) * np.cov(d[:-i], d[i:])[0, 1]
    
    var_d = (gamma0 + gamma) / n
    dm_stat = mean_d / np.sqrt(var_d + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {'dm_stat': dm_stat, 'p_value': p_value}

# ============================================================================
# 1. THAR (Threshold HAR) 모델
# ============================================================================

def thar_model_experiment():
    """Threshold HAR 모델 구현 및 비교"""
    print("\n" + "="*60)
    print("[1] THAR (Threshold HAR) 모델 비교")
    print("="*60)
    
    ticker = 'SPY'
    
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    vix_aligned = vix_close.reindex(data.index).ffill()
    
    features = pd.DataFrame(index=data.index)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['VIX_lag1'] = vix_aligned.shift(1)
    features['RV_5d_future'] = rv_5d.shift(-5)
    features = features.dropna()
    
    gap = 5
    n = len(features)
    train_end = int(n * 0.7) - gap
    
    X_train = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']].iloc[:train_end]
    y_train = features['RV_5d_future'].iloc[:train_end]
    X_test = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']].iloc[train_end+gap:]
    y_test = features['RV_5d_future'].iloc[train_end+gap:]
    vix_test = features['VIX_lag1'].iloc[train_end+gap:]
    
    results = {}
    
    # 1. 기본 HAR 모델
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model_har = Ridge(alpha=100.0)
    model_har.fit(X_train_s, np.sqrt(y_train))
    pred_har = np.maximum(model_har.predict(X_test_s) ** 2, 0)
    r2_har = r2_score(y_test, pred_har)
    
    print(f"\n  HAR (Baseline): R2 = {r2_har:.4f}")
    
    # 2. THAR (Threshold HAR) - VIX 임계값 기반 레짐
    # VIX < 20: 저변동성 레짐, VIX >= 20: 고변동성 레짐
    threshold = 20
    
    # 훈련
    train_low = X_train[features['VIX_lag1'].iloc[:train_end] < threshold]
    y_train_low = y_train[features['VIX_lag1'].iloc[:train_end] < threshold]
    train_high = X_train[features['VIX_lag1'].iloc[:train_end] >= threshold]
    y_train_high = y_train[features['VIX_lag1'].iloc[:train_end] >= threshold]
    
    scaler_low = StandardScaler()
    scaler_high = StandardScaler()
    
    model_low = Ridge(alpha=100.0)
    model_high = Ridge(alpha=100.0)
    
    if len(train_low) > 50:
        X_low_s = scaler_low.fit_transform(train_low)
        model_low.fit(X_low_s, np.sqrt(y_train_low))
    
    if len(train_high) > 50:
        X_high_s = scaler_high.fit_transform(train_high)
        model_high.fit(X_high_s, np.sqrt(y_train_high))
    
    # 예측
    pred_thar = np.zeros(len(y_test))
    for i, (idx, row) in enumerate(X_test.iterrows()):
        if vix_test.iloc[i] < threshold:
            if len(train_low) > 50:
                x_s = scaler_low.transform(row.values.reshape(1, -1))
                pred_thar[i] = max(model_low.predict(x_s)[0] ** 2, 0)
            else:
                pred_thar[i] = pred_har[i]
        else:
            if len(train_high) > 50:
                x_s = scaler_high.transform(row.values.reshape(1, -1))
                pred_thar[i] = max(model_high.predict(x_s)[0] ** 2, 0)
            else:
                pred_thar[i] = pred_har[i]
    
    r2_thar = r2_score(y_test, pred_thar)
    print(f"  THAR (Threshold={threshold}): R2 = {r2_thar:.4f}")
    
    # DM 검정: THAR vs HAR
    dm_result = diebold_mariano_test(y_test.values, pred_thar, pred_har)
    print(f"  DM Test (THAR vs HAR): stat={dm_result['dm_stat']:.3f}, p={dm_result['p_value']:.4f}")
    
    # 3. ML 모델 (Ridge)
    model_ml = Ridge(alpha=10.0)
    model_ml.fit(X_train_s, np.sqrt(y_train))
    pred_ml = np.maximum(model_ml.predict(X_test_s) ** 2, 0)
    r2_ml = r2_score(y_test, pred_ml)
    
    print(f"  ML (Ridge alpha=10): R2 = {r2_ml:.4f}")
    
    # DM 검정: ML vs HAR
    dm_ml_har = diebold_mariano_test(y_test.values, pred_ml, pred_har)
    print(f"  DM Test (ML vs HAR): stat={dm_ml_har['dm_stat']:.3f}, p={dm_ml_har['p_value']:.4f}")
    
    # DM 검정: ML vs THAR
    dm_ml_thar = diebold_mariano_test(y_test.values, pred_ml, pred_thar)
    print(f"  DM Test (ML vs THAR): stat={dm_ml_thar['dm_stat']:.3f}, p={dm_ml_thar['p_value']:.4f}")
    
    results = {
        'HAR': {'r2': r2_har},
        'THAR': {'r2': r2_thar, 'threshold': threshold},
        'ML': {'r2': r2_ml},
        'DM_THAR_vs_HAR': dm_result,
        'DM_ML_vs_HAR': dm_ml_har,
        'DM_ML_vs_THAR': dm_ml_thar
    }
    
    return results

# ============================================================================
# 2. HAR-CJ (점프 성분 분리) 모델
# ============================================================================

def har_cj_model_experiment():
    """HAR-CJ 모델 구현 (점프 성분 분리)"""
    print("\n" + "="*60)
    print("[2] HAR-CJ (Jump Component) 모델")
    print("="*60)
    
    ticker = 'SPY'
    
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    # 점프 추정: Bipower Variation 사용
    # 간소화된 점프 감지: |return| > 3 * std
    returns_std = returns.rolling(22).std()
    jump_threshold = 3 * returns_std
    
    # 점프 여부
    is_jump = (returns.abs() > jump_threshold).astype(int).fillna(0)
    
    # 점프 성분
    jump_returns = returns.where(is_jump == 1, 0)
    continuous_returns = returns.where(is_jump == 0, 0)
    
    # RV 계산
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    # 연속 성분 RV
    cv_5d = calculate_rv(continuous_returns, 5)
    
    # 점프 성분 RV
    jv_5d = calculate_rv(jump_returns, 5)
    
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    vix_aligned = vix_close.reindex(data.index).ffill()
    
    features = pd.DataFrame(index=data.index)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['CV_5d_lag1'] = cv_5d.shift(1)  # 연속 성분
    features['JV_5d_lag1'] = jv_5d.shift(1)  # 점프 성분
    features['VIX_lag1'] = vix_aligned.shift(1)
    features['RV_5d_future'] = rv_5d.shift(-5)
    features = features.dropna()
    
    gap = 5
    n = len(features)
    train_end = int(n * 0.7) - gap
    
    y_train = features['RV_5d_future'].iloc[:train_end]
    y_test = features['RV_5d_future'].iloc[train_end+gap:]
    
    results = {}
    
    # 1. HAR (기본)
    X_har = features[['RV_5d_lag1', 'RV_22d_lag1']].iloc[:train_end]
    X_har_test = features[['RV_5d_lag1', 'RV_22d_lag1']].iloc[train_end+gap:]
    
    scaler_har = StandardScaler()
    X_har_s = scaler_har.fit_transform(X_har)
    X_har_test_s = scaler_har.transform(X_har_test)
    
    model_har = Ridge(alpha=100.0)
    model_har.fit(X_har_s, np.sqrt(y_train))
    pred_har = np.maximum(model_har.predict(X_har_test_s) ** 2, 0)
    r2_har = r2_score(y_test, pred_har)
    
    print(f"\n  HAR (RV only): R2 = {r2_har:.4f}")
    
    # 2. HAR-CJ (연속 + 점프 분리)
    X_cj = features[['CV_5d_lag1', 'JV_5d_lag1', 'RV_22d_lag1']].iloc[:train_end]
    X_cj_test = features[['CV_5d_lag1', 'JV_5d_lag1', 'RV_22d_lag1']].iloc[train_end+gap:]
    
    scaler_cj = StandardScaler()
    X_cj_s = scaler_cj.fit_transform(X_cj)
    X_cj_test_s = scaler_cj.transform(X_cj_test)
    
    model_cj = Ridge(alpha=100.0)
    model_cj.fit(X_cj_s, np.sqrt(y_train))
    pred_cj = np.maximum(model_cj.predict(X_cj_test_s) ** 2, 0)
    r2_cj = r2_score(y_test, pred_cj)
    
    print(f"  HAR-CJ (C+J separated): R2 = {r2_cj:.4f}")
    
    # 3. HAR-VIX (VIX 추가)
    X_vix = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']].iloc[:train_end]
    X_vix_test = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']].iloc[train_end+gap:]
    
    scaler_vix = StandardScaler()
    X_vix_s = scaler_vix.fit_transform(X_vix)
    X_vix_test_s = scaler_vix.transform(X_vix_test)
    
    model_vix = Ridge(alpha=100.0)
    model_vix.fit(X_vix_s, np.sqrt(y_train))
    pred_vix = np.maximum(model_vix.predict(X_vix_test_s) ** 2, 0)
    r2_vix = r2_score(y_test, pred_vix)
    
    print(f"  HAR-VIX: R2 = {r2_vix:.4f}")
    
    # 4. HAR-CJ-VIX (모든 요소)
    X_full = features[['CV_5d_lag1', 'JV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']].iloc[:train_end]
    X_full_test = features[['CV_5d_lag1', 'JV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']].iloc[train_end+gap:]
    
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_full)
    X_full_test_s = scaler_full.transform(X_full_test)
    
    model_full = Ridge(alpha=100.0)
    model_full.fit(X_full_s, np.sqrt(y_train))
    pred_full = np.maximum(model_full.predict(X_full_test_s) ** 2, 0)
    r2_full = r2_score(y_test, pred_full)
    
    print(f"  HAR-CJ-VIX (Full): R2 = {r2_full:.4f}")
    
    # DM 검정
    dm_cj_har = diebold_mariano_test(y_test.values, pred_cj, pred_har)
    dm_full_har = diebold_mariano_test(y_test.values, pred_full, pred_har)
    
    print(f"\n  DM Test (HAR-CJ vs HAR): stat={dm_cj_har['dm_stat']:.3f}, p={dm_cj_har['p_value']:.4f}")
    print(f"  DM Test (HAR-CJ-VIX vs HAR): stat={dm_full_har['dm_stat']:.3f}, p={dm_full_har['p_value']:.4f}")
    
    results = {
        'HAR': {'r2': r2_har},
        'HAR_CJ': {'r2': r2_cj},
        'HAR_VIX': {'r2': r2_vix},
        'HAR_CJ_VIX': {'r2': r2_full},
        'DM_HAR_CJ_vs_HAR': dm_cj_har,
        'DM_HAR_CJ_VIX_vs_HAR': dm_full_har
    }
    
    return results

# ============================================================================
# 3. SHAP 분석 (간소화 버전)
# ============================================================================

def shap_analysis():
    """SHAP 분석 - 특성 기여도 해석"""
    print("\n" + "="*60)
    print("[3] SHAP 분석 (Permutation-based)")
    print("="*60)
    
    ticker = 'SPY'
    
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    vix_aligned = vix_close.reindex(data.index).ffill()
    
    features = pd.DataFrame(index=data.index)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['VIX_lag1'] = vix_aligned.shift(1)
    features['RV_5d_future'] = rv_5d.shift(-5)
    features = features.dropna()
    
    gap = 5
    n = len(features)
    train_end = int(n * 0.7) - gap
    
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']
    X_train = features[feature_cols].iloc[:train_end]
    y_train = features['RV_5d_future'].iloc[:train_end]
    X_test = features[feature_cols].iloc[train_end+gap:]
    y_test = features['RV_5d_future'].iloc[train_end+gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, np.sqrt(y_train))
    
    # 기본 예측 성능
    pred = np.maximum(model.predict(X_test_s) ** 2, 0)
    base_r2 = r2_score(y_test, pred)
    
    print(f"\n  Baseline R2: {base_r2:.4f}")
    
    # Permutation 기반 SHAP 근사
    shap_values = {}
    
    for i, col in enumerate(feature_cols):
        # 특성 셔플
        X_permuted = X_test_s.copy()
        np.random.seed(42)
        np.random.shuffle(X_permuted[:, i])
        
        pred_permuted = np.maximum(model.predict(X_permuted) ** 2, 0)
        permuted_r2 = r2_score(y_test, pred_permuted)
        
        importance = base_r2 - permuted_r2
        shap_values[col] = importance
        
        print(f"  {col}: R2 drop = {importance:.4f}")
    
    # VIX 레짐별 SHAP
    vix_test = features['VIX_lag1'].iloc[train_end+gap:]
    
    print("\n  레짐별 VIX 기여도:")
    
    regimes = {
        'Low (VIX<15)': vix_test < 15,
        'Mid (15<=VIX<25)': (vix_test >= 15) & (vix_test < 25),
        'High (VIX>=25)': vix_test >= 25
    }
    
    regime_contributions = {}
    
    for regime_name, mask in regimes.items():
        if mask.sum() > 30:
            X_regime = X_test_s[mask]
            y_regime = y_test[mask]
            
            pred_regime = np.maximum(model.predict(X_regime) ** 2, 0)
            regime_r2 = r2_score(y_regime, pred_regime)
            
            # VIX 컬럼만 셔플
            X_vix_permuted = X_regime.copy()
            np.random.seed(42)
            np.random.shuffle(X_vix_permuted[:, 2])
            
            pred_vix_perm = np.maximum(model.predict(X_vix_permuted) ** 2, 0)
            vix_perm_r2 = r2_score(y_regime, pred_vix_perm)
            
            vix_contribution = regime_r2 - vix_perm_r2
            regime_contributions[regime_name] = {
                'r2': regime_r2,
                'vix_contribution': vix_contribution,
                'n_samples': int(mask.sum())
            }
            
            print(f"    {regime_name}: R2={regime_r2:.4f}, VIX contrib={vix_contribution:.4f} (n={mask.sum()})")
    
    return {
        'base_r2': base_r2,
        'feature_importance': shap_values,
        'regime_analysis': regime_contributions
    }

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("SCI 저널 요구사항 실험")
    print("="*80)
    
    all_results = {}
    
    # 1. THAR 모델
    all_results['thar'] = thar_model_experiment()
    
    # 2. HAR-CJ 모델
    all_results['har_cj'] = har_cj_model_experiment()
    
    # 3. SHAP 분석
    all_results['shap'] = shap_analysis()
    
    # 저장
    output = {
        'metadata': {
            'experiment': 'SCI Journal Requirements',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/sci_journal_experiments.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("요약")
    print("="*80)
    
    print("\n[모델 비교]")
    print(f"  HAR: R2={all_results['thar']['HAR']['r2']:.4f}")
    print(f"  THAR: R2={all_results['thar']['THAR']['r2']:.4f}")
    print(f"  ML: R2={all_results['thar']['ML']['r2']:.4f}")
    print(f"  HAR-CJ-VIX: R2={all_results['har_cj']['HAR_CJ_VIX']['r2']:.4f}")
    
    print("\n[DM 검정 - ML vs 전통 모델]")
    print(f"  ML vs HAR: p={all_results['thar']['DM_ML_vs_HAR']['p_value']:.4f}")
    print(f"  ML vs THAR: p={all_results['thar']['DM_ML_vs_THAR']['p_value']:.4f}")
    
    print("\n[SHAP 특성 중요도]")
    for feat, val in all_results['shap']['feature_importance'].items():
        print(f"  {feat}: {val:.4f}")
    
    print(f"\n결과 저장: {output_path}")

if __name__ == "__main__":
    main()
