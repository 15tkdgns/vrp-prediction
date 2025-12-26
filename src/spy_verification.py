"""
SPY R² = 0.59 결과 검증
========================
의심 사항:
- R² = 0.59는 VRP 예측에서 비현실적으로 높음
- 가능한 원인: 데이터 누출, 우연히 좋은 테스트 구간

검증 방법:
1. Walk-Forward CV (5-fold)
2. 다른 테스트 기간 (2023, 2024, COVID 등)
3. 특성별 중요도 확인
4. 예측 vs 실제 시각화
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window=22):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_data(start='2010-01-01', end='2025-01-01'):
    """데이터 준비"""
    spy = yf.download('SPY', start=start, end=end, progress=False)
    returns = spy['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_22d = calculate_rv(returns, 22)
    
    features = pd.DataFrame(index=spy.index)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_22d_lag5'] = rv_22d.shift(5)
    features['RV_22d_lag22'] = rv_22d.shift(22)
    
    vix = yf.download('^VIX', start=start, end=end, progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(spy.index).ffill().shift(1)
    
    features['RV_22d_future'] = rv_22d.shift(-22)
    
    return features.dropna()

# ============================================================================
# 검증 1: Walk-Forward CV
# ============================================================================

def verify_walk_forward_cv():
    """Walk-Forward CV로 검증"""
    print("\n[Verification 1] Walk-Forward CV (5-fold)")
    
    features = prepare_data('2010-01-01', '2025-01-01')
    
    feature_cols = ['RV_22d_lag1', 'RV_22d_lag5', 'RV_22d_lag22', 'VIX_lag1']
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    tscv = TimeSeriesSplit(n_splits=5)
    gap = 22
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Gap 적용
        train_idx = train_idx[:-gap]
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        if len(X_train) < 200 or len(X_test) < 50:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        y_train_log = np.log(y_train + 1)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train_log)
        
        test_pred = np.exp(model.predict(X_test_s)) - 1
        r2 = r2_score(y_test, test_pred)
        
        test_start = X_test.index[0].strftime('%Y-%m')
        test_end = X_test.index[-1].strftime('%Y-%m')
        
        results.append({
            'fold': fold,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_period': f'{test_start} ~ {test_end}',
            'r2': r2
        })
        
        print(f"  Fold {fold}: {test_start}~{test_end} | Train={len(X_train)}, Test={len(X_test)} | R2={r2:.4f}")
    
    avg_r2 = np.mean([r['r2'] for r in results])
    std_r2 = np.std([r['r2'] for r in results])
    
    print(f"\n  Average R2: {avg_r2:.4f} (+/- {std_r2:.4f})")
    
    return {'method': 'walk_forward_cv', 'avg_r2': avg_r2, 'std_r2': std_r2, 'folds': results}

# ============================================================================
# 검증 2: 특정 기간별 테스트
# ============================================================================

def verify_specific_periods():
    """특정 기간별 테스트"""
    print("\n[Verification 2] Specific Test Periods")
    
    features = prepare_data('2010-01-01', '2025-01-01')
    
    feature_cols = ['RV_22d_lag1', 'RV_22d_lag5', 'RV_22d_lag22', 'VIX_lag1']
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    # 테스트 기간 정의
    test_periods = [
        ('2020-02', '2020-12', 'COVID Period'),
        ('2021-01', '2021-12', '2021 (Post-COVID)'),
        ('2022-01', '2022-12', '2022 (High Vol)'),
        ('2023-01', '2023-12', '2023 (Recovery)'),
        ('2024-01', '2024-12', '2024 (Latest)'),
    ]
    
    results = []
    
    for start, end, name in test_periods:
        # 테스트 기간 이전 데이터로 학습
        train_end = pd.to_datetime(start) - pd.Timedelta(days=22)
        
        train_mask = X.index < train_end
        test_mask = (X.index >= start) & (X.index <= end)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        if len(X_train) < 200 or len(X_test) < 20:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        y_train_log = np.log(y_train + 1)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train_log)
        
        test_pred = np.exp(model.predict(X_test_s)) - 1
        r2 = r2_score(y_test, test_pred)
        
        results.append({
            'period': name,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'r2': r2
        })
        
        print(f"  {name}: Train={len(X_train)}, Test={len(X_test)} | R2={r2:.4f}")
    
    return {'method': 'specific_periods', 'results': results}

# ============================================================================
# 검증 3: 원본 결과 재현
# ============================================================================

def verify_original_result():
    """원본 결과 재현 (70/15/15 split)"""
    print("\n[Verification 3] Reproduce Original Result (70/15/15 split)")
    
    features = prepare_data('2010-01-01', '2025-01-01')
    
    feature_cols = ['RV_22d_lag1', 'RV_22d_lag5', 'RV_22d_lag22', 'VIX_lag1']
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    gap = 22
    n = len(X)
    train_end = int(n * 0.7) - gap
    val_end = int(n * 0.85) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end+gap:val_end]
    y_val = y.iloc[train_end+gap:val_end]
    X_test = X.iloc[val_end+gap:]
    y_test = y.iloc[val_end+gap:]
    
    print(f"  Train: {len(X_train)} ({X_train.index[0].strftime('%Y-%m')} ~ {X_train.index[-1].strftime('%Y-%m')})")
    print(f"  Val: {len(X_val)} ({X_val.index[0].strftime('%Y-%m')} ~ {X_val.index[-1].strftime('%Y-%m')})")
    print(f"  Test: {len(X_test)} ({X_test.index[0].strftime('%Y-%m')} ~ {X_test.index[-1].strftime('%Y-%m')})")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    y_train_log = np.log(y_train + 1)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train_log)
    
    val_pred = np.exp(model.predict(X_val_s)) - 1
    test_pred = np.exp(model.predict(X_test_s)) - 1
    
    val_r2 = r2_score(y_val, val_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\n  Val R2: {val_r2:.4f}")
    print(f"  Test R2: {test_r2:.4f}")
    
    # 테스트 기간 분석
    print(f"\n  Test period analysis:")
    print(f"    Test y mean: {y_test.mean():.4f}")
    print(f"    Test y std: {y_test.std():.4f}")
    print(f"    Pred mean: {test_pred.mean():.4f}")
    print(f"    Pred std: {test_pred.std():.4f}")
    
    return {'method': 'original_reproduction', 'val_r2': val_r2, 'test_r2': test_r2}

# ============================================================================
# 검증 4: 다른 분할 비율
# ============================================================================

def verify_different_splits():
    """다른 분할 비율로 테스트"""
    print("\n[Verification 4] Different Split Ratios")
    
    features = prepare_data('2010-01-01', '2025-01-01')
    
    feature_cols = ['RV_22d_lag1', 'RV_22d_lag5', 'RV_22d_lag22', 'VIX_lag1']
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    splits = [
        (0.6, 0.8, '60/20/20'),
        (0.7, 0.85, '70/15/15'),
        (0.8, 0.9, '80/10/10'),
    ]
    
    results = []
    
    for train_ratio, val_ratio, name in splits:
        gap = 22
        n = len(X)
        train_end = int(n * train_ratio) - gap
        val_end = int(n * val_ratio) - gap
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[val_end+gap:]
        y_test = y.iloc[val_end+gap:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        y_train_log = np.log(y_train + 1)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train_log)
        
        test_pred = np.exp(model.predict(X_test_s)) - 1
        r2 = r2_score(y_test, test_pred)
        
        results.append({'split': name, 'test_size': len(X_test), 'r2': r2})
        
        print(f"  {name}: Test={len(X_test)} | R2={r2:.4f}")
    
    return {'method': 'different_splits', 'results': results}

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("SPY R² = 0.59 Result Verification")
    print("="*80)
    
    all_results = {}
    
    # 1. Walk-Forward CV
    all_results['walk_forward'] = verify_walk_forward_cv()
    
    # 2. 특정 기간별 테스트
    all_results['specific_periods'] = verify_specific_periods()
    
    # 3. 원본 결과 재현
    all_results['original'] = verify_original_result()
    
    # 4. 다른 분할 비율
    all_results['different_splits'] = verify_different_splits()
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'SPY R2=0.59 Verification',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/spy_verification.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print("Verification Summary")
    print("="*80)
    
    # Walk-Forward CV 결과
    wf = all_results['walk_forward']
    print(f"\n1. Walk-Forward CV: {wf['avg_r2']:.4f} (+/- {wf['std_r2']:.4f})")
    
    # 기간별 결과
    print("\n2. Specific Period Results:")
    for r in all_results['specific_periods']['results']:
        print(f"   {r['period']}: R2={r['r2']:.4f}")
    
    # 결론
    avg_cv = wf['avg_r2']
    if avg_cv < 0.10:
        print(f"\n⚠️ 결론: Walk-Forward CV 평균 R²={avg_cv:.4f} → R²=0.59는 과대평가")
        print("   원인: 70/15/15 분할에서 테스트 기간이 우연히 예측하기 쉬운 구간")
    else:
        print(f"\n✅ Walk-Forward CV도 양호: {avg_cv:.4f}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
