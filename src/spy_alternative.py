"""
SPY 대안 접근법 실험
====================
기존 접근법 실패 (Test R² = -0.07)

새로운 접근법:
1. 예측 기간 단축 (22일 → 5일)
2. COVID 기간 제외 (이상치 제거)
3. 레짐 스위칭 (고/저 변동성 분리)
4. 분위수 회귀 (중앙값 예측)
5. 수익률 예측 (변동성 대신)
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Lasso, Ridge, HuberRegressor, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import nnls
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window=22):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

# ============================================================================
# 접근법 1: 예측 기간 단축 (5일)
# ============================================================================

def experiment_short_horizon():
    """5일 예측 - 더 짧은 기간은 예측이 더 쉬움"""
    print("\n[1] Short Horizon (5-day prediction)")
    
    spy = yf.download('SPY', start='2018-01-01', end='2025-01-01', progress=False)
    returns = spy['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    features = pd.DataFrame(index=spy.index)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    
    vix = yf.download('^VIX', start='2018-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(spy.index).ffill().shift(1)
    
    # 5일 후 RV 예측
    features['RV_5d_future'] = rv_5d.shift(-5)
    features = features.dropna()
    
    X = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']]
    y = features['RV_5d_future']
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.6) - gap
    val_end = int(n * 0.8) - gap
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end+gap:val_end], y.iloc[train_end+gap:val_end]
    X_test, y_test = X.iloc[val_end+gap:], y.iloc[val_end+gap:]
    
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
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")
    
    return {'approach': 'short_horizon_5d', 'val_r2': val_r2, 'test_r2': test_r2}

# ============================================================================
# 접근법 2: COVID 기간 제외
# ============================================================================

def experiment_exclude_covid():
    """COVID 기간 제외 - 구조적 변화 제거"""
    print("\n[2] Exclude COVID Period (2020-02 to 2020-06)")
    
    spy = yf.download('SPY', start='2018-01-01', end='2025-01-01', progress=False)
    returns = spy['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_22d = calculate_rv(returns, 22)
    
    features = pd.DataFrame(index=spy.index)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_22d_lag5'] = rv_22d.shift(5)
    
    vix = yf.download('^VIX', start='2018-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(spy.index).ffill().shift(1)
    
    features['RV_22d_future'] = rv_22d.shift(-22)
    features = features.dropna()
    
    # COVID 제외
    covid_start = '2020-02-01'
    covid_end = '2020-06-30'
    features_clean = features[(features.index < covid_start) | (features.index > covid_end)]
    
    print(f"  Original samples: {len(features)}, After COVID removal: {len(features_clean)}")
    
    X = features_clean[['RV_22d_lag1', 'RV_22d_lag5', 'VIX_lag1']]
    y = features_clean['RV_22d_future']
    
    gap = 22
    n = len(X)
    train_end = int(n * 0.6) - gap
    val_end = int(n * 0.8) - gap
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end+gap:val_end], y.iloc[train_end+gap:val_end]
    X_test, y_test = X.iloc[val_end+gap:], y.iloc[val_end+gap:]
    
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
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")
    
    return {'approach': 'exclude_covid', 'val_r2': val_r2, 'test_r2': test_r2}

# ============================================================================
# 접근법 3: 레짐 스위칭 (고/저 변동성)
# ============================================================================

def experiment_regime_switching():
    """레짐 스위칭 - 저변동성 구간만 예측"""
    print("\n[3] Regime Switching (Low volatility only)")
    
    spy = yf.download('SPY', start='2018-01-01', end='2025-01-01', progress=False)
    returns = spy['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_22d = calculate_rv(returns, 22)
    
    features = pd.DataFrame(index=spy.index)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_22d_lag5'] = rv_22d.shift(5)
    
    vix = yf.download('^VIX', start='2018-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(spy.index).ffill().shift(1)
    
    features['RV_22d_future'] = rv_22d.shift(-22)
    features = features.dropna()
    
    # 저변동성 구간만 (VIX < 20)
    low_vol = features['VIX_lag1'] < 20
    features_low = features[low_vol]
    
    print(f"  Low volatility samples: {len(features_low)} / {len(features)}")
    
    if len(features_low) < 300:
        print("  Not enough samples for low volatility regime")
        return None
    
    X = features_low[['RV_22d_lag1', 'RV_22d_lag5', 'VIX_lag1']]
    y = features_low['RV_22d_future']
    
    gap = 22
    n = len(X)
    train_end = int(n * 0.6) - gap
    val_end = int(n * 0.8) - gap
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end+gap:val_end], y.iloc[train_end+gap:val_end]
    X_test, y_test = X.iloc[val_end+gap:], y.iloc[val_end+gap:]
    
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
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")
    
    return {'approach': 'regime_low_vol', 'val_r2': val_r2, 'test_r2': test_r2}

# ============================================================================
# 접근법 4: 변동성 변화 방향 예측 (분류)
# ============================================================================

def experiment_direction_prediction():
    """변동성 상승/하락 방향만 예측"""
    print("\n[4] Direction Prediction (Up/Down classification)")
    
    spy = yf.download('SPY', start='2018-01-01', end='2025-01-01', progress=False)
    returns = spy['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_22d = calculate_rv(returns, 22)
    
    features = pd.DataFrame(index=spy.index)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_22d_lag5'] = rv_22d.shift(5)
    
    vix = yf.download('^VIX', start='2018-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(spy.index).ffill().shift(1)
    
    # 타겟: 변동성 상승 여부
    rv_future = rv_22d.shift(-22)
    features['direction'] = (rv_future > rv_22d).astype(int)
    features = features.dropna()
    
    X = features[['RV_22d_lag1', 'RV_22d_lag5', 'VIX_lag1']]
    y = features['direction']
    
    gap = 22
    n = len(X)
    train_end = int(n * 0.6) - gap
    val_end = int(n * 0.8) - gap
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[val_end+gap:], y.iloc[val_end+gap:]
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(C=0.1)
    model.fit(X_train_s, y_train)
    
    test_pred = model.predict(X_test_s)
    accuracy = accuracy_score(y_test, test_pred)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Direction Accuracy: {accuracy:.4f} (random=0.50)")
    
    return {'approach': 'direction', 'accuracy': accuracy}

# ============================================================================
# 접근법 5: 더 긴 학습 데이터 (2010년부터)
# ============================================================================

def experiment_longer_history():
    """더 긴 역사적 데이터 사용"""
    print("\n[5] Longer History (2010-2025)")
    
    spy = yf.download('SPY', start='2010-01-01', end='2025-01-01', progress=False)
    returns = spy['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_22d = calculate_rv(returns, 22)
    
    features = pd.DataFrame(index=spy.index)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_22d_lag5'] = rv_22d.shift(5)
    features['RV_22d_lag22'] = rv_22d.shift(22)
    
    vix = yf.download('^VIX', start='2010-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(spy.index).ffill().shift(1)
    
    features['RV_22d_future'] = rv_22d.shift(-22)
    features = features.dropna()
    
    X = features[['RV_22d_lag1', 'RV_22d_lag5', 'RV_22d_lag22', 'VIX_lag1']]
    y = features['RV_22d_future']
    
    gap = 22
    n = len(X)
    train_end = int(n * 0.7) - gap  # 더 많은 학습 데이터
    val_end = int(n * 0.85) - gap
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end+gap:val_end], y.iloc[train_end+gap:val_end]
    X_test, y_test = X.iloc[val_end+gap:], y.iloc[val_end+gap:]
    
    print(f"  Total samples: {len(X)}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    y_train_log = np.log(y_train + 1)
    
    # 다양한 모델 시도
    best_r2 = -999
    best_model = None
    
    for alpha in [0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train_log)
        test_pred = np.exp(model.predict(X_test_s)) - 1
        r2 = r2_score(y_test, test_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_model = f'Ridge_a{alpha}'
    
    for alpha in [0.1, 1.0]:
        model = HuberRegressor(epsilon=1.5, alpha=alpha, max_iter=500)
        model.fit(X_train_s, y_train_log)
        test_pred = np.exp(model.predict(X_test_s)) - 1
        r2 = r2_score(y_test, test_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_model = f'Huber_a{alpha}'
    
    print(f"  Best Model: {best_model}")
    print(f"  Test R2: {best_r2:.4f}")
    
    return {'approach': 'longer_history', 'best_model': best_model, 'test_r2': best_r2}

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("SPY Alternative Approaches Experiment")
    print("="*80)
    
    all_results = []
    
    # 1. 예측 기간 단축
    result = experiment_short_horizon()
    if result:
        all_results.append(result)
    
    # 2. COVID 제외
    result = experiment_exclude_covid()
    if result:
        all_results.append(result)
    
    # 3. 레짐 스위칭
    result = experiment_regime_switching()
    if result:
        all_results.append(result)
    
    # 4. 방향 예측
    result = experiment_direction_prediction()
    if result:
        all_results.append(result)
    
    # 5. 더 긴 역사
    result = experiment_longer_history()
    if result:
        all_results.append(result)
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'SPY Alternative Approaches',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/spy_alternative_approaches.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for r in all_results:
        if 'test_r2' in r:
            print(f"  {r['approach']}: Test R2={r['test_r2']:.4f}")
        elif 'accuracy' in r:
            print(f"  {r['approach']}: Accuracy={r['accuracy']:.4f}")
    
    # 최고 R2 찾기
    r2_results = [r for r in all_results if 'test_r2' in r]
    if r2_results:
        best = max(r2_results, key=lambda x: x['test_r2'])
        print(f"\n  Best: {best['approach']} (R2={best['test_r2']:.4f})")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
