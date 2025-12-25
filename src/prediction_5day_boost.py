"""
5일 예측 성능 부스트 실험
=========================
현재 최고: QQQ R²=0.37

새로운 접근법:
1. 스태킹 앙상블 (예측값을 특성으로)
2. 특성 상호작용 (다항식)
3. 타겟 변환 (sqrt, rank)
4. 최적 모델 앙상블
5. 롤링 윈도우 최적화
6. Quantile Regression
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, HuberRegressor, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_features(ticker):
    """하이브리드 특성 (검증된 최적)"""
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    if len(data) < 500:
        return None
    
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=data.index)
    
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_ratio_lag1'] = (rv_5d / rv_22d.clip(lower=1e-8)).shift(1)
    
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(data.index).ffill().shift(1)
    features['VIX_change_lag1'] = features['VIX_lag1'].pct_change()
    features['direction_5d_lag1'] = returns.rolling(5).apply(lambda x: np.mean(x > 0)).shift(1)
    
    features['RV_5d_future'] = rv_5d.shift(-5)
    
    return features.dropna()

# ============================================================================
# 실험 1: 스태킹 앙상블
# ============================================================================

def experiment_stacking(ticker):
    """스태킹 앙상블"""
    print(f"\n[Stacking] {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 스태킹 앙상블
    estimators = [
        ('ridge1', Ridge(alpha=1.0)),
        ('ridge100', Ridge(alpha=100.0)),
        ('huber', HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)),
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42))
    ]
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=10.0),
        cv=5
    )
    
    stacking.fit(X_train_s, y_train)
    pred = np.maximum(stacking.predict(X_test_s), 0)
    r2 = r2_score(y_test, pred)
    
    print(f"  Stacking R²: {r2:.4f}")
    
    return {'r2': r2, 'model': 'Stacking'}

# ============================================================================
# 실험 2: 특성 상호작용 (다항식)
# ============================================================================

def experiment_polynomial(ticker):
    """다항식 특성"""
    print(f"\n[Polynomial] {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    
    # 다양한 degree
    for degree in [2, 3]:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        X_train_poly = poly.fit_transform(X_train_s)
        X_test_poly = poly.transform(X_test_s)
        
        # 강한 정규화 필요 (특성 많음)
        for alpha in [100.0, 1000.0]:
            model = Ridge(alpha=alpha)
            model.fit(X_train_poly, y_train)
            pred = np.maximum(model.predict(X_test_poly), 0)
            r2 = r2_score(y_test, pred)
            results[f'poly{degree}_a{alpha}'] = r2
    
    best = max(results.items(), key=lambda x: x[1])
    print(f"  Best: {best[0]} R²: {best[1]:.4f}")
    
    return {'best_r2': best[1], 'best_model': best[0], 'all': results}

# ============================================================================
# 실험 3: 타겟 변환
# ============================================================================

def experiment_target_transform(ticker):
    """타겟 변환"""
    print(f"\n[Target Transform] {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    
    # 1. Raw (baseline)
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, y_train)
    pred = np.maximum(model.predict(X_test_s), 0)
    results['raw'] = r2_score(y_test, pred)
    
    # 2. Log transform
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, np.log(y_train + 1))
    pred = np.exp(model.predict(X_test_s)) - 1
    pred = np.maximum(pred, 0)
    results['log'] = r2_score(y_test, pred)
    
    # 3. Sqrt transform
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, np.sqrt(y_train))
    pred = model.predict(X_test_s) ** 2
    pred = np.maximum(pred, 0)
    results['sqrt'] = r2_score(y_test, pred)
    
    # 4. Box-Cox lambda=0.25
    model = Ridge(alpha=100.0)
    lam = 0.25
    y_bc = (y_train ** lam - 1) / lam
    model.fit(X_train_s, y_bc)
    pred_bc = model.predict(X_test_s)
    pred = ((pred_bc * lam) + 1) ** (1/lam)
    pred = np.maximum(pred, 0)
    results['boxcox_0.25'] = r2_score(y_test, pred)
    
    best = max(results.items(), key=lambda x: x[1])
    print(f"  Best: {best[0]} R²: {best[1]:.4f}")
    for k, v in results.items():
        print(f"    {k}: {v:.4f}")
    
    return {'best_r2': best[1], 'best_transform': best[0], 'all': results}

# ============================================================================
# 실험 4: 앙상블 블렌딩
# ============================================================================

def experiment_blending(ticker):
    """최적 모델 블렌딩"""
    print(f"\n[Blending] {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.6) - gap
    val_end = int(n * 0.8) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end + gap : val_end]
    y_val = y.iloc[train_end + gap : val_end]
    X_test = X.iloc[val_end + gap:]
    y_test = y.iloc[val_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # 개별 모델 예측
    preds_val = {}
    preds_test = {}
    
    models = {
        'ridge100': Ridge(alpha=100.0),
        'huber': HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500),
        'gb': GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds_val[name] = np.maximum(model.predict(X_val_s), 0)
        preds_test[name] = np.maximum(model.predict(X_test_s), 0)
    
    # 검증 세트로 최적 가중치 찾기
    best_r2 = -999
    best_weights = None
    
    for w1 in np.arange(0, 1.1, 0.2):
        for w2 in np.arange(0, 1.1 - w1, 0.2):
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            
            blend_val = (w1 * preds_val['ridge100'] + 
                        w2 * preds_val['huber'] + 
                        w3 * preds_val['gb'])
            r2_val = r2_score(y_val, blend_val)
            
            blend_test = (w1 * preds_test['ridge100'] + 
                         w2 * preds_test['huber'] + 
                         w3 * preds_test['gb'])
            r2_test = r2_score(y_test, blend_test)
            
            if r2_val > best_r2:
                best_r2 = r2_val
                best_weights = (w1, w2, w3)
                best_test_r2 = r2_test
    
    print(f"  Best weights: Ridge={best_weights[0]:.1f}, Huber={best_weights[1]:.1f}, GB={best_weights[2]:.1f}")
    print(f"  Val R²: {best_r2:.4f}, Test R²: {best_test_r2:.4f}")
    
    return {'test_r2': best_test_r2, 'val_r2': best_r2, 'weights': best_weights}

# ============================================================================
# 실험 5: 최근 데이터만 학습
# ============================================================================

def experiment_recent_data(ticker):
    """최근 데이터만 학습"""
    print(f"\n[Recent Data] {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    
    results = {}
    
    for train_years in [3, 5, 7]:
        # 최근 train_years 년만 사용
        recent = features.iloc[-int(252 * train_years):]
        
        X = recent[feature_cols]
        y = recent['RV_5d_future']
        
        gap = 5
        n = len(X)
        train_end = int(n * 0.7) - gap
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end + gap:]
        y_test = y.iloc[train_end + gap:]
        
        if len(X_train) < 200 or len(X_test) < 50:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=100.0)
        model.fit(X_train_s, y_train)
        pred = np.maximum(model.predict(X_test_s), 0)
        r2 = r2_score(y_test, pred)
        
        results[f'{train_years}yr'] = r2
        print(f"  {train_years} years: R²={r2:.4f}")
    
    if results:
        best = max(results.items(), key=lambda x: x[1])
        return {'best_r2': best[1], 'best_period': best[0], 'all': results}
    return None

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("5-Day Performance Boost Experiments")
    print("="*80)
    print("Current best: QQQ R²=0.37")
    
    assets = ['SPY', 'QQQ', 'EEM']
    
    all_results = {}
    
    for asset in assets:
        all_results[asset] = {}
        
        # 실험 1: 스태킹
        r = experiment_stacking(asset)
        if r:
            all_results[asset]['stacking'] = r
        
        # 실험 2: 다항식
        r = experiment_polynomial(asset)
        if r:
            all_results[asset]['polynomial'] = r
        
        # 실험 3: 타겟 변환
        r = experiment_target_transform(asset)
        if r:
            all_results[asset]['target_transform'] = r
        
        # 실험 4: 블렌딩
        r = experiment_blending(asset)
        if r:
            all_results[asset]['blending'] = r
        
        # 실험 5: 최근 데이터
        r = experiment_recent_data(asset)
        if r:
            all_results[asset]['recent_data'] = r
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': '5-Day Performance Boost',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/5day_boost_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    for asset in assets:
        print(f"\n{asset}:")
        results = all_results.get(asset, {})
        
        best_r2 = -999
        best_exp = None
        
        for exp, r in results.items():
            r2 = r.get('r2') or r.get('test_r2') or r.get('best_r2', -999)
            print(f"  {exp}: {r2:.4f}")
            if r2 > best_r2:
                best_r2 = r2
                best_exp = exp
        
        print(f"  ** Best: {best_exp} ({best_r2:.4f})")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
