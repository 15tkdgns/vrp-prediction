"""
5일 예측 추가 검증
==================
1. 특성 중요도 분석
2. 방향 정확도 상세 분석
3. 잔차 분석
4. 극단값 성능 분석
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_features(ticker):
    """하이브리드 특성"""
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
# 검증 1: 특성 중요도
# ============================================================================

def analyze_feature_importance(ticker, model_type='ridge', transform='sqrt'):
    """특성 중요도 분석"""
    print(f"\n[특성 중요도] {ticker}")
    
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
    
    if transform == 'sqrt':
        y_train_t = np.sqrt(y_train)
    elif transform == 'log':
        y_train_t = np.log(y_train + 1)
    else:
        y_train_t = y_train.values
    
    if model_type == 'ridge':
        model = Ridge(alpha=100.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.01, max_iter=3000)
    else:
        model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
    
    model.fit(X_train_s, y_train_t)
    
    # 1. 계수 중요도
    if hasattr(model, 'coef_'):
        coef_importance = dict(zip(feature_cols, np.abs(model.coef_)))
        print("  [계수 중요도]")
        for name, imp in sorted(coef_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name}: {imp:.4f}")
    
    # 2. 순열 중요도
    if transform == 'sqrt':
        pred = model.predict(X_test_s) ** 2
    elif transform == 'log':
        pred = np.exp(model.predict(X_test_s)) - 1
    else:
        pred = model.predict(X_test_s)
    
    pred = np.maximum(pred, 0)
    base_r2 = r2_score(y_test, pred)
    
    perm_importance = {}
    for i, col in enumerate(feature_cols):
        X_test_perm = X_test_s.copy()
        np.random.seed(42)
        X_test_perm[:, i] = np.random.permutation(X_test_perm[:, i])
        
        if transform == 'sqrt':
            pred_perm = model.predict(X_test_perm) ** 2
        elif transform == 'log':
            pred_perm = np.exp(model.predict(X_test_perm)) - 1
        else:
            pred_perm = model.predict(X_test_perm)
        
        pred_perm = np.maximum(pred_perm, 0)
        perm_r2 = r2_score(y_test, pred_perm)
        perm_importance[col] = base_r2 - perm_r2
    
    print("\n  [순열 중요도 (R² 감소량)]")
    for name, imp in sorted(perm_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name}: {imp:.4f}")
    
    return {'coef_importance': coef_importance if hasattr(model, 'coef_') else {}, 
            'perm_importance': perm_importance}

# ============================================================================
# 검증 2: 방향 정확도 상세 분석
# ============================================================================

def analyze_direction_accuracy(ticker, model_type='ridge', transform='sqrt'):
    """방향 정확도 상세 분석"""
    print(f"\n[방향 정확도] {ticker}")
    
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
    
    if transform == 'sqrt':
        y_train_t = np.sqrt(y_train)
    elif transform == 'log':
        y_train_t = np.log(y_train + 1)
    else:
        y_train_t = y_train.values
    
    if model_type == 'ridge':
        model = Ridge(alpha=100.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.01, max_iter=3000)
    else:
        model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
    
    model.fit(X_train_s, y_train_t)
    
    if transform == 'sqrt':
        pred = model.predict(X_test_s) ** 2
    elif transform == 'log':
        pred = np.exp(model.predict(X_test_s)) - 1
    else:
        pred = model.predict(X_test_s)
    
    pred = np.maximum(pred, 0)
    
    # 방향 분석
    rv_lag = X_test['RV_5d_lag1'].values
    actual_direction = (y_test.values > rv_lag).astype(int)  # 1=상승, 0=하락
    pred_direction = (pred > rv_lag).astype(int)
    
    # 전체 정확도
    overall_acc = (actual_direction == pred_direction).mean()
    
    # 상승/하락별 정확도
    up_mask = actual_direction == 1
    down_mask = actual_direction == 0
    
    up_acc = (pred_direction[up_mask] == 1).mean() if up_mask.sum() > 0 else 0
    down_acc = (pred_direction[down_mask] == 0).mean() if down_mask.sum() > 0 else 0
    
    print(f"  전체 방향 정확도: {overall_acc:.2%}")
    print(f"  상승 예측 정확도: {up_acc:.2%} (실제 상승 {up_mask.sum()}건)")
    print(f"  하락 예측 정확도: {down_acc:.2%} (실제 하락 {down_mask.sum()}건)")
    
    # 예측 확신도별 정확도
    pred_change = (pred - rv_lag) / (rv_lag + 1e-8)
    
    print("\n  [예측 변화폭별 정확도]")
    for threshold in [0.05, 0.10, 0.20]:
        strong_mask = np.abs(pred_change) > threshold
        if strong_mask.sum() > 0:
            strong_acc = (actual_direction[strong_mask] == pred_direction[strong_mask]).mean()
            print(f"    변화 >{threshold:.0%}: {strong_acc:.2%} ({strong_mask.sum()}건)")
    
    return {
        'overall_accuracy': overall_acc,
        'up_accuracy': up_acc,
        'down_accuracy': down_acc,
        'up_count': int(up_mask.sum()),
        'down_count': int(down_mask.sum())
    }

# ============================================================================
# 검증 3: 잔차 분석
# ============================================================================

def analyze_residuals(ticker, model_type='ridge', transform='sqrt'):
    """잔차 분석"""
    print(f"\n[잔차 분석] {ticker}")
    
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
    
    if transform == 'sqrt':
        y_train_t = np.sqrt(y_train)
    elif transform == 'log':
        y_train_t = np.log(y_train + 1)
    else:
        y_train_t = y_train.values
    
    if model_type == 'ridge':
        model = Ridge(alpha=100.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.01, max_iter=3000)
    else:
        model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
    
    model.fit(X_train_s, y_train_t)
    
    if transform == 'sqrt':
        pred = model.predict(X_test_s) ** 2
    elif transform == 'log':
        pred = np.exp(model.predict(X_test_s)) - 1
    else:
        pred = model.predict(X_test_s)
    
    pred = np.maximum(pred, 0)
    
    # 잔차 계산
    residuals = y_test.values - pred
    
    # 기본 통계
    print(f"  잔차 평균: {residuals.mean():.4f}")
    print(f"  잔차 표준편차: {residuals.std():.4f}")
    print(f"  MAE: {np.abs(residuals).mean():.4f}")
    
    # VIX 레짐별 성능
    vix_values = X_test['VIX_lag1'].values
    
    print("\n  [VIX 레짐별 R²]")
    for threshold in [15, 20, 25, 30]:
        low_mask = vix_values < threshold
        high_mask = vix_values >= threshold
        
        if low_mask.sum() > 20 and high_mask.sum() > 20:
            low_r2 = r2_score(y_test.values[low_mask], pred[low_mask])
            high_r2 = r2_score(y_test.values[high_mask], pred[high_mask])
            print(f"    VIX<{threshold}: R²={low_r2:.4f} ({low_mask.sum()}건)")
            print(f"    VIX>={threshold}: R²={high_r2:.4f} ({high_mask.sum()}건)")
    
    return {
        'residual_mean': residuals.mean(),
        'residual_std': residuals.std(),
        'mae': np.abs(residuals).mean()
    }

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("5일 예측 추가 검증")
    print("="*80)
    
    # 최적 설정
    configs = {
        'SPY': ('ridge', 'sqrt'),
        'QQQ': ('lasso', 'log'),
        'XLK': ('ridge', 'raw'),
        'XLF': ('huber', 'log')
    }
    
    all_results = {}
    
    for ticker, (model_type, transform) in configs.items():
        print(f"\n{'='*60}")
        print(f"{ticker} (model={model_type}, transform={transform})")
        print(f"{'='*60}")
        
        all_results[ticker] = {}
        
        # 1. 특성 중요도
        importance = analyze_feature_importance(ticker, model_type, transform)
        if importance:
            all_results[ticker]['importance'] = importance
        
        # 2. 방향 정확도
        direction = analyze_direction_accuracy(ticker, model_type, transform)
        if direction:
            all_results[ticker]['direction'] = direction
        
        # 3. 잔차 분석
        residuals = analyze_residuals(ticker, model_type, transform)
        if residuals:
            all_results[ticker]['residuals'] = residuals
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': '5-Day Additional Verification',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/5day_additional_verification.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("요약")
    print("="*80)
    
    print(f"\n{'자산':<8} {'방향정확도':>12} {'상승정확도':>12} {'하락정확도':>12} {'MAE':>10}")
    print("-"*60)
    
    for ticker, result in all_results.items():
        if 'direction' in result:
            d = result['direction']
            mae = result.get('residuals', {}).get('mae', 0)
            print(f"{ticker:<8} {d['overall_accuracy']:>12.2%} {d['up_accuracy']:>12.2%} {d['down_accuracy']:>12.2%} {mae:>10.4f}")
    
    print(f"\n결과 저장: {output_path}")

if __name__ == "__main__":
    main()
