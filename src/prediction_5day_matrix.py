"""
5일 예측 전체 자산 비교
========================
모든 자산 × 모든 모델 성능 매트릭스

자산: SPY, QQQ, EEM, GLD, TLT, IWM, XLE, XLK, XLF, FXI
모델: Ridge, Huber, GB, Stacking, Log Transform, Sqrt, Box-Cox
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
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

def run_all_models(ticker):
    """모든 모델 실험"""
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
    
    # 1. Ridge (baseline)
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, y_train)
    pred = np.maximum(model.predict(X_test_s), 0)
    results['Ridge'] = r2_score(y_test, pred)
    
    # 2. Huber
    try:
        model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
        model.fit(X_train_s, y_train)
        pred = np.maximum(model.predict(X_test_s), 0)
        results['Huber'] = r2_score(y_test, pred)
    except:
        results['Huber'] = np.nan
    
    # 3. GB
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train_s, y_train)
    pred = np.maximum(model.predict(X_test_s), 0)
    results['GB'] = r2_score(y_test, pred)
    
    # 4. Stacking
    try:
        estimators = [
            ('ridge', Ridge(alpha=100.0)),
            ('huber', HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)),
            ('gb', GradientBoostingRegressor(n_estimators=30, max_depth=2, learning_rate=0.05, random_state=42))
        ]
        stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=10.0), cv=3)
        stacking.fit(X_train_s, y_train)
        pred = np.maximum(stacking.predict(X_test_s), 0)
        results['Stacking'] = r2_score(y_test, pred)
    except:
        results['Stacking'] = np.nan
    
    # 5. Log Transform
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, np.log(y_train + 1))
    pred = np.exp(model.predict(X_test_s)) - 1
    pred = np.maximum(pred, 0)
    results['Log'] = r2_score(y_test, pred)
    
    # 6. Sqrt Transform
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, np.sqrt(y_train))
    pred = model.predict(X_test_s) ** 2
    pred = np.maximum(pred, 0)
    results['Sqrt'] = r2_score(y_test, pred)
    
    # 7. Box-Cox λ=0.25
    try:
        model = Ridge(alpha=100.0)
        lam = 0.25
        y_bc = (y_train ** lam - 1) / lam
        model.fit(X_train_s, y_bc)
        pred_bc = model.predict(X_test_s)
        pred = ((pred_bc * lam) + 1) ** (1/lam)
        pred = np.maximum(pred, 0)
        pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
        results['BoxCox'] = r2_score(y_test, pred)
    except:
        results['BoxCox'] = np.nan
    
    # Persistence (benchmark)
    results['Persistence'] = r2_score(y_test, X_test['RV_5d_lag1'].values)
    
    # 최고 모델 찾기 (Persistence 제외)
    model_results = {k: v for k, v in results.items() if k != 'Persistence' and not np.isnan(v)}
    if model_results:
        best = max(model_results.items(), key=lambda x: x[1])
        results['Best_Model'] = best[0]
        results['Best_R2'] = best[1]
    
    return results

def main():
    print("="*80)
    print("5-Day Full Asset × Model Comparison")
    print("="*80)
    
    assets = ['SPY', 'QQQ', 'EEM', 'GLD', 'TLT', 'IWM', 'XLE', 'XLK', 'XLF', 'FXI']
    
    all_results = {}
    
    for ticker in assets:
        print(f"\nProcessing {ticker}...")
        results = run_all_models(ticker)
        if results:
            all_results[ticker] = results
            best = results.get('Best_Model', 'N/A')
            best_r2 = results.get('Best_R2', 0)
            print(f"  Best: {best} (R²={best_r2:.4f})")
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': '5-Day Full Asset Comparison',
            'assets': assets,
            'models': ['Ridge', 'Huber', 'GB', 'Stacking', 'Log', 'Sqrt', 'BoxCox'],
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/5day_full_comparison.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 성능 매트릭스 출력
    print("\n" + "="*80)
    print("Performance Matrix (R² × 100)")
    print("="*80)
    
    models = ['Ridge', 'Huber', 'GB', 'Stacking', 'Log', 'Sqrt', 'BoxCox', 'Persistence']
    
    # 헤더
    header = f"{'Asset':<8}"
    for m in models:
        header += f"{m:<10}"
    header += "Best"
    print(header)
    print("-"*100)
    
    # 데이터
    for ticker in assets:
        if ticker in all_results:
            row = f"{ticker:<8}"
            for m in models:
                val = all_results[ticker].get(m, np.nan)
                if np.isnan(val):
                    row += f"{'N/A':<10}"
                else:
                    row += f"{val*100:>7.1f}   "
            row += f"{all_results[ticker].get('Best_Model', 'N/A')}"
            print(row)
    
    # 최고 성과 요약
    print("\n" + "="*80)
    print("Best Results by Asset")
    print("="*80)
    
    sorted_assets = sorted(all_results.items(), key=lambda x: x[1].get('Best_R2', -999), reverse=True)
    
    for i, (ticker, results) in enumerate(sorted_assets, 1):
        best = results.get('Best_Model', 'N/A')
        best_r2 = results.get('Best_R2', 0)
        persist = results.get('Persistence', 0)
        improve = best_r2 - persist if not np.isnan(persist) else 0
        print(f"  {i}. {ticker}: R²={best_r2:.4f} ({best}) | vs Persistence: +{improve:.4f}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
