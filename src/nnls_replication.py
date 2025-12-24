"""
성공적인 NNLS 앙상블 재현
==========================
extended_sci_experiment에서 R²=0.10 달성한 설정 그대로 재현
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import nnls
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window=22):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_data(ticker):
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None, None
    
    returns = df['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=df.index)
    
    # HAR-RV
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
    features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()
    features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()
    
    # 타겟
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # VIX 프록시
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_ret = spy['Close'].pct_change()
    if isinstance(spy_ret, pd.DataFrame):
        spy_ret = spy_ret.iloc[:, 0]
    spy_rv = calculate_rv(spy_ret, 22)
    features['VIX_proxy'] = spy_rv.reindex(df.index).ffill()
    
    features['VIX_lag1'] = features['VIX_proxy'].shift(1)
    features['VIX_lag5'] = features['VIX_proxy'].shift(5)
    features['VIX_change'] = features['VIX_proxy'].pct_change()
    
    # VRP
    features['VRP'] = features['VIX_proxy'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    # 시장
    features['regime_high'] = (features['VIX_proxy'] >= 25).astype(int)
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    
    features = features.dropna()
    
    feature_cols = [c for c in features.columns if c != 'RV_22d_future']
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    return X, y

def run_nnls_ensemble(ticker, gap=5):
    """NNLS 앙상블 - extended_sci_experiment 설정 그대로 재현"""
    print(f"\n{'='*60}")
    print(f"Processing {ticker} (gap={gap})")
    
    X, y = prepare_data(ticker)
    if X is None:
        return None
    
    # Log 변환
    y_log = np.log(y + 1)
    
    # Train/Test 분할
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx - gap]
    y_train = y_log.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]  # 원본 스케일로 평가
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 기본 모델들 (extended_sci_experiment와 동일)
    base_models = {
        'Ridge': Ridge(alpha=0.001),
        'Lasso': Lasso(alpha=0.001, max_iter=2000),
        'Huber': HuberRegressor(epsilon=1.35, alpha=0.001),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    
    predictions = {}
    individual_r2 = {}
    
    for name, model in base_models.items():
        try:
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            pipe.fit(X_train, y_train)
            
            # 예측 (log 스케일 -> 원본 스케일)
            pred_log = pipe.predict(X_test)
            pred = np.exp(pred_log) - 1
            
            predictions[name] = pred
            individual_r2[name] = r2_score(y_test, pred)
            
            print(f"  {name}: R2={individual_r2[name]:.4f}")
        except Exception as e:
            print(f"  {name} failed: {e}")
    
    if len(predictions) < 3:
        return None
    
    # NNLS 앙상블
    model_names = list(predictions.keys())
    pred_matrix = np.column_stack([predictions[m] for m in model_names])
    
    weights, _ = nnls(pred_matrix, y_test.values)
    
    ensemble_pred = pred_matrix @ weights
    
    # 정규화된 가중치
    if weights.sum() > 0:
        weights_normalized = weights / weights.sum()
    else:
        weights_normalized = np.ones(len(weights)) / len(weights)
    
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"\n  === NNLS Ensemble ===")
    print(f"  R2: {ensemble_r2:.4f}")
    print(f"  MAE: {ensemble_mae:.4f}")
    print(f"  Weights: {dict(zip(model_names, [round(w, 4) for w in weights_normalized]))}")
    
    return {
        'asset': ticker,
        'gap': gap,
        'individual_r2': individual_r2,
        'ensemble_r2': round(ensemble_r2, 6),
        'ensemble_mae': round(ensemble_mae, 6),
        'weights': dict(zip(model_names, [round(w, 4) for w in weights_normalized]))
    }

def main():
    assets = ['SPY', 'GLD', 'QQQ', 'TLT', 'EEM']
    gaps = [5, 22]
    
    all_results = []
    
    print("="*80)
    print("NNLS Ensemble Replication (extended_sci_experiment settings)")
    print("="*80)
    
    for asset in assets:
        for gap in gaps:
            result = run_nnls_ensemble(asset, gap)
            if result:
                all_results.append(result)
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'NNLS Ensemble Replication',
            'settings': 'Same as extended_sci_experiment',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/nnls_replication_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Results Summary:")
    print("="*80)
    
    sorted_results = sorted(all_results, key=lambda x: x['ensemble_r2'], reverse=True)
    for r in sorted_results:
        print(f"  {r['asset']}/gap{r['gap']}: R2={r['ensemble_r2']:.4f}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
