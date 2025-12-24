"""
전체 모델 재실험 (22일 Gap 적용 + 최적화)
==========================================
모든 모델에 동일한 조건 적용:
- 22일 Gap
- Log 변환 + Winsorization
- HAR-RV 특성
- 상호작용 변수
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import mstats
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window=22):
    """실현 변동성 계산"""
    return returns.rolling(window=window).std() * np.sqrt(252) * 100

def prepare_features(df):
    """특성 생성 (HAR-RV + 상호작용)"""
    features = pd.DataFrame(index=df.index)
    returns = df['Close'].pct_change()
    
    # HAR-RV 구조
    features['RV_1d'] = returns.abs() * np.sqrt(252) * 100
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
    features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()
    features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()
    
    # 타겟: 22일 후 RV
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # VIX 프록시
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_returns = spy['Close'].pct_change()
    spy_vix = calculate_rv(spy_returns, 22)
    features['VIX'] = spy_vix.reindex(df.index).ffill()
    
    # VIX lag
    features['VIX_lag1'] = features['VIX'].shift(1)
    features['VIX_lag5'] = features['VIX'].shift(5)
    features['VIX_change'] = features['VIX'].pct_change()
    
    # VRP
    features['VRP'] = features['VIX'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['VRP_lag5'] = features['VRP'].shift(5)
    features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    # 시장 상태
    features['regime_high'] = (features['VIX'] >= 25).astype(int)
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    
    # 상호작용 변수
    features['VIX_x_return'] = features['VIX'] * features['return_5d']
    features['RV_ratio'] = features['RV_5d'] / (features['RV_22d'] + 0.01)
    features['RV_momentum'] = features['RV_22d'].pct_change(5)
    
    return features.dropna()

def run_experiment(ticker):
    """단일 자산 전체 모델 실험"""
    print(f"\n{'='*60}")
    print(f"Processing {ticker}...")
    
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None
    
    features = prepare_features(df)
    
    feature_cols = [
        'RV_1d', 'RV_5d', 'RV_22d', 
        'RV_daily_mean', 'RV_weekly_mean', 'RV_monthly_mean',
        'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
        'regime_high', 'return_5d', 'return_22d',
        'VIX_x_return', 'RV_ratio', 'RV_momentum'
    ]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    # Winsorization (상하위 1%)
    y_winsorized = pd.Series(mstats.winsorize(y, limits=[0.01, 0.01]), index=y.index)
    
    # Log 변환
    y_log = np.log1p(y_winsorized)
    
    # ===== 22일 Gap 적용 =====
    gap = 22
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx - gap]
    y_train = y_log.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test_log = y_log.iloc[split_idx:]
    y_test_original = y.iloc[split_idx:]
    
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}, Gap: {gap}d")
    
    # 모델 정의 (전체)
    models = {
        'ElasticNet_001': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=2000))
        ]),
        'ElasticNet_01': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000))
        ]),
        'Ridge_001': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=0.001, random_state=42))
        ]),
        'Lasso_001': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.001, random_state=42, max_iter=2000))
        ]),
        'Huber': Pipeline([
            ('scaler', StandardScaler()),
            ('model', HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=500))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42))
        ]),
        'MLP_32': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42, early_stopping=True))
        ]),
        'MLP_64': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=42, early_stopping=True))
        ]),
        'MLP_64_32': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True))
        ]),
        'SVR_rbf': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ]),
    }
    
    results = {}
    
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred_log = model.predict(X_test)
            
            # 역변환
            y_pred = np.expm1(y_pred_log)
            
            # 평가
            r2 = r2_score(y_test_original, y_pred)
            mae = mean_absolute_error(y_test_original, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
            
            # 방향 정확도
            actual_up = y_test_original.values > np.median(y_test_original.values)
            pred_up = y_pred > np.median(y_pred)
            direction_acc = (actual_up == pred_up).mean()
            
            results[model_name] = {
                'R2': round(r2, 4),
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'Direction': round(direction_acc * 100, 2)
            }
            
            print(f"  {model_name}: R2={r2:.4f}, MAE={mae:.2f}, Dir={direction_acc*100:.1f}%")
            
        except Exception as e:
            print(f"  {model_name} failed: {e}")
            results[model_name] = {'R2': None, 'MAE': None, 'RMSE': None, 'Direction': None}
    
    return results

def main():
    assets = ['SPY', 'GLD', 'QQQ', 'TLT', 'EEM']
    
    all_results = {}
    
    for asset in assets:
        result = run_experiment(asset)
        if result:
            all_results[asset] = result
    
    # 결과 저장
    output = {
        'results': all_results,
        'assets': assets,
        'settings': {
            'gap': 22,
            'transform': 'log + winsorize',
            'features': 18,
            'train_ratio': 0.8
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = 'data/results/full_model_experiment_gap22.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    
    # 요약
    print("\n=== Best Model per Asset ===")
    for asset in assets:
        if asset in all_results:
            best_r2 = -999
            best_model = ""
            for model, metrics in all_results[asset].items():
                if metrics.get('R2') is not None and metrics['R2'] > best_r2:
                    best_r2 = metrics['R2']
                    best_model = model
            print(f"  {asset}: {best_model} (R2={best_r2:.4f})")
    
    # 전체 평균
    print("\n=== Model Average (All Assets) ===")
    model_totals = {}
    for asset, models in all_results.items():
        for model, metrics in models.items():
            if metrics.get('R2') is not None:
                if model not in model_totals:
                    model_totals[model] = []
                model_totals[model].append(metrics['R2'])
    
    for model, values in sorted(model_totals.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f"  {model}: avg R2={np.mean(values):.4f}")

if __name__ == "__main__":
    main()
