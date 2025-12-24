"""
다중 자산 x 다중 모델 성능 비교 실험
=====================================
자산: SPY, GLD, TLT, QQQ, EEM
모델: ElasticNet, Ridge, Lasso, RandomForest, GradientBoosting
지표: R2, MAE, RMSE, Direction Accuracy
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window=22):
    """실현 변동성 계산"""
    return returns.rolling(window=window).std() * np.sqrt(252) * 100

def calculate_vix_proxy(data):
    """VIX 프록시 (SPY 기반)"""
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy['VIX'] = calculate_rv(spy['Close'].pct_change(), 22)
    return spy['VIX'].reindex(data.index).ffill()

def prepare_features(df):
    """특성 생성"""
    features = pd.DataFrame(index=df.index)
    
    # 실현 변동성
    returns = df['Close'].pct_change()
    features['RV_1d'] = returns.abs() * np.sqrt(252) * 100
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    
    # 타겟: 22일 후 RV
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # VIX 프록시
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_returns = spy['Close'].pct_change()
    spy_vix = calculate_rv(spy_returns, 22)
    features['VIX'] = spy_vix.reindex(df.index).ffill()
    
    # VIX lag 특성
    features['VIX_lag1'] = features['VIX'].shift(1)
    features['VIX_lag5'] = features['VIX'].shift(5)
    features['VIX_change'] = features['VIX'].pct_change()
    
    # VRP
    features['VRP'] = features['VIX'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['VRP_lag5'] = features['VRP'].shift(5)
    features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    # 시장
    features['regime_high'] = (features['VIX'] >= 25).astype(int)
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    
    return features.dropna()

def run_experiment(ticker, models_dict):
    """단일 자산에 대해 모든 모델 실험"""
    print(f"\n{'='*50}")
    print(f"Processing {ticker}...")
    
    # 데이터 다운로드
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        print(f"  Skipping {ticker}: insufficient data")
        return None
    
    # 특성 생성
    features = prepare_features(df)
    
    # 특성 및 타겟
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 'VIX_change',
                   'VRP_lag1', 'VRP_lag5', 'VRP_ma5', 'regime_high', 'return_5d', 'return_22d']
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    # Train/Test 분할 (80/20, 22일 Gap)
    gap = 22
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx - gap]
    y_train = y.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    results = {}
    
    for model_name, model in models_dict.items():
        try:
            # 학습
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            
            # 평가 지표
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # 방향 정확도 (VRP 방향 예측)
            y_test_arr = y_test.values
            y_pred_arr = y_pred
            
            # 단순화: 예측값과 실제값이 같은 방향으로 움직이는지
            actual_up = y_test_arr > np.median(y_test_arr)
            pred_up = y_pred_arr > np.median(y_pred_arr)
            direction_acc = (actual_up == pred_up).mean()
            
            results[model_name] = {
                'R2': round(r2, 4),
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'Direction': round(direction_acc * 100, 2)
            }
            print(f"  {model_name}: R2={r2:.4f}, MAE={mae:.2f}, Direction={direction_acc*100:.1f}%")
            
        except Exception as e:
            print(f"  {model_name} failed: {e}")
            results[model_name] = {'R2': None, 'MAE': None, 'RMSE': None, 'Direction': None}
    
    return results

def main():
    # 자산 목록
    assets = ['SPY', 'GLD', 'TLT', 'QQQ', 'EEM']
    
    # 모델 정의
    models = {
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    }
    
    # 결과 저장
    all_results = {}
    
    for asset in assets:
        # 모델 재초기화 (매 자산마다)
        models = {
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            'MLP_64': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=42, early_stopping=True))
            ]),
            'MLP_128_64': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True))
            ])
        }
        
        result = run_experiment(asset, models)
        if result:
            all_results[asset] = result
    
    # 결과 저장
    output = {
        'results': all_results,
        'assets': assets,
        'models': list(models.keys()),
        'metrics': ['R2', 'MAE', 'RMSE', 'Direction'],
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = 'data/results/multi_model_comparison.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Results saved to {output_path}")
    
    # 요약 출력
    print("\n=== Summary Table ===")
    for asset in assets:
        if asset in all_results:
            print(f"\n{asset}:")
            for model, metrics in all_results[asset].items():
                if metrics['R2'] is not None:
                    print(f"  {model}: R2={metrics['R2']:.3f}, Direction={metrics['Direction']:.1f}%")

if __name__ == "__main__":
    main()
