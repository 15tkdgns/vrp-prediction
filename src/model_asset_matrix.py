"""
모델 × 자산 성능 매트릭스
========================
여러 모델과 자산에 대한 종합 성능 평가

모델:
- Ridge (α=10, 100, 1000)
- Lasso (α=0.01, 0.1)
- Huber
- ElasticNet

자산:
- 주식 지수: SPY, QQQ, IWM, DIA
- 섹터: XLK, XLF, XLE, XLV

지표:
- R², RMSE, MAE, 방향정확도
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_features(ticker):
    """특성 준비"""
    try:
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
    except Exception as e:
        return None

def get_models():
    """모델 정의"""
    return {
        'Ridge_10': Ridge(alpha=10.0),
        'Ridge_100': Ridge(alpha=100.0),
        'Ridge_1000': Ridge(alpha=1000.0),
        'Lasso_0.01': Lasso(alpha=0.01, max_iter=3000),
        'Lasso_0.1': Lasso(alpha=0.1, max_iter=3000),
        'Huber': HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000)
    }

def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, scaler):
    """모델 평가"""
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 타겟 변환 (sqrt)
    y_train_t = np.sqrt(y_train)
    
    try:
        from sklearn.base import clone
        model_copy = clone(model)
        model_copy.fit(X_train_s, y_train_t)
        pred_t = model_copy.predict(X_test_s)
        pred = np.maximum(pred_t ** 2, 0)
        
        # 지표 계산
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        
        # 방향 정확도
        persist_pred = X_test.iloc[:, 0].values  # RV_5d_lag1
        actual_dir = (y_test.values > persist_pred).astype(int)
        pred_dir = (pred > persist_pred).astype(int)
        direction_acc = (actual_dir == pred_dir).mean()
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'direction_acc': direction_acc
        }
    except Exception as e:
        return None

def main():
    print("="*80)
    print("모델 × 자산 성능 매트릭스")
    print("="*80)
    
    # 자산 리스트 (주요 자산만)
    assets = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV']
    
    # 모델
    models = get_models()
    
    # 결과 저장용
    results_matrix = {}
    
    for ticker in assets:
        print(f"\n[{ticker}]", end="")
        
        features = prepare_features(ticker)
        if features is None:
            print(" SKIP")
            continue
        
        feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                        'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
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
        
        results_matrix[ticker] = {}
        
        for model_name, model in models.items():
            result = evaluate_model(model_name, model, X_train, y_train, 
                                   X_test, y_test, scaler)
            if result:
                results_matrix[ticker][model_name] = result
                print(f" {model_name}={result['r2']:.3f}", end="")
        
        print()
    
    # JSON 저장
    output = {
        'metadata': {
            'experiment': 'Model-Asset Performance Matrix',
            'timestamp': datetime.now().isoformat(),
            'assets': assets,
            'models': list(models.keys())
        },
        'matrix': results_matrix
    }
    
    output_path = 'data/results/model_asset_matrix.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약 테이블
    print("\n" + "="*80)
    print("R² 성능 매트릭스")
    print("="*80)
    
    # DataFrame으로 변환
    r2_data = {}
    for asset in assets:
        if asset in results_matrix:
            r2_data[asset] = {}
            for model in models.keys():
                if model in results_matrix[asset]:
                    r2_data[asset][model] = results_matrix[asset][model]['r2']
    
    df_r2 = pd.DataFrame(r2_data).T
    print("\n" + df_r2.to_string())
    
    # 각 자산별 최고 모델
    print("\n" + "="*80)
    print("자산별 최고 모델")
    print("="*80)
    
    for asset in assets:
        if asset in results_matrix:
            best_model = max(results_matrix[asset].items(), 
                           key=lambda x: x[1]['r2'])
            print(f"  {asset}: {best_model[0]} (R²={best_model[1]['r2']:.4f})")
    
    print(f"\n결과 저장: {output_path}")

if __name__ == "__main__":
    main()
