"""
SPY 실제값 vs 예측값 시계열 데이터 생성
========================================
대시보드 시각화용 데이터 생성
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def generate_spy_predictions():
    """SPY 실제값 vs 예측값 시계열 생성"""
    print("SPY 데이터 다운로드...")
    
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
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, np.sqrt(y_train))
    
    pred = np.maximum(model.predict(X_test_s) ** 2, 0)
    
    # 시계열 데이터 생성
    dates = features.index[train_end+gap:].strftime('%Y-%m-%d').tolist()
    actual = y_test.values.tolist()
    predicted = pred.tolist()
    
    # 250일 Rolling R² 계산
    rolling_r2 = []
    window = 250
    for i in range(len(actual)):
        if i < window:
            rolling_r2.append(None)
        else:
            y_slice = np.array(actual[i-window:i])
            p_slice = np.array(predicted[i-window:i])
            ss_res = np.sum((y_slice - p_slice) ** 2)
            ss_tot = np.sum((y_slice - np.mean(y_slice)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            rolling_r2.append(r2)
    
    result = {
        'dates': dates,
        'actual': actual,
        'predicted': predicted,
        'rolling_r2': rolling_r2,
        'metadata': {
            'asset': 'SPY',
            'model': 'Ridge_100',
            'test_start': dates[0],
            'test_end': dates[-1],
            'n_samples': len(dates)
        }
    }
    
    # 저장
    output_path = 'data/results/spy_predictions_timeseries.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"저장: {output_path}")
    print(f"샘플 수: {len(dates)}")
    print(f"테스트 기간: {dates[0]} ~ {dates[-1]}")
    
    return result

if __name__ == "__main__":
    generate_spy_predictions()
