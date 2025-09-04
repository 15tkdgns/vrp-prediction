#!/usr/bin/env python3
"""
2025년 1-6월 기간에 대한 예측 데이터 생성
훈련된 모델을 사용하여 실제 데이터와 비교 가능한 예측 생성
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_trained_model():
    """훈련된 모델 로드"""
    try:
        with open('models/spy_ensemble_model_20250903_142108.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        print("훈련된 모델을 찾을 수 없습니다.")
        return None

def prepare_features_for_prediction(df):
    """예측을 위한 특성 준비 (기존 ML 파이프라인과 동일한 특성)"""
    # 기본 수익률 계산
    df['Return_1d'] = df['Close'].pct_change()
    df['Return_2d'] = df['Close'].pct_change(periods=2)
    df['Return_5d'] = df['Close'].pct_change(periods=5)
    
    # 변동성
    df['Volatility_5d'] = df['Return_1d'].rolling(5).std()
    df['Volatility_20d'] = df['Return_1d'].rolling(20).std()
    
    # 거래량 변화
    df['Volume_Change_1d'] = df['Volume'].pct_change()
    df['Volume_Change_2d'] = df['Volume'].pct_change(periods=2)
    df['Volume_Change_5d'] = df['Volume'].pct_change(periods=5)
    
    # 이동평균
    for period in [5, 10, 20]:
        df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
        df[f'Price_SMA_Ratio_{period}'] = df['Close'] / df[f'SMA_{period}']
    
    # 가격 비율
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Lag 특성
    for lag in [1, 2, 3]:
        df[f'Return_Lag_{lag}'] = df['Return_1d'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    
    # 거래량 이동평균 비율
    df['Volume_SMA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA_Ratio'] = df['Volume'] / df['Volume_SMA_5']
    
    return df

def generate_predictions_for_period():
    """2025년 1-6월 기간에 대한 예측 생성"""
    
    # 훈련된 모델 로드
    model_data = load_trained_model()
    if model_data is None:
        return None
        
    ensemble_model = model_data['ensemble_model']
    scaler = model_data['scaler']
    feature_selector = model_data['feature_selector']
    
    # 전체 데이터 로드 (예측을 위해 더 긴 기간 필요)
    df = pd.read_csv('data/raw/spy_data_2020_2025.csv')
    df['Date'] = pd.to_datetime(df['Unnamed: 0'])
    df = df.drop('Unnamed: 0', axis=1)
    
    # 특성 생성
    df = prepare_features_for_prediction(df)
    
    # 2025년 1-6월 데이터 추출
    mask = (df['Date'] >= '2025-01-01') & (df['Date'] <= '2025-06-30')
    df_2025 = df[mask].copy()
    
    if len(df_2025) == 0:
        print("2025년 데이터가 없습니다.")
        return None
    
    # 특성 선택 (NaN 제거)
    feature_columns = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    # 예측 가능한 데이터만 선택 (NaN 없는 행)
    df_pred = df_2025[['Date', 'Close'] + feature_columns].dropna()
    
    if len(df_pred) == 0:
        print("예측 가능한 데이터가 없습니다.")
        return None
    
    X_pred = df_pred[feature_columns]
    
    # 특성 스케일링 및 선택
    X_pred_scaled = scaler.transform(X_pred)
    X_pred_selected = feature_selector.transform(X_pred_scaled)
    
    # 예측 수행
    predictions = ensemble_model.predict(X_pred_selected)
    prediction_proba = ensemble_model.predict_proba(X_pred_selected)
    
    # 결과 정리
    results = []
    for i, (_, row) in enumerate(df_pred.iterrows()):
        actual_return = row['Close'] / df_pred['Close'].iloc[max(0, i-1)] - 1 if i > 0 else 0
        
        results.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'actual_price': float(row['Close']),
            'prediction': int(predictions[i]),  # 0: Down, 1: Up
            'prediction_label': 'Up' if predictions[i] == 1 else 'Down',
            'confidence': float(max(prediction_proba[i])),
            'up_probability': float(prediction_proba[i][1]),
            'down_probability': float(prediction_proba[i][0]),
            'actual_return': float(actual_return)
        })
    
    # 예측 정확도 계산
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(1, len(results)):
        actual_direction = 1 if results[i]['actual_return'] > 0 else 0
        predicted_direction = results[i]['prediction']
        
        if actual_direction == predicted_direction:
            correct_predictions += 1
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # 결과 저장
    prediction_data = {
        'period': '2025-01-01 to 2025-06-30',
        'model_info': {
            'accuracy_on_period': round(accuracy, 4),
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions
        },
        'predictions': results
    }
    
    with open('data/raw/spy_2025_h1_predictions.json', 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    print(f"\n예측 결과 생성 완료:")
    print(f"- 예측 기간: {df_pred['Date'].min().strftime('%Y-%m-%d')} ~ {df_pred['Date'].max().strftime('%Y-%m-%d')}")
    print(f"- 총 예측 수: {len(results)}")
    print(f"- 예측 정확도: {accuracy:.2%}")
    print(f"- 저장 파일: data/raw/spy_2025_h1_predictions.json")
    
    return prediction_data

if __name__ == "__main__":
    generate_predictions_for_period()