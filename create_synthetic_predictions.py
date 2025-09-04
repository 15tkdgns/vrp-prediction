#!/usr/bin/env python3
"""
2025년 1-6월 기간에 대한 합성 예측 데이터 생성 (모델 호환성 문제로 인해)
실제 가격 데이터를 기반으로 기술적 분석 기반 예측 시뮬레이션
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def calculate_technical_indicators(df):
    """기술적 지표 계산"""
    # RSI 계산
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # 이동평균
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # 볼린저 밴드
    df['BB_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # 거래량 이동평균
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    
    return df

def generate_technical_predictions(df):
    """기술적 분석 기반 예측 생성"""
    predictions = []
    
    for i in range(len(df)):
        if i == 0:
            continue
            
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 기술적 지표 기반 점수 계산
        signals = []
        
        # RSI 신호 (과매수/과매도)
        if not pd.isna(row['RSI']):
            if row['RSI'] < 30:
                signals.append(1)  # 과매도 -> 상승 신호
            elif row['RSI'] > 70:
                signals.append(-1)  # 과매수 -> 하락 신호
            else:
                signals.append(0)
        
        # 이동평균 크로스오버
        if not pd.isna(row['SMA_5']) and not pd.isna(row['SMA_20']):
            if row['SMA_5'] > row['SMA_20']:
                signals.append(1)  # 상승 신호
            else:
                signals.append(-1)  # 하락 신호
        
        # MACD 신호
        if not pd.isna(row['MACD']) and not pd.isna(row['MACD_signal']):
            if row['MACD'] > row['MACD_signal']:
                signals.append(1)
            else:
                signals.append(-1)
        
        # 볼린저 밴드 신호
        if not pd.isna(row['BB_upper']) and not pd.isna(row['BB_lower']):
            if row['Close'] < row['BB_lower']:
                signals.append(1)  # 하단선 터치 -> 상승
            elif row['Close'] > row['BB_upper']:
                signals.append(-1)  # 상단선 터치 -> 하락
        
        # 거래량 신호
        if not pd.isna(row['Volume_MA']):
            volume_ratio = row['Volume'] / row['Volume_MA']
            if volume_ratio > 1.5:  # 거래량 급증
                # 가격 변화와 같은 방향
                price_change = (row['Close'] - prev_row['Close']) / prev_row['Close']
                if price_change > 0:
                    signals.append(1)
                else:
                    signals.append(-1)
        
        # 전체 신호 집계
        if len(signals) > 0:
            avg_signal = np.mean(signals)
            prediction = 1 if avg_signal > 0 else 0
            confidence = min(0.9, 0.5 + abs(avg_signal) * 0.3)  # 0.5~0.9 범위
        else:
            prediction = 1 if np.random.random() > 0.5 else 0
            confidence = 0.5
        
        # 실제 수익률 계산
        actual_return = (row['Close'] - prev_row['Close']) / prev_row['Close']
        
        predictions.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'actual_price': float(row['Close']),
            'prediction': int(prediction),
            'prediction_label': 'Up' if prediction == 1 else 'Down',
            'confidence': float(confidence),
            'up_probability': float(confidence if prediction == 1 else 1 - confidence),
            'down_probability': float(1 - confidence if prediction == 1 else confidence),
            'actual_return': float(actual_return),
            'signals_count': len(signals)
        })
    
    return predictions

def create_synthetic_predictions():
    """합성 예측 데이터 생성"""
    
    # 2025년 데이터 로드
    df = pd.read_csv('data/raw/spy_2025_h1.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"데이터 기간: {df['Date'].min()} ~ {df['Date'].max()}")
    print(f"총 데이터 수: {len(df)}")
    
    # 기술적 지표 계산을 위해 더 긴 기간 데이터 필요
    df_full = pd.read_csv('data/raw/spy_data_2020_2025.csv')
    df_full['Date'] = pd.to_datetime(df_full['Unnamed: 0'])
    df_full = df_full.drop('Unnamed: 0', axis=1)
    
    # 2024년 11월부터 가져와서 충분한 기술적 지표 계산
    df_extended = df_full[df_full['Date'] >= '2024-11-01'].copy()
    df_extended = calculate_technical_indicators(df_extended)
    
    # 2025년 1-6월만 추출
    df_2025 = df_extended[(df_extended['Date'] >= '2025-01-01') & 
                          (df_extended['Date'] <= '2025-06-30')].copy()
    
    # 예측 생성
    predictions = generate_technical_predictions(df_2025)
    
    # 정확도 계산
    correct_predictions = 0
    total_predictions = 0
    
    for pred in predictions:
        actual_direction = 1 if pred['actual_return'] > 0 else 0
        predicted_direction = pred['prediction']
        
        if actual_direction == predicted_direction:
            correct_predictions += 1
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # 결과 저장
    prediction_data = {
        'period': '2025-01-01 to 2025-06-30',
        'model_info': {
            'type': 'Technical Analysis Based',
            'accuracy_on_period': round(accuracy, 4),
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'description': 'RSI, MACD, Bollinger Bands, Moving Average, Volume analysis'
        },
        'predictions': predictions
    }
    
    with open('data/raw/spy_2025_h1_predictions.json', 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    print(f"\n예측 결과 생성 완료:")
    print(f"- 총 예측 수: {len(predictions)}")
    print(f"- 예측 정확도: {accuracy:.2%}")
    print(f"- 저장 파일: data/raw/spy_2025_h1_predictions.json")
    
    # 월별 통계
    df_pred = pd.DataFrame(predictions)
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    df_pred['month'] = df_pred['date'].dt.strftime('%Y-%m')
    
    print(f"\n월별 예측 정확도:")
    for month in df_pred['month'].unique():
        month_data = df_pred[df_pred['month'] == month]
        month_correct = sum(1 for _, row in month_data.iterrows() 
                          if (1 if row['actual_return'] > 0 else 0) == row['prediction'])
        month_accuracy = month_correct / len(month_data)
        print(f"- {month}: {month_accuracy:.2%} ({month_correct}/{len(month_data)})")
    
    return prediction_data

if __name__ == "__main__":
    create_synthetic_predictions()