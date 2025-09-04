#!/usr/bin/env python3
"""
예측 가격 생성: AI 예측 신호를 바탕으로 실제 예측 가격을 계산
"""

import json
import numpy as np

def generate_predicted_prices():
    """AI 예측 신호를 기반으로 예측 가격 계산"""
    
    # 실제 데이터 로드
    with open('data/raw/spy_2025_h1.json', 'r') as f:
        actual_data = json.load(f)
    
    # 예측 데이터 로드
    with open('data/raw/spy_2025_h1_predictions.json', 'r') as f:
        prediction_data = json.load(f)
    
    # 실제 가격 데이터를 딕셔너리로 변환 (날짜별 인덱싱)
    actual_prices = {}
    for item in actual_data['data']:
        actual_prices[item['date']] = item['close']
    
    # 예측 가격 계산
    updated_predictions = []
    
    for pred in prediction_data['predictions']:
        date = pred['date']
        actual_price = pred['actual_price']
        confidence = pred['confidence']
        prediction_direction = pred['prediction']  # 0: Down, 1: Up
        
        # 예측 가격 계산 로직:
        # 1. 신뢰도에 따라 예측 강도 결정 (0.5% ~ 3% 범위)
        # 2. 상승 예측(1)이면 +, 하락 예측(0)이면 -
        prediction_strength = 0.005 + (confidence - 0.5) * 0.025  # 0.5% ~ 3%
        
        if prediction_direction == 1:  # 상승 예측
            predicted_price = actual_price * (1 + prediction_strength)
        else:  # 하락 예측
            predicted_price = actual_price * (1 - prediction_strength)
        
        # 예측 가격을 추가
        pred_updated = pred.copy()
        pred_updated['predicted_price'] = round(predicted_price, 2)
        pred_updated['prediction_strength'] = round(prediction_strength * 100, 2)  # 백분율
        
        updated_predictions.append(pred_updated)
    
    # 업데이트된 예측 데이터 저장
    prediction_data['predictions'] = updated_predictions
    
    with open('data/raw/spy_2025_h1_predictions.json', 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    print(f"✅ 예측 가격 생성 완료: {len(updated_predictions)}개 예측")
    print(f"예측 가격 범위: ${min(p['predicted_price'] for p in updated_predictions):.2f} ~ ${max(p['predicted_price'] for p in updated_predictions):.2f}")
    
    # 샘플 출력
    print("\n📊 샘플 예측 데이터:")
    for i, pred in enumerate(updated_predictions[:5]):
        direction = "↗️ 상승" if pred['prediction'] == 1 else "↘️ 하락"
        print(f"{pred['date']}: 실제 ${pred['actual_price']:.2f} → 예측 ${pred['predicted_price']:.2f} ({direction}, 신뢰도: {pred['confidence']:.1%})")

if __name__ == "__main__":
    generate_predicted_prices()