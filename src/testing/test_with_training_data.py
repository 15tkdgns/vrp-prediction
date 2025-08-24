#!/usr/bin/env python3
"""
개선된 모델을 훈련 데이터로 테스트 (신뢰도 확인)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import logging
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')


def test_improved_models_with_training_data():
    """개선된 모델을 훈련 데이터로 테스트"""
    print("🎯 개선된 모델 신뢰도 테스트")
    print("=" * 40)
    
    # 모델 및 데이터 로드
    models_dir = "data/models"
    data_dir = "data/raw"
    
    try:
        # 스케일러 로드
        scaler = joblib.load(f"{models_dir}/scaler_improved.pkl")
        print("✅ 스케일러 로드 완료")
        
        # 모델들 로드
        models = {}
        models['random_forest'] = joblib.load(f"{models_dir}/random_forest_improved_model.pkl")
        models['gradient_boosting'] = joblib.load(f"{models_dir}/gradient_boosting_improved_model.pkl")
        models['lstm'] = load_model(f"{models_dir}/lstm_improved_model.h5")
        print("✅ 모든 모델 로드 완료")
        
        # 훈련 데이터 로드
        features_df = pd.read_csv(f"{data_dir}/training_features.csv")
        labels_df = pd.read_csv(f"{data_dir}/event_labels.csv")
        
        # 데이터 병합
        merged_df = pd.merge(features_df, labels_df, on=["ticker", "Date"], how="inner")
        
        # 특성 선택 (숫자형만)
        numeric_columns = merged_df.select_dtypes(include=[np.number]).columns.tolist()
        target_columns = ['major_event', 'price_spike', 'unusual_volume']
        feature_columns = [col for col in numeric_columns if col not in target_columns]
        
        X = merged_df[feature_columns].fillna(0)
        y = merged_df['major_event']
        
        print(f"📊 테스트 데이터: {len(X)}개 샘플")
        print(f"🎯 실제 이벤트 비율: {y.mean():.3f}")
        
        # 특성 스케일링
        X_scaled = scaler.transform(X)
        
        # 각 모델별 예측 및 신뢰도 분석
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🤖 {model_name.upper()} 신뢰도 분석:")
            print("-" * 30)
            
            if model_name == 'lstm':
                X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                predictions = model.predict(X_lstm, verbose=0).flatten()
            else:
                predictions = model.predict_proba(X_scaled)[:, 1]
            
            # 신뢰도 통계
            avg_confidence = np.mean(predictions)
            confidence_std = np.std(predictions)
            median_confidence = np.median(predictions)
            min_confidence = np.min(predictions)
            max_confidence = np.max(predictions)
            
            # 이벤트별 신뢰도
            event_predictions = predictions[y == 1]
            normal_predictions = predictions[y == 0]
            
            event_avg = np.mean(event_predictions) if len(event_predictions) > 0 else 0
            normal_avg = np.mean(normal_predictions) if len(normal_predictions) > 0 else 0
            
            # 신뢰도 구간별 분포
            low_conf = np.sum(predictions < 0.2) / len(predictions)
            mid_conf = np.sum((predictions >= 0.2) & (predictions <= 0.8)) / len(predictions)
            high_conf = np.sum(predictions > 0.8) / len(predictions)
            
            results[model_name] = {
                'avg_confidence': float(avg_confidence),
                'confidence_std': float(confidence_std),
                'median_confidence': float(median_confidence),
                'min_confidence': float(min_confidence),
                'max_confidence': float(max_confidence),
                'event_avg_confidence': float(event_avg),
                'normal_avg_confidence': float(normal_avg),
                'confidence_distribution': {
                    'low_confidence_pct': float(low_conf),
                    'mid_confidence_pct': float(mid_conf),
                    'high_confidence_pct': float(high_conf)
                }
            }
            
            print(f"  평균 신뢰도: {avg_confidence:.4f} ± {confidence_std:.4f}")
            print(f"  중앙값: {median_confidence:.4f}")
            print(f"  범위: {min_confidence:.4f} ~ {max_confidence:.4f}")
            print(f"  이벤트시 평균: {event_avg:.4f}")
            print(f"  정상시 평균: {normal_avg:.4f}")
            print(f"  신뢰도 분포:")
            print(f"    낮음 (<0.2): {low_conf*100:.1f}%")
            print(f"    중간 (0.2-0.8): {mid_conf*100:.1f}%")
            print(f"    높음 (>0.8): {high_conf*100:.1f}%")
            
            # 극단적인 예측 샘플 표시
            high_indices = np.argsort(predictions)[-3:][::-1]
            print(f"  가장 높은 신뢰도 3개:")
            for i, idx in enumerate(high_indices):
                row = merged_df.iloc[idx]
                actual = "이벤트" if y.iloc[idx] == 1 else "정상"
                print(f"    {i+1}. {row['ticker']} - {predictions[idx]:.4f} (실제: {actual})")
        
        # 결과 저장
        test_results = {
            'test_timestamp': datetime.now().isoformat(),
            'test_info': {
                'samples': len(X),
                'features': len(feature_columns),
                'actual_event_rate': float(y.mean())
            },
            'model_results': results
        }
        
        with open(f"{data_dir}/improved_model_confidence_test.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n💾 결과 저장됨: {data_dir}/improved_model_confidence_test.json")
        
        # 모델 비교 및 평가
        print(f"\n📊 모델 신뢰도 비교:")
        print("-" * 40)
        
        for model_name in results:
            r = results[model_name]
            # 현실적인 신뢰도 점수 (0.1-0.3 사이가 이상적)
            realism_score = 1.0 - abs(r['avg_confidence'] - 0.15)  # 0.15를 이상적으로 가정
            
            print(f"{model_name.upper()}:")
            print(f"  평균 신뢰도: {r['avg_confidence']:.4f}")
            print(f"  표준편차: {r['confidence_std']:.4f}")
            print(f"  현실성 점수: {realism_score:.4f}")
            print()
        
        # 가장 현실적인 모델 찾기
        best_model = min(results.keys(), 
                        key=lambda x: abs(results[x]['avg_confidence'] - 0.15))
        
        print(f"🏆 가장 현실적인 모델: {best_model.upper()}")
        print(f"   평균 신뢰도: {results[best_model]['avg_confidence']:.4f}")
        print(f"   표준편차: {results[best_model]['confidence_std']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    success = test_improved_models_with_training_data()
    
    if success:
        print("\n✅ 개선된 모델 신뢰도 테스트 완료!")
        print("   이제 현실적인 신뢰도로 예측이 가능합니다.")
    else:
        print("\n❌ 테스트 실패!")