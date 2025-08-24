#!/usr/bin/env python3
"""
사이킷런 기반 모델 테스트 (TensorFlow 없이)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def test_sklearn_models():
    """사이킷런 기반 모델들 테스트"""
    print("🔄 사이킷런 기반 모델 테스트 시작...")
    
    models_dir = "data/models"
    data_dir = "data/raw"
    
    # 테스트 데이터 로드
    print("📊 테스트 데이터 로드...")
    try:
        features_df = pd.read_csv(f"{data_dir}/training_features.csv")
        labels_df = pd.read_csv(f"{data_dir}/event_labels.csv")
        
        # 데이터 분할 (간단히 80/20)
        split_idx = int(len(features_df) * 0.8)
        X_test = features_df.iloc[split_idx:].drop(['timestamp', 'symbol'], axis=1, errors='ignore')
        y_test = labels_df.iloc[split_idx:]['has_event'].values
        
        print(f"✅ 테스트 데이터: {len(X_test)}개 샘플")
        
        # 스케일러 로드 및 적용
        scaler = joblib.load(f"{models_dir}/scaler.pkl")
        X_test_scaled = scaler.transform(X_test)
        
        # 테스트할 모델들
        model_files = {
            'Random Forest': 'random_forest_model.pkl',
            'Gradient Boosting': 'gradient_boosting_model.pkl',
            'XGBoost': 'xgboost_model.pkl',
            'Random Forest (Improved)': 'random_forest_improved_model.pkl',
            'Gradient Boosting (Improved)': 'gradient_boosting_improved_model.pkl'
        }
        
        results = {}
        
        for model_name, filename in model_files.items():
            model_path = f"{models_dir}/{filename}"
            
            if os.path.exists(model_path):
                print(f"\n🔍 {model_name} 테스트 중...")
                
                try:
                    # 모델 로드
                    model = joblib.load(model_path)
                    
                    # 예측
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        y_pred = (y_pred_proba > 0.5).astype(int)
                    else:
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = None
                    
                    # 메트릭 계산
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    model_results = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'predictions': len(y_pred),
                        'positive_predictions': int(np.sum(y_pred)),
                        'actual_positives': int(np.sum(y_test))
                    }
                    
                    if y_pred_proba is not None:
                        try:
                            auc = roc_auc_score(y_test, y_pred_proba)
                            model_results['auc'] = auc
                        except:
                            model_results['auc'] = None
                    
                    results[model_name] = model_results
                    
                    # 결과 출력
                    print(f"  ✅ 정확도: {accuracy:.3f}")
                    print(f"  ✅ 정밀도: {precision:.3f}")
                    print(f"  ✅ 재현율: {recall:.3f}")
                    print(f"  ✅ F1 점수: {f1:.3f}")
                    if 'auc' in model_results and model_results['auc']:
                        print(f"  ✅ AUC: {model_results['auc']:.3f}")
                    print(f"  📊 예측: {model_results['positive_predictions']}/{len(y_pred)} 긍정")
                    
                except Exception as e:
                    print(f"  ❌ 오류: {str(e)}")
            else:
                print(f"  ⚠️  모델 파일 없음: {filename}")
        
        # 결과 저장
        results_file = f"{data_dir}/sklearn_model_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 결과 저장: {results_file}")
        
        # 최고 성능 모델 찾기
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
            print(f"\n🏆 최고 성능 모델: {best_model[0]}")
            print(f"   F1 점수: {best_model[1]['f1_score']:.3f}")
            
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        return None


if __name__ == "__main__":
    test_sklearn_models()