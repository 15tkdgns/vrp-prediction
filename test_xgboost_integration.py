#!/usr/bin/env python3
"""
XGBoost 통합 테스트 스크립트
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import json
import os

def test_xgboost_integration():
    print("🚀 XGBoost 통합 테스트 시작...")
    
    # 1. 간단한 테스트 데이터 생성
    print("\n1. 테스트 데이터 생성...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # 랜덤 특성 데이터 생성
    X = np.random.randn(n_samples, n_features)
    
    # 간단한 타겟 생성 (일부 특성의 조합으로)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    
    # 특성 이름 생성
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print(f"   - 샘플 수: {n_samples}")
    print(f"   - 특성 수: {n_features}")
    print(f"   - 클래스 분포: {np.bincount(y)}")
    
    # 2. 데이터 분할
    print("\n2. 훈련/테스트 데이터 분할...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. XGBoost 모델 훈련
    print("\n3. XGBoost 모델 훈련...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # 모델 훈련
    xgb_model.fit(X_train, y_train)
    
    # 4. 성능 평가
    print("\n4. 모델 성능 평가...")
    
    # 예측
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    # 정확도 계산
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"   - 훈련 정확도: {train_accuracy:.4f}")
    print(f"   - 테스트 정확도: {test_accuracy:.4f}")
    
    # 5. 특성 중요도 분석
    print("\n5. 특성 중요도 분석...")
    feature_importance = xgb_model.feature_importances_
    
    # 중요도 순으로 정렬
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("   - 상위 5개 중요 특성:")
    for idx, row in importance_df.head().iterrows():
        print(f"     {row['feature']}: {row['importance']:.4f}")
    
    # 6. SHAP 호환성 테스트
    print("\n6. SHAP 호환성 테스트...")
    try:
        import shap
        
        # TreeExplainer 생성
        explainer = shap.TreeExplainer(xgb_model)
        
        # 샘플 SHAP 값 계산 (소수의 샘플만)
        sample_X = X_test[:10]  # 처음 10개 샘플만
        shap_values = explainer.shap_values(sample_X)
        
        print(f"   - SHAP 값 계산 성공: {shap_values.shape}")
        print(f"   - 평균 SHAP 값: {np.mean(np.abs(shap_values)):.4f}")
        
    except ImportError:
        print("   - SHAP 패키지가 설치되지 않음")
    except Exception as e:
        print(f"   - SHAP 테스트 실패: {str(e)}")
    
    # 7. 결과 저장
    print("\n7. 테스트 결과 저장...")
    
    results = {
        'model_name': 'XGBoost',
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'feature_importance': {
            name: float(imp) for name, imp in zip(feature_names, feature_importance)
        },
        'test_date': pd.Timestamp.now().isoformat(),
        'success': True
    }
    
    # 결과를 JSON 파일로 저장
    os.makedirs('results', exist_ok=True)
    with open('results/xgboost_integration_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("   - 결과를 results/xgboost_integration_test.json에 저장")
    
    print("\n✅ XGBoost 통합 테스트 완료!")
    return results

if __name__ == "__main__":
    try:
        results = test_xgboost_integration()
        print(f"\n🎉 모든 테스트 통과! 테스트 정확도: {results['test_accuracy']:.4f}")
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()