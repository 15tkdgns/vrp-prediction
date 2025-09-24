#!/usr/bin/env python3
"""
실제 SPY 데이터로 변동성 예측 모델의 R² = 0.2136 달성 가능성 검증
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_volatility_target(df):
    """실제 변동성 타겟 생성"""
    # 일일 수익률이 이미 있다고 가정
    if 'Returns' not in df.columns:
        df['Returns'] = df['Close'].pct_change()

    # 다음날 변동성 계산 (절대값 기준)
    df['next_day_volatility'] = np.abs(df['Returns'].shift(-1))

    return df

def test_volatility_prediction():
    """실제 데이터로 변동성 예측 테스트"""
    print("🔍 실제 SPY 데이터로 변동성 예측 검증")
    print("="*60)

    # 1. 데이터 로드
    df = pd.read_csv('/root/workspace/data/training/sp500_leak_free_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"📊 데이터 로드: {df.shape}")

    # 2. 변동성 타겟 생성
    df = create_volatility_target(df)

    # 3. 특성 선택 (현재시스템상태2.txt에서 언급된 7개 특성 사용)
    safe_features = [
        'MA_20', 'MA_50', 'RSI', 'Volatility_20',
        'Volume_ratio_20', 'Returns_lag_1', 'Returns_lag_2'
    ]

    # 사용 가능한 특성만 선택
    available_features = [col for col in safe_features if col in df.columns]
    print(f"🔧 사용 특성: {len(available_features)}개")
    for feature in available_features:
        print(f"   ✅ {feature}")

    # 4. 데이터 준비
    X = df[available_features].copy()
    y = df['next_day_volatility'].copy()

    # NaN 제거
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[mask]
    y_clean = y[mask]

    print(f"📊 최종 데이터: {len(X_clean)} 샘플")
    print(f"🎯 타겟 통계:")
    print(f"   평균: {y_clean.mean():.6f}")
    print(f"   표준편차: {y_clean.std():.6f}")
    print(f"   범위: {y_clean.min():.6f} ~ {y_clean.max():.6f}")

    # 5. 시계열 분할 (현재시스템상태2.txt에서 언급된 방식)
    print(f"\n🔄 시계열 분할로 교차검증...")

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    scaler = StandardScaler()

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
        X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]

        # 정규화
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 모델 훈련 (현재시스템상태2.txt에서 언급된 파라미터)
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 예측 및 평가
        y_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, y_pred)
        cv_scores.append(r2)

        print(f"   Fold {fold+1}: R² = {r2:.4f}")

    # 6. 최종 모델 훈련 (전체 데이터의 80%로)
    print(f"\n🤖 최종 모델 훈련...")

    split_idx = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split_idx]
    X_test = X_clean.iloc[split_idx:]
    y_train = y_clean.iloc[:split_idx]
    y_test = y_clean.iloc[split_idx:]

    # 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 훈련
    final_model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
    final_model.fit(X_train_scaled, y_train)

    # 예측
    y_train_pred = final_model.predict(X_train_scaled)
    y_test_pred = final_model.predict(X_test_scaled)

    # 평가
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # 7. 결과 요약
    print(f"\n🎯 최종 결과:")
    print(f"="*40)
    print(f"📊 교차검증 R²:")
    print(f"   평균: {np.mean(cv_scores):.4f}")
    print(f"   표준편차: {np.std(cv_scores):.4f}")
    print(f"   범위: {np.min(cv_scores):.4f} ~ {np.max(cv_scores):.4f}")

    print(f"\n📊 최종 모델 성능:")
    print(f"   훈련 R²: {train_r2:.4f}")
    print(f"   테스트 R²: {test_r2:.4f}")
    print(f"   테스트 MAE: {test_mae:.6f}")
    print(f"   테스트 RMSE: {test_rmse:.6f}")

    # 8. 주장 검증
    claimed_r2 = 0.2136
    print(f"\n🔍 주장 검증:")
    print(f"   현재시스템상태2.txt 주장: R² = {claimed_r2:.4f}")
    print(f"   실제 달성: R² = {test_r2:.4f}")

    if test_r2 >= claimed_r2 * 0.8:  # 80% 이상이면 달성 가능
        print(f"   ✅ 주장 달성 가능: {test_r2:.4f} >= {claimed_r2 * 0.8:.4f}")
        is_achievable = True
    else:
        print(f"   ❌ 주장 달성 불가: {test_r2:.4f} < {claimed_r2 * 0.8:.4f}")
        is_achievable = False

    # 9. 특성 중요도
    print(f"\n🔧 특성 중요도:")
    feature_importance = np.abs(final_model.coef_)
    feature_names = available_features

    for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
        print(f"   {feature:15s}: {importance:.4f}")

    return {
        'cv_r2_mean': np.mean(cv_scores),
        'cv_r2_std': np.std(cv_scores),
        'test_r2': test_r2,
        'train_r2': train_r2,
        'claimed_r2': claimed_r2,
        'is_achievable': is_achievable,
        'sample_count': len(X_clean),
        'feature_count': len(available_features)
    }

if __name__ == "__main__":
    result = test_volatility_prediction()

    print(f"\n🎉 검증 완료!")
    print(f"   샘플 수: {result['sample_count']}")
    print(f"   특성 수: {result['feature_count']}")
    print(f"   달성 가능: {result['is_achievable']}")