#!/usr/bin/env python3
"""
모델 성능 문제점 진단 및 개선 방안 분석
현재 R² = -0.0209 문제 원인 규명
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

def load_spy_data():
    """SPY 데이터 로드"""
    print("📊 SPY 데이터 로드 중...")
    spy = yf.download('SPY', start='2015-01-01', end='2024-12-31', progress=False)
    spy['returns'] = spy['Close'].pct_change()
    spy = spy.dropna()
    print(f"✅ 데이터 로드 완료: {len(spy)} 관측치")
    return spy

def analyze_basic_statistics(data):
    """기본 통계 분석"""
    print("\n📈 기본 통계 분석")
    print("=" * 50)

    returns = data['returns']

    # 기본 통계량
    stats_dict = {
        'Mean': returns.mean(),
        'Std': returns.std(),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Min': returns.min(),
        'Max': returns.max(),
        'Jarque-Bera p-value': stats.jarque_bera(returns)[1]
    }

    for key, value in stats_dict.items():
        print(f"{key:20}: {value:.6f}")

    return stats_dict

def create_enhanced_features(data):
    """향상된 특성 생성"""
    print("\n🔧 향상된 특성 생성 중...")

    features = pd.DataFrame(index=data.index)
    returns = data['returns']
    prices = data['Close']
    volume = data['Volume']

    # 1. 기본 변동성 특성
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = returns.rolling(window).std()
        features[f'realized_vol_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)

    # 2. 수익률 통계
    for window in [5, 10, 20]:
        features[f'mean_return_{window}'] = returns.rolling(window).mean()
        features[f'skew_{window}'] = returns.rolling(window).skew()
        features[f'kurt_{window}'] = returns.rolling(window).kurt()

    # 3. 래그 변수
    for lag in [1, 2, 3, 5]:
        features[f'return_lag_{lag}'] = returns.shift(lag)
        features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

    # 4. 볼륨 기반 특성 (새로 추가)
    volume_sma_20 = volume.rolling(20).mean()
    features['volume_ratio'] = volume / (volume_sma_20 + 1e-8)
    features['price_volume'] = np.log(prices * volume + 1e-8)

    # 5. 가격 모멘텀 특성 (새로 추가)
    for window in [5, 10, 20]:
        features[f'momentum_{window}'] = returns.rolling(window).sum()
        features[f'rsi_{window}'] = calculate_rsi(prices, window)

    # 6. 변동성 비율
    features['vol_ratio_5_20'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
    features['vol_ratio_10_50'] = features['volatility_10'] / (features['volatility_50'] + 1e-8)

    # 7. 고급 통계 특성
    features['returns_zscore'] = (returns - returns.rolling(20).mean()) / (returns.rolling(20).std() + 1e-8)
    features['vol_zscore'] = (features['volatility_5'] - features['volatility_5'].rolling(50).mean()) / (features['volatility_5'].rolling(50).std() + 1e-8)

    print(f"✅ 특성 생성 완료: {len(features.columns)}개")
    return features

def calculate_rsi(prices, window=14):
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_future_volatility_targets(data):
    """미래 변동성 타겟 생성"""
    print("\n🎯 미래 변동성 타겟 생성 중...")

    targets = pd.DataFrame(index=data.index)
    returns = data['returns']

    # 다양한 기간의 미래 변동성
    for window in [1, 3, 5, 10, 20]:
        vol_values = []
        for i in range(len(returns)):
            if i + window < len(returns):
                future_window = returns.iloc[i+1:i+1+window]
                vol_values.append(future_window.std())
            else:
                vol_values.append(np.nan)
        targets[f'target_vol_{window}d'] = vol_values

    print(f"✅ 타겟 생성 완료: {len(targets.columns)}개")
    return targets

def analyze_feature_target_correlation(features, targets):
    """특성-타겟 상관관계 분석"""
    print("\n🔍 특성-타겟 상관관계 분석")
    print("=" * 50)

    # 결합 데이터 (NaN 처리 개선)
    combined = pd.concat([features, targets], axis=1)

    # 각 컬럼별 NaN 개수 확인
    nan_counts = combined.isnull().sum()
    print(f"\nNaN 개수 확인:")
    for col, count in nan_counts.head(10).items():
        print(f"  {col}: {count}")

    # 최소 200개 샘플이 남도록 dropna
    combined_clean = combined.dropna(thresh=len(combined.columns) * 0.7)  # 70% 이상 데이터가 있는 행만
    print(f"\n데이터 정리 후 샘플 수: {len(combined_clean)}")

    if len(combined_clean) < 100:
        print("⚠️ 샘플 수가 너무 적습니다. 보다 관대한 기준 적용")
        combined_clean = combined.dropna(subset=['target_vol_5d'])  # 타겟만 필수
        combined_clean = combined_clean.fillna(combined_clean.median())  # 나머지는 중간값으로 채움
        print(f"수정 후 샘플 수: {len(combined_clean)}")

    # 각 타겟에 대한 최고 상관관계 특성 찾기
    for target_col in targets.columns:
        if target_col in combined_clean.columns and not combined_clean[target_col].isnull().all():
            # 유효한 특성만 선택
            valid_features = []
            for feature in features.columns:
                if feature in combined_clean.columns and not combined_clean[feature].isnull().all():
                    valid_features.append(feature)

            if valid_features:
                correlations = combined_clean[valid_features].corrwith(combined_clean[target_col]).abs().sort_values(ascending=False)
                correlations = correlations.dropna()  # NaN 상관관계 제거

                print(f"\n{target_col} 예측을 위한 상위 10개 특성:")
                for i, (feature, corr) in enumerate(correlations.head(10).items()):
                    print(f"  {i+1:2d}. {feature:25}: {corr:.4f}")

    return combined_clean

def test_multiple_models(X, y):
    """다양한 모델 알고리즘 테스트"""
    print("\n🤖 다양한 모델 알고리즘 테스트")
    print("=" * 50)

    # NaN 처리 - 완전한 데이터만 사용
    combined_data = pd.concat([X, y], axis=1).dropna()
    print(f"NaN 제거 후 샘플 수: {len(combined_data)}")

    if len(combined_data) < 100:
        print("⚠️ 샘플 수가 너무 적습니다.")
        return {}

    X_clean = combined_data[X.columns]
    y_clean = combined_data[y.name]

    # 시간 순서 분할
    split_idx = int(len(X_clean) * 0.8)
    X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

    print(f"훈련 세트: {len(X_train)} 샘플")
    print(f"테스트 세트: {len(X_test)} 샘플")

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Ridge (α=0.01)': Ridge(alpha=0.01),
        'Ridge (α=0.1)': Ridge(alpha=0.1),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Ridge (α=100.0)': Ridge(alpha=100.0)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results[name] = {'r2': r2, 'mse': mse}
        print(f"{name:15}: R² = {r2:7.4f}, MSE = {mse:.6f}")

    return results

def analyze_target_predictability(targets):
    """타겟 예측 가능성 분석"""
    print("\n🎯 타겟 예측 가능성 분석")
    print("=" * 50)

    for col in targets.columns:
        target_data = targets[col].dropna()

        # 자기상관 분석
        autocorr_1 = target_data.autocorr(lag=1) if len(target_data) > 1 else 0
        autocorr_5 = target_data.autocorr(lag=5) if len(target_data) > 5 else 0

        # 기본 통계
        mean_val = target_data.mean()
        std_val = target_data.std()
        cv = std_val / mean_val if mean_val != 0 else float('inf')

        print(f"{col}:")
        print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}, CV: {cv:.4f}")
        print(f"  Autocorr(1): {autocorr_1:.4f}, Autocorr(5): {autocorr_5:.4f}")

def main():
    """메인 진단 함수"""
    print("🔬 모델 성능 문제점 진단 시작")
    print("=" * 60)

    # 1. 데이터 로드
    spy_data = load_spy_data()

    # 2. 기본 통계 분석
    basic_stats = analyze_basic_statistics(spy_data)

    # 3. 향상된 특성 생성
    features = create_enhanced_features(spy_data)

    # 4. 타겟 생성
    targets = create_future_volatility_targets(spy_data)

    # 5. 타겟 예측 가능성 분석
    analyze_target_predictability(targets)

    # 6. 특성-타겟 상관관계 분석
    combined = analyze_feature_target_correlation(features, targets)

    # 7. 5일 변동성 예측 모델 테스트
    if 'target_vol_5d' in combined.columns:
        print(f"\n📊 5일 변동성 예측 모델 테스트 (샘플 수: {len(combined)})")
        X = combined[features.columns]
        y = combined['target_vol_5d']
        results = test_multiple_models(X, y)

        # 최고 성능 모델 찾기
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"\n🏆 최고 성능 모델: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")

    # 8. 결과 저장
    os.makedirs('results', exist_ok=True)

    diagnosis_results = {
        'basic_statistics': basic_stats,
        'feature_count': len(features.columns),
        'target_count': len(targets.columns),
        'sample_count': len(combined) if 'combined' in locals() else 0,
        'model_results': results if 'results' in locals() else {}
    }

    import json
    with open('results/model_diagnosis.json', 'w') as f:
        json.dump(diagnosis_results, f, indent=2, default=str)

    print(f"\n💾 진단 결과 저장: results/model_diagnosis.json")
    print("=" * 60)

if __name__ == "__main__":
    main()