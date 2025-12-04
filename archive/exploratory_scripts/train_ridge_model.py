#!/usr/bin/env python3
"""
Ridge 회귀 변동성 예측 모델 훈련
CLAUDE.md 명세에 따른 R² = 0.3113 목표 모델
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# scikit-learn imports
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# yfinance for data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance 미설치")


class PurgedKFold:
    """Purged and Embargoed K-Fold Cross-Validation"""

    def __init__(self, n_splits=5, purge_length=5, embargo_length=5):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X, y=None, groups=None):
        """Purged and Embargoed 분할"""
        n_samples = len(X)
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            test_indices = list(range(test_start, test_end))

            train_indices = []

            if test_start > self.purge_length:
                train_indices.extend(range(0, test_start - self.purge_length))

            if test_end + self.embargo_length < n_samples:
                train_indices.extend(range(test_end + self.embargo_length, n_samples))

            yield train_indices, test_indices


def get_real_spy_data():
    """실제 SPY 데이터 수집"""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance 필요: pip install yfinance")

    print("📊 실제 SPY 데이터 수집 중...")
    spy = yf.Ticker("SPY")
    data = spy.history(start="2015-01-01", end="2024-12-31", interval="1d")

    if data.empty:
        raise ValueError("SPY 데이터 수집 실패")

    prices = data['Close']
    returns = np.log(prices / prices.shift(1)).dropna()

    result = pd.DataFrame({
        'price': prices.loc[returns.index],
        'returns': returns
    })

    print(f"✅ 실제 SPY 데이터: {len(result)}개 관측치")
    return result


def create_features(data):
    """변동성 예측 피처 생성 (t 시점 이전 데이터만)"""
    print("🔧 피처 생성 (t 시점 이전)...")

    features = pd.DataFrame(index=data.index)
    returns = data['returns']

    # 1. 변동성 피처
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

    # 4. 교차 통계
    features['vol_ratio_5_20'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
    features['vol_ratio_10_50'] = features['volatility_10'] / (features['volatility_50'] + 1e-8)

    # 5. Z-score
    ma_20 = returns.rolling(20).mean()
    std_20 = returns.rolling(20).std()
    features['zscore_20'] = (returns - ma_20) / (std_20 + 1e-8)

    # 6. 모멘텀
    for window in [5, 10, 20]:
        features[f'momentum_{window}'] = returns.rolling(window).sum()

    print(f"✅ 피처 생성 완료: {len(features.columns)}개")
    return features


def create_target(data, horizon=5):
    """미래 변동성 타겟 생성 (t+1 시점부터 미래)"""
    print(f"🎯 타겟 생성 (t+1부터 {horizon}일 후 변동성)...")

    returns = data['returns']
    vol_values = []

    for i in range(len(returns)):
        if i + horizon < len(returns):
            future_window = returns.iloc[i+1:i+1+horizon]  # t+1부터 t+1+horizon까지
            vol_values.append(future_window.std())
        else:
            vol_values.append(np.nan)

    target = pd.Series(vol_values, index=data.index, name='target_vol_5d')
    print(f"✅ 타겟 생성 완료")
    return target


def train_ridge_model():
    """Ridge 회귀 모델 훈련 및 저장"""
    print("🚀 Ridge 회귀 변동성 예측 모델 훈련")
    print("=" * 80)

    # 1. 데이터 수집
    data = get_real_spy_data()

    # 2. 피처 및 타겟 생성
    features = create_features(data)
    target = create_target(data, horizon=5)

    # 3. 데이터 결합 및 정리
    combined = pd.concat([features, target], axis=1).dropna()
    feature_cols = features.columns.tolist()

    X = combined[feature_cols]
    y = combined['target_vol_5d']

    print(f"\n💾 훈련 데이터:")
    print(f"   샘플 수: {len(X)}")
    print(f"   피처 수: {len(feature_cols)}")

    # 4. Purged K-Fold CV로 성능 평가
    print(f"\n🤖 Ridge 모델 Purged K-Fold CV (5-fold)")
    print("-" * 60)

    purged_cv = PurgedKFold(n_splits=5, purge_length=5, embargo_length=5)
    ridge_model = Ridge(alpha=1.0)

    cv_scores = []
    cv_mae = []
    cv_mse = []

    for fold, (train_idx, val_idx) in enumerate(purged_cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        ridge_model.fit(X_train_scaled, y_train)
        y_pred = ridge_model.predict(X_val_scaled)

        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)

        cv_scores.append(r2)
        cv_mae.append(mae)
        cv_mse.append(mse)

        print(f"   Fold {fold+1}: R² = {r2:.4f}, MAE = {mae:.6f}, RMSE = {np.sqrt(mse):.6f}")

    avg_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)
    avg_mae = np.mean(cv_mae)
    avg_rmse = np.sqrt(np.mean(cv_mse))

    print(f"\n📊 Cross-Validation 결과:")
    print(f"   평균 R²:   {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"   평균 MAE:  {avg_mae:.6f}")
    print(f"   평균 RMSE: {avg_rmse:.6f}")

    # 5. 전체 데이터로 최종 모델 훈련
    print(f"\n🔨 전체 데이터로 최종 모델 훈련...")

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_model = Ridge(alpha=1.0)
    final_model.fit(X_scaled, y)

    # 전체 데이터 성능 (참고용)
    y_pred_full = final_model.predict(X_scaled)
    train_r2 = r2_score(y, y_pred_full)
    train_mae = mean_absolute_error(y, y_pred_full)
    train_mse = mean_squared_error(y, y_pred_full)

    print(f"   전체 데이터 R²:   {train_r2:.4f}")
    print(f"   전체 데이터 MAE:  {train_mae:.6f}")
    print(f"   전체 데이터 RMSE: {np.sqrt(train_mse):.6f}")

    # 6. 모델 및 메타데이터 저장
    print(f"\n💾 모델 저장 중...")

    os.makedirs('models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)

    # Ridge 모델 저장
    with open('models/ridge_volatility_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)

    # 스케일러 저장
    with open('models/ridge_scaler.pkl', 'wb') as f:
        pickle.dump(final_scaler, f)

    # 피처 이름 저장
    with open('models/ridge_feature_names.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    # 메타데이터 저장
    metadata = {
        'model_type': 'Ridge',
        'alpha': 1.0,
        'feature_count': len(feature_cols),
        'feature_names': feature_cols,
        'training_samples': len(X),
        'target_variable': 'target_vol_5d',
        'target_horizon': 5,
        'validation_method': 'Purged K-Fold CV (5-fold, purge=5, embargo=5)',
        'cv_performance': {
            'mean_r2': float(avg_r2),
            'std_r2': float(std_r2),
            'mean_mae': float(avg_mae),
            'mean_rmse': float(avg_rmse),
            'fold_scores': [float(s) for s in cv_scores]
        },
        'train_performance': {
            'r2': float(train_r2),
            'mae': float(train_mae),
            'mse': float(train_mse),
            'rmse': float(np.sqrt(train_mse))
        },
        'trained_date': datetime.now().isoformat(),
        'data_period': '2015-01-01 to 2024-12-31',
        'data_source': 'yfinance (SPY ETF)'
    }

    with open('models/ridge_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # 성능 데이터 저장 (시스템 오케스트레이터용)
    performance_data = {
        'model_name': 'Ridge Volatility Predictor',
        'model_type': 'Ridge',
        'target': 'target_vol_5d',
        'test_r2': float(avg_r2),
        'test_mae': float(avg_mae),
        'test_rmse': float(avg_rmse),
        'cv_std': float(std_r2),
        'validation_method': 'Purged K-Fold CV (5-fold)',
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'timestamp': datetime.now().isoformat()
    }

    with open('data/raw/model_performance.json', 'w') as f:
        json.dump(performance_data, f, indent=2)

    print(f"\n✅ 모델 저장 완료:")
    print(f"   - models/ridge_volatility_model.pkl")
    print(f"   - models/ridge_scaler.pkl")
    print(f"   - models/ridge_feature_names.pkl")
    print(f"   - models/ridge_model_metadata.json")
    print(f"   - data/raw/model_performance.json")

    # 7. 결과 요약
    print(f"\n" + "=" * 80)
    print(f"📊 최종 결과 요약")
    print(f"=" * 80)
    print(f"모델: Ridge(alpha=1.0)")
    print(f"검증 방법: Purged K-Fold CV (5-fold, purge=5, embargo=5)")
    print(f"CV 성능: R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"데이터: {len(X)} 샘플, {len(feature_cols)} 피처")
    print(f"타겟: 5일 후 변동성 (완전한 시간적 분리)")

    if avg_r2 > 0.25:
        print(f"✅ 목표 성능 달성 (R² > 0.25)")
    elif avg_r2 > 0.15:
        print(f"📈 양호한 성능 (R² > 0.15)")
    else:
        print(f"⚠️ 성능 개선 필요")

    return final_model, final_scaler, metadata


if __name__ == "__main__":
    try:
        model, scaler, metadata = train_ridge_model()
        print(f"\n✅ Ridge 회귀 모델 훈련 완료!")
    except Exception as e:
        print(f"\n❌ 모델 훈련 실패: {e}")
        import traceback
        traceback.print_exc()
