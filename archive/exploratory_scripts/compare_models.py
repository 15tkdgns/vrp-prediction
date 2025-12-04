#!/usr/bin/env python3
"""
수익률 예측 vs 변동성 예측 모델 성능 비교
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class PurgedKFold:
    """Purged and Embargoed K-Fold Cross-Validation"""
    def __init__(self, n_splits=5, purge_length=5, embargo_length=5):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X, y=None, groups=None):
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


def get_spy_data():
    """SPY 데이터 수집"""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance 필요")

    spy = yf.Ticker("SPY")
    data = spy.history(start="2015-01-01", end="2024-12-31", interval="1d")
    prices = data['Close']
    returns = np.log(prices / prices.shift(1)).dropna()

    return pd.DataFrame({'price': prices.loc[returns.index], 'returns': returns})


def create_features(data):
    """공통 피처 생성"""
    features = pd.DataFrame(index=data.index)
    returns = data['returns']

    # 변동성 피처
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = returns.rolling(window).std()
        features[f'realized_vol_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)

    # 수익률 통계
    for window in [5, 10, 20]:
        features[f'mean_return_{window}'] = returns.rolling(window).mean()
        features[f'skew_{window}'] = returns.rolling(window).skew()
        features[f'kurt_{window}'] = returns.rolling(window).kurt()

    # 래그 변수
    for lag in [1, 2, 3, 5]:
        features[f'return_lag_{lag}'] = returns.shift(lag)
        features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

    # 교차 통계
    features['vol_ratio_5_20'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
    features['vol_ratio_10_50'] = features['volatility_10'] / (features['volatility_50'] + 1e-8)

    # Z-score
    ma_20 = returns.rolling(20).mean()
    std_20 = returns.rolling(20).std()
    features['zscore_20'] = (returns - ma_20) / (std_20 + 1e-8)

    # 모멘텀
    for window in [5, 10, 20]:
        features[f'momentum_{window}'] = returns.rolling(window).sum()

    return features


def create_volatility_target(data, horizon=5):
    """변동성 예측 타겟"""
    returns = data['returns']
    vol_values = []

    for i in range(len(returns)):
        if i + horizon < len(returns):
            future_window = returns.iloc[i+1:i+1+horizon]
            vol_values.append(future_window.std())
        else:
            vol_values.append(np.nan)

    return pd.Series(vol_values, index=data.index, name='target_vol_5d')


def create_return_target(data, horizon=5):
    """수익률 예측 타겟"""
    returns = data['returns']
    return_values = []

    for i in range(len(returns)):
        if i + horizon < len(returns):
            future_window = returns.iloc[i+1:i+1+horizon]
            return_values.append(future_window.mean())  # 평균 수익률
        else:
            return_values.append(np.nan)

    return pd.Series(return_values, index=data.index, name='target_return_5d')


def train_and_evaluate_model(X, y, model_name, target_type):
    """모델 훈련 및 평가"""
    print(f"\n{'='*80}")
    print(f"모델: {model_name} | 타겟: {target_type}")
    print(f"{'='*80}")

    purged_cv = PurgedKFold(n_splits=5, purge_length=5, embargo_length=5)
    model = Ridge(alpha=1.0)

    cv_r2 = []
    cv_mae = []
    cv_mse = []

    for fold, (train_idx, val_idx) in enumerate(purged_cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)

        cv_r2.append(r2)
        cv_mae.append(mae)
        cv_mse.append(mse)

        print(f"   Fold {fold+1}: R² = {r2:7.4f}, MAE = {mae:.6f}, RMSE = {np.sqrt(mse):.6f}")

    avg_r2 = np.mean(cv_r2)
    std_r2 = np.std(cv_r2)
    avg_mae = np.mean(cv_mae)
    avg_rmse = np.sqrt(np.mean(cv_mse))

    print(f"\n📊 평균 성능:")
    print(f"   R²:   {avg_r2:7.4f} ± {std_r2:.4f}")
    print(f"   MAE:  {avg_mae:.6f}")
    print(f"   RMSE: {avg_rmse:.6f}")

    # 성능 판정
    print(f"\n🎯 성능 평가:")
    if target_type == "변동성 예측":
        if avg_r2 > 0.25:
            print(f"   ✅ 우수한 성능 (R² > 0.25)")
        elif avg_r2 > 0.15:
            print(f"   📈 양호한 성능 (R² > 0.15)")
        elif avg_r2 > 0.05:
            print(f"   📊 적정한 성능 (R² > 0.05)")
        else:
            print(f"   ⚠️ 성능 미흡 (R² < 0.05)")
    else:  # 수익률 예측
        if avg_r2 > 0.10:
            print(f"   ✅ 매우 우수 (R² > 0.10) - 금융에서 극히 드물음")
        elif avg_r2 > 0.05:
            print(f"   📈 우수한 성능 (R² > 0.05)")
        elif avg_r2 > 0.02:
            print(f"   📊 적정한 성능 (R² > 0.02)")
        elif avg_r2 > 0:
            print(f"   ⚠️ 약한 성능 (R² > 0)")
        else:
            print(f"   ❌ 예측력 없음 (R² < 0)")

    return {
        'model_name': model_name,
        'target_type': target_type,
        'avg_r2': avg_r2,
        'std_r2': std_r2,
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'fold_scores': cv_r2,
        'n_samples': len(X)
    }


def compare_models():
    """수익률 vs 변동성 예측 모델 비교"""
    print("🔍 수익률 예측 vs 변동성 예측 모델 성능 비교")
    print("="*80)

    # 데이터 로드
    data = get_spy_data()
    print(f"📊 SPY 데이터: {len(data)} 관측치 (2015-2024)\n")

    # 공통 피처 생성
    features = create_features(data)

    # 1. 변동성 예측 모델
    target_vol = create_volatility_target(data, horizon=5)
    combined_vol = pd.concat([features, target_vol], axis=1).dropna()
    X_vol = combined_vol[features.columns]
    y_vol = combined_vol['target_vol_5d']

    result_vol = train_and_evaluate_model(X_vol, y_vol, "Ridge(alpha=1.0)", "변동성 예측")

    # 2. 수익률 예측 모델
    target_ret = create_return_target(data, horizon=5)
    combined_ret = pd.concat([features, target_ret], axis=1).dropna()
    X_ret = combined_ret[features.columns]
    y_ret = combined_ret['target_return_5d']

    result_ret = train_and_evaluate_model(X_ret, y_ret, "Ridge(alpha=1.0)", "수익률 예측")

    # 비교 표
    print("\n" + "="*80)
    print("📊 최종 비교 결과")
    print("="*80)

    comparison = pd.DataFrame([
        {
            '모델': result_vol['target_type'],
            'R²': f"{result_vol['avg_r2']:.4f} ± {result_vol['std_r2']:.4f}",
            'MAE': f"{result_vol['avg_mae']:.6f}",
            'RMSE': f"{result_vol['avg_rmse']:.6f}",
            '샘플': result_vol['n_samples']
        },
        {
            '모델': result_ret['target_type'],
            'R²': f"{result_ret['avg_r2']:.4f} ± {result_ret['std_r2']:.4f}",
            'MAE': f"{result_ret['avg_mae']:.6f}",
            'RMSE': f"{result_ret['avg_rmse']:.6f}",
            '샘플': result_ret['n_samples']
        }
    ])

    print(comparison.to_string(index=False))

    # 결론
    print("\n" + "="*80)
    print("🎯 결론")
    print("="*80)

    print(f"\n1. 변동성 예측:")
    print(f"   - R² = {result_vol['avg_r2']:.4f}: ", end="")
    if result_vol['avg_r2'] > 0.25:
        print("우수한 예측력")
    else:
        print("적정한 예측력")
    print(f"   - 변동성은 지속성(persistence)이 있어 예측 가능")
    print(f"   - 리스크 관리, VIX 옵션 거래, 동적 헤징에 유용")

    print(f"\n2. 수익률 예측:")
    print(f"   - R² = {result_ret['avg_r2']:.4f}: ", end="")
    if result_ret['avg_r2'] > 0.05:
        print("우수한 예측력 (금융에서 드묾)")
    elif result_ret['avg_r2'] > 0:
        print("약한 예측력")
    else:
        print("예측력 없음")
    print(f"   - 수익률은 거의 랜덤워크(random walk) 특성")
    print(f"   - 효율적 시장 가설(EMH)로 인해 예측 어려움")

    print(f"\n3. 권장 사항:")
    if result_vol['avg_r2'] > result_ret['avg_r2'] * 2:
        print(f"   ✅ 변동성 예측 모델 사용 권장")
        print(f"      - 변동성 예측 성능이 수익률 예측보다 {result_vol['avg_r2']/max(result_ret['avg_r2'], 0.001):.1f}배 우수")
        print(f"      - 리스크 관리 중심 전략 구축")
    else:
        print(f"   - 두 모델 모두 활용 가능")

    # 결과 저장
    import json
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'volatility_prediction': {
            'r2_mean': result_vol['avg_r2'],
            'r2_std': result_vol['std_r2'],
            'mae': result_vol['avg_mae'],
            'rmse': result_vol['avg_rmse'],
            'fold_scores': result_vol['fold_scores']
        },
        'return_prediction': {
            'r2_mean': result_ret['avg_r2'],
            'r2_std': result_ret['std_r2'],
            'mae': result_ret['avg_mae'],
            'rmse': result_ret['avg_rmse'],
            'fold_scores': result_ret['fold_scores']
        },
        'comparison': {
            'volatility_better': bool(result_vol['avg_r2'] > result_ret['avg_r2']),
            'performance_ratio': float(result_vol['avg_r2'] / max(result_ret['avg_r2'], 0.001))
        }
    }

    with open('data/raw/model_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 비교 결과 저장: data/raw/model_comparison.json")

    return result_vol, result_ret


if __name__ == "__main__":
    import os
    os.makedirs('data/raw', exist_ok=True)
    result_vol, result_ret = compare_models()
