#!/usr/bin/env python3
"""
Quick Win 개선사항 (1주일 내 즉시 적용 가능)
1. 예측 범위 Clipping
2. Regime-Specific Alpha 튜닝
3. Regime Indicator 특성 추가
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5):
    """Purged K-Fold Cross-Validation"""
    n_samples = len(X)
    fold_size = n_samples // n_splits
    indices = np.arange(n_samples)

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        test_indices = indices[test_start:test_end]

        purge_start = max(0, test_start - purge_length)
        embargo_end = min(n_samples, test_end + embargo_length)

        train_indices = np.concatenate([
            indices[:purge_start],
            indices[embargo_end:]
        ])

        yield train_indices, test_indices

print("="*70)
print("⚡ Quick Win 개선사항 적용")
print("="*70)

# 데이터 로드
print("\n1️⃣  데이터 로드 및 기본 특성 생성...")
spy = yf.Ticker("SPY")
df = spy.history(start="2015-01-01", end="2024-12-31")
df.index = pd.to_datetime(df.index).tz_localize(None)
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

# 타겟 생성
targets = []
for i in range(len(df)):
    if i + 5 < len(df):
        future_returns = df['returns'].iloc[i+1:i+6]
        targets.append(future_returns.std())
    else:
        targets.append(np.nan)
df['target_vol_5d'] = targets

# 기본 특성
for window in [5, 10, 20, 60]:
    df[f'volatility_{window}d'] = df['returns'].rolling(window).std()

for lag in [1, 2, 3, 5, 10, 20]:
    df[f'vol_lag_{lag}'] = df['volatility_20d'].shift(lag)

df['vol_mean_5d'] = df['volatility_20d'].rolling(5).mean()
df['vol_mean_10d'] = df['volatility_20d'].rolling(10).mean()
df['vol_std_5d'] = df['volatility_20d'].rolling(5).std()
df['vol_std_10d'] = df['volatility_20d'].rolling(10).std()

for window in [5, 10, 20]:
    df[f'momentum_{window}d'] = df['returns'].rolling(window).sum()

df['returns_mean_5d'] = df['returns'].rolling(5).mean()
df['returns_mean_10d'] = df['returns'].rolling(10).mean()
df['returns_std_5d'] = df['returns'].rolling(5).std()
df['returns_std_10d'] = df['returns'].rolling(10).std()

df['vol_change_5d'] = df['volatility_20d'].pct_change(5)
df['vol_change_10d'] = df['volatility_20d'].pct_change(10)

df['extreme_returns'] = (df['returns'].abs() > 2 * df['volatility_20d']).astype(int)
df['extreme_count_20d'] = df['extreme_returns'].rolling(20).sum()

# 🆕 Quick Win 1: Regime Indicator 특성 추가
print("\n2️⃣  Quick Win 1: Regime Indicator 특성 추가...")

# Volatility Regime (현재 변동성 / 장기 평균)
vol_ma_60 = df['volatility_20d'].rolling(60).mean()
df['vol_regime'] = (df['volatility_20d'] / vol_ma_60 - 1) * 100

# Volatility of Volatility
df['vol_of_vol_5d'] = df['volatility_20d'].rolling(5).std()
df['vol_of_vol_20d'] = df['volatility_20d'].rolling(20).std()

# Volatility Percentile (최근 60일 중 현재 위치)
df['vol_percentile_60d'] = df['volatility_20d'].rolling(60).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
)

# Regime Transition 감지
vol_ma_20 = df['volatility_20d'].rolling(20).mean()
df['vol_crossing_up'] = ((df['volatility_20d'] > vol_ma_20) &
                         (df['volatility_20d'].shift(1) <= vol_ma_20.shift(1))).astype(int)
df['vol_crossing_down'] = ((df['volatility_20d'] < vol_ma_20) &
                           (df['volatility_20d'].shift(1) >= vol_ma_20.shift(1))).astype(int)

print("   추가된 특성:")
print("   - vol_regime: 장기 평균 대비 변동성 비율")
print("   - vol_of_vol: 변동성의 변동성")
print("   - vol_percentile: 최근 60일 중 현재 위치")
print("   - vol_crossing: Regime 전환 감지")

df = df.dropna()

feature_cols = [col for col in df.columns if col not in
                ['returns', 'target_vol_5d', 'Close', 'Open', 'High', 'Low',
                 'Volume', 'Dividends', 'Stock Splits']]

X = df[feature_cols]
y = df['target_vol_5d']

print(f"\n   전체 데이터: {len(df)} 샘플")
print(f"   특성 수: {len(feature_cols)}개 (기존 26개 → {len(feature_cols)}개)")

# Baseline 성능 (개선 전)
print("\n" + "="*70)
print("📊 Baseline 성능 (개선 전)")
print("="*70)

all_predictions_baseline = np.full(len(X), np.nan)
all_actuals_baseline = np.full(len(X), np.nan)
fold_r2_baseline = []

for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 기존 방식: alpha=1.0, clipping 없음
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    all_predictions_baseline[test_idx] = y_pred
    all_actuals_baseline[test_idx] = y_test.values

    fold_r2 = r2_score(y_test, y_pred)
    fold_r2_baseline.append(fold_r2)
    print(f"Fold {fold_idx}: R² = {fold_r2:.4f}")

test_mask = ~np.isnan(all_predictions_baseline)
baseline_r2 = r2_score(all_actuals_baseline[test_mask], all_predictions_baseline[test_mask])
print(f"\nBaseline CV R² Mean: {np.mean(fold_r2_baseline):.4f} (±{np.std(fold_r2_baseline):.4f})")
print(f"Baseline Test R²: {baseline_r2:.4f}")

# 🆕 Quick Win 2 & 3: Regime-Specific Alpha + Clipping
print("\n" + "="*70)
print("⚡ Quick Win 적용: Regime-Specific Alpha + Clipping")
print("="*70)

all_predictions_improved = np.full(len(X), np.nan)
all_actuals_improved = np.full(len(X), np.nan)
fold_r2_improved = []

for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🆕 Quick Win 2: Regime별 다른 Alpha 사용
    # Low Vol: 강한 정규화 (alpha=10.0)
    # High Vol: 약한 정규화 (alpha=0.5)

    # Regime threshold
    vol_median = y_train.median()

    # Low/High Vol 분리 학습
    low_vol_mask_train = y_train <= vol_median
    high_vol_mask_train = y_train > vol_median

    # Low Vol 모델
    low_vol_model = Ridge(alpha=10.0)  # 강한 정규화
    low_vol_model.fit(X_train_scaled[low_vol_mask_train], y_train[low_vol_mask_train])

    # High Vol 모델
    high_vol_model = Ridge(alpha=0.5)  # 약한 정규화
    high_vol_model.fit(X_train_scaled[high_vol_mask_train], y_train[high_vol_mask_train])

    # 예측: 현재 변동성에 따라 모델 선택
    y_pred = np.zeros(len(X_test))

    for i, (idx, row) in enumerate(X_test.iterrows()):
        current_vol = row['volatility_20d']

        if current_vol <= vol_median:
            y_pred[i] = low_vol_model.predict(X_test_scaled[i:i+1])[0]
        else:
            y_pred[i] = high_vol_model.predict(X_test_scaled[i:i+1])[0]

    # 🆕 Quick Win 3: Clipping (예측 범위 제약)
    y_min = y_train.quantile(0.01)
    y_max = y_train.quantile(0.99)
    y_pred_clipped = np.clip(y_pred, y_min, y_max)

    all_predictions_improved[test_idx] = y_pred_clipped
    all_actuals_improved[test_idx] = y_test.values

    fold_r2 = r2_score(y_test, y_pred_clipped)
    fold_r2_improved.append(fold_r2)
    print(f"Fold {fold_idx}: R² = {fold_r2:.4f} (Baseline: {fold_r2_baseline[fold_idx-1]:.4f}, "
          f"Δ = {fold_r2 - fold_r2_baseline[fold_idx-1]:+.4f})")

test_mask = ~np.isnan(all_predictions_improved)
improved_r2 = r2_score(all_actuals_improved[test_mask], all_predictions_improved[test_mask])

print(f"\nImproved CV R² Mean: {np.mean(fold_r2_improved):.4f} (±{np.std(fold_r2_improved):.4f})")
print(f"Improved Test R²: {improved_r2:.4f}")

# 개선 효과
print("\n" + "="*70)
print("📈 개선 효과 분석")
print("="*70)

print(f"\nCV R² Mean:")
print(f"  Baseline:  {np.mean(fold_r2_baseline):.4f}")
print(f"  Improved:  {np.mean(fold_r2_improved):.4f}")
print(f"  Δ:         {np.mean(fold_r2_improved) - np.mean(fold_r2_baseline):+.4f} "
      f"({(np.mean(fold_r2_improved) / np.mean(fold_r2_baseline) - 1) * 100:+.1f}%)")

print(f"\nCV R² Std (안정성):")
print(f"  Baseline:  {np.std(fold_r2_baseline):.4f}")
print(f"  Improved:  {np.std(fold_r2_improved):.4f}")
print(f"  Δ:         {np.std(fold_r2_improved) - np.std(fold_r2_baseline):+.4f}")

print(f"\nTest R²:")
print(f"  Baseline:  {baseline_r2:.4f}")
print(f"  Improved:  {improved_r2:.4f}")
print(f"  Δ:         {improved_r2 - baseline_r2:+.4f} "
      f"({(improved_r2 / baseline_r2 - 1) * 100:+.1f}%)")

# Regime별 성능 비교
y_test_all_baseline = all_actuals_baseline[test_mask]
y_pred_all_baseline = all_predictions_baseline[test_mask]
y_test_all_improved = all_actuals_improved[test_mask]
y_pred_all_improved = all_predictions_improved[test_mask]

terciles = pd.qcut(y_test_all_baseline, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')

print("\n" + "="*70)
print("📊 Regime별 성능 비교")
print("="*70)

for tercile in ['Low', 'Medium', 'High']:
    mask = terciles == tercile

    baseline_regime_r2 = r2_score(y_test_all_baseline[mask], y_pred_all_baseline[mask])
    improved_regime_r2 = r2_score(y_test_all_improved[mask], y_pred_all_improved[mask])

    print(f"\n{tercile} Volatility ({mask.sum()} 샘플):")
    print(f"  Baseline R²:  {baseline_regime_r2:7.4f}")
    print(f"  Improved R²:  {improved_regime_r2:7.4f}")
    print(f"  Δ:            {improved_regime_r2 - baseline_regime_r2:+7.4f}")

    # 예측 범위 확인
    baseline_pred_mean = y_pred_all_baseline[mask].mean()
    improved_pred_mean = y_pred_all_improved[mask].mean()
    actual_mean = y_test_all_baseline[mask].mean()

    print(f"  실제 평균:     {actual_mean:.6f}")
    print(f"  Baseline 예측: {baseline_pred_mean:.6f} (오차: {abs(baseline_pred_mean - actual_mean) / actual_mean * 100:.1f}%)")
    print(f"  Improved 예측: {improved_pred_mean:.6f} (오차: {abs(improved_pred_mean - actual_mean) / actual_mean * 100:.1f}%)")

print("\n" + "="*70)
print("✅ Quick Win 적용 완료")
print("="*70)

print("""
적용된 개선사항:
1. ✅ Regime Indicator 특성 5개 추가
2. ✅ Regime별 다른 Alpha 사용 (Low: 10.0, High: 0.5)
3. ✅ 예측 범위 Clipping (1%-99% quantile)

예상 결과:
- CV R² Mean: +5~10% 개선
- Low Vol R² 대폭 개선 (음수 → 양수 가능)
- 예측 안정성 향상 (과대예측 방지)

다음 단계:
- VIX 데이터 통합
- GARCH 특성 추가
- Quantile Regression
""")

# 성능 저장
import json
results = {
    "baseline": {
        "cv_r2_mean": float(np.mean(fold_r2_baseline)),
        "cv_r2_std": float(np.std(fold_r2_baseline)),
        "test_r2": float(baseline_r2),
        "cv_scores": [float(r2) for r2 in fold_r2_baseline]
    },
    "quick_win_improved": {
        "cv_r2_mean": float(np.mean(fold_r2_improved)),
        "cv_r2_std": float(np.std(fold_r2_improved)),
        "test_r2": float(improved_r2),
        "cv_scores": [float(r2) for r2 in fold_r2_improved],
        "improvements": [
            "Regime indicator features (5개)",
            "Regime-specific alpha (Low: 10.0, High: 0.5)",
            "Prediction clipping (1%-99% quantile)"
        ]
    },
    "improvement": {
        "cv_r2_delta": float(np.mean(fold_r2_improved) - np.mean(fold_r2_baseline)),
        "cv_r2_pct": float((np.mean(fold_r2_improved) / np.mean(fold_r2_baseline) - 1) * 100),
        "test_r2_delta": float(improved_r2 - baseline_r2),
        "test_r2_pct": float((improved_r2 / baseline_r2 - 1) * 100)
    }
}

with open('data/raw/quick_win_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 결과 저장: data/raw/quick_win_results.json")
