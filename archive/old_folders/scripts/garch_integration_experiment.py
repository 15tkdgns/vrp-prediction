#!/usr/bin/env python3
"""
GARCH 통합 실험
- GARCH(1,1) 모델로 조건부 변동성 추정
- GARCH 예측값을 특성으로 사용하여 예측력 향상 기대
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from arch import arch_model
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
print("🔬 GARCH 통합 실험")
print("="*70)

# 1. 데이터 로드
print("\n1️⃣  데이터 로드...")
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

# 2. GARCH 모델 적합
print("\n2️⃣  GARCH(1,1) 모델 적합...")
print("   (시간이 다소 소요될 수 있습니다...)")

# 수익률 데이터 준비 (GARCH는 %로 계산하므로 *100)
returns_clean = df['returns'].dropna() * 100

try:
    # GARCH(1,1) 모델
    garch_model = arch_model(returns_clean, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp='off', show_warning=False)

    print("   GARCH 모델 적합 완료!")
    print(f"   omega: {garch_fit.params['omega']:.6f}")
    print(f"   alpha[1]: {garch_fit.params['alpha[1]']:.6f}")
    print(f"   beta[1]: {garch_fit.params['beta[1]']:.6f}")

    # GARCH 조건부 변동성
    garch_vol = garch_fit.conditional_volatility / 100  # 다시 소수로 변환

    # GARCH 예측값을 데이터프레임에 추가
    df_garch = pd.DataFrame({'garch_vol': garch_vol}, index=returns_clean.index)
    df = df.join(df_garch, how='left')

    # GARCH 표준화 잔차
    df['garch_residual'] = df['returns'] / df['garch_vol']

    garch_available = True
    print(f"   GARCH 변동성: {df['garch_vol'].notna().sum()} 샘플")
    print(f"   GARCH 범위: [{df['garch_vol'].min():.6f}, {df['garch_vol'].max():.6f}]")

except Exception as e:
    print(f"   ⚠️ GARCH 모델 적합 실패: {e}")
    print("   대체 방법: EWMA 사용")
    garch_available = False

    # EWMA (Exponentially Weighted Moving Average) 대체
    df['garch_vol'] = df['returns'].ewm(span=20).std()
    df['garch_residual'] = df['returns'] / df['garch_vol']

# 3. 기본 특성 생성
print("\n3️⃣  기본 특성 생성...")

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

# 4. GARCH 기반 특성 생성
print("\n4️⃣  GARCH 기반 특성 생성...")

# GARCH 변동성 lag
for lag in [1, 2, 3, 5, 10]:
    df[f'garch_vol_lag_{lag}'] = df['garch_vol'].shift(lag)

# GARCH 변동성 변화율
df['garch_vol_change_5d'] = df['garch_vol'].pct_change(5)
df['garch_vol_change_10d'] = df['garch_vol'].pct_change(10)

# GARCH vs Realized Vol 비교
df['garch_rv_ratio'] = df['garch_vol'] / df['volatility_20d']
df['garch_rv_spread'] = df['garch_vol'] - df['volatility_20d']

# GARCH 잔차 통계
df['garch_residual_abs'] = df['garch_residual'].abs()
df['garch_residual_sq'] = df['garch_residual'] ** 2

# GARCH 잔차 롤링 통계
df['garch_residual_ma_5'] = df['garch_residual'].rolling(5).mean()
df['garch_residual_std_5'] = df['garch_residual'].rolling(5).std()

# GARCH 변동성 persistence
df['garch_vol_ma_5'] = df['garch_vol'].rolling(5).mean()
df['garch_vol_ma_20'] = df['garch_vol'].rolling(20).mean()
df['garch_vol_momentum'] = df['garch_vol'] / df['garch_vol_ma_20']

print("   추가된 GARCH 특성:")
print("   - garch_vol: GARCH 조건부 변동성")
print("   - garch_vol_lag_*: GARCH 변동성 lag (1, 2, 3, 5, 10)")
print("   - garch_vol_change_*: GARCH 변동성 변화율")
print("   - garch_rv_ratio: GARCH / Realized Vol 비율")
print("   - garch_residual_*: GARCH 표준화 잔차 통계")
print("   - garch_vol_momentum: GARCH 변동성 모멘텀")

df = df.dropna()

# 5. Baseline 모델 (GARCH 없음)
print("\n" + "="*70)
print("📊 Baseline 성능 (GARCH 없음)")
print("="*70)

baseline_features = [col for col in df.columns if col not in
                     ['returns', 'target_vol_5d', 'Close', 'Open', 'High', 'Low',
                      'Volume', 'Dividends', 'Stock Splits',
                      'garch_vol', 'garch_residual', 'garch_vol_lag_1', 'garch_vol_lag_2',
                      'garch_vol_lag_3', 'garch_vol_lag_5', 'garch_vol_lag_10',
                      'garch_vol_change_5d', 'garch_vol_change_10d',
                      'garch_rv_ratio', 'garch_rv_spread',
                      'garch_residual_abs', 'garch_residual_sq',
                      'garch_residual_ma_5', 'garch_residual_std_5',
                      'garch_vol_ma_5', 'garch_vol_ma_20', 'garch_vol_momentum']]

X_baseline = df[baseline_features]
y = df['target_vol_5d']

print(f"\n데이터: {len(df)} 샘플")
print(f"Baseline 특성: {len(baseline_features)}개")

fold_r2_baseline = []
all_predictions_baseline = np.full(len(X_baseline), np.nan)
all_actuals_baseline = np.full(len(X_baseline), np.nan)

for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X_baseline, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train = X_baseline.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X_baseline.iloc[test_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    all_predictions_baseline[test_idx] = y_pred
    all_actuals_baseline[test_idx] = y_test.values

    fold_r2 = r2_score(y_test, y_pred)
    fold_r2_baseline.append(fold_r2)
    print(f"Fold {fold_idx}: R² = {fold_r2:.4f}")

baseline_cv_r2 = np.mean(fold_r2_baseline)
baseline_cv_std = np.std(fold_r2_baseline)

print(f"\nBaseline CV R² Mean: {baseline_cv_r2:.4f} (±{baseline_cv_std:.4f})")

# 6. GARCH 통합 모델
print("\n" + "="*70)
print("🔬 GARCH 통합 모델")
print("="*70)

garch_features = baseline_features + [
    'garch_vol', 'garch_residual',
    'garch_vol_lag_1', 'garch_vol_lag_2', 'garch_vol_lag_3', 'garch_vol_lag_5', 'garch_vol_lag_10',
    'garch_vol_change_5d', 'garch_vol_change_10d',
    'garch_rv_ratio', 'garch_rv_spread',
    'garch_residual_abs', 'garch_residual_sq',
    'garch_residual_ma_5', 'garch_residual_std_5',
    'garch_vol_ma_5', 'garch_vol_ma_20', 'garch_vol_momentum'
]

X_garch = df[garch_features]

print(f"\nGARCH 통합 특성: {len(garch_features)}개 (+{len(garch_features) - len(baseline_features)}개)")

fold_r2_garch = []
all_predictions_garch = np.full(len(X_garch), np.nan)
all_actuals_garch = np.full(len(X_garch), np.nan)

for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X_garch, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train = X_garch.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X_garch.iloc[test_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    all_predictions_garch[test_idx] = y_pred
    all_actuals_garch[test_idx] = y_test.values

    fold_r2 = r2_score(y_test, y_pred)
    fold_r2_garch.append(fold_r2)
    print(f"Fold {fold_idx}: R² = {fold_r2:.4f} (Baseline: {fold_r2_baseline[fold_idx-1]:.4f}, "
          f"Δ = {fold_r2 - fold_r2_baseline[fold_idx-1]:+.4f})")

garch_cv_r2 = np.mean(fold_r2_garch)
garch_cv_std = np.std(fold_r2_garch)

print(f"\nGARCH 통합 CV R² Mean: {garch_cv_r2:.4f} (±{garch_cv_std:.4f})")

# 7. 결과 비교
print("\n" + "="*70)
print("📈 GARCH 통합 효과 분석")
print("="*70)

print(f"\nCV R² Mean:")
print(f"  Baseline (GARCH 없음):  {baseline_cv_r2:.4f}")
print(f"  GARCH 통합:             {garch_cv_r2:.4f}")
print(f"  Δ:                      {garch_cv_r2 - baseline_cv_r2:+.4f} "
      f"({(garch_cv_r2 / baseline_cv_r2 - 1) * 100:+.1f}%)")

print(f"\nCV R² Std (안정성):")
print(f"  Baseline:  {baseline_cv_std:.4f}")
print(f"  GARCH:     {garch_cv_std:.4f}")
print(f"  Δ:         {garch_cv_std - baseline_cv_std:+.4f}")

# Fold별 개선 분석
print("\nFold별 개선 효과:")
improved_folds = 0
for i in range(5):
    improvement = fold_r2_garch[i] - fold_r2_baseline[i]
    if improvement > 0:
        improved_folds += 1
    print(f"  Fold {i+1}: {improvement:+.4f} {'✅' if improvement > 0 else '❌'}")

print(f"\n개선된 Fold: {improved_folds}/5")

# 8. 변동성 구간별 성능
print("\n" + "="*70)
print("📊 변동성 구간별 성능 분석")
print("="*70)

test_mask = ~np.isnan(all_predictions_garch)
y_test_all = all_actuals_garch[test_mask]
y_pred_baseline = all_predictions_baseline[test_mask]
y_pred_garch = all_predictions_garch[test_mask]

vol_terciles = pd.qcut(y_test_all, q=3, labels=['Low Vol', 'Medium Vol', 'High Vol'], duplicates='drop')

for tercile in ['Low Vol', 'Medium Vol', 'High Vol']:
    mask = vol_terciles == tercile

    baseline_r2 = r2_score(y_test_all[mask], y_pred_baseline[mask])
    garch_r2 = r2_score(y_test_all[mask], y_pred_garch[mask])

    print(f"\n{tercile} ({mask.sum()} 샘플):")
    print(f"  Baseline R²:  {baseline_r2:7.4f}")
    print(f"  GARCH R²:     {garch_r2:7.4f}")
    print(f"  Δ:            {garch_r2 - baseline_r2:+7.4f} {'✅' if garch_r2 > baseline_r2 else '❌'}")

# 9. Feature Importance
print("\n" + "="*70)
print("🔍 GARCH 특성 중요도 분석")
print("="*70)

X_train = X_garch.iloc[train_idx]
y_train = y.iloc[train_idx]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

garch_feature_names = [f for f in garch_features if 'garch' in f]
garch_feature_indices = [garch_features.index(f) for f in garch_feature_names]
garch_coefficients = model.coef_[garch_feature_indices]

importance_df = pd.DataFrame({
    'feature': garch_feature_names,
    'coefficient': garch_coefficients,
    'abs_coefficient': np.abs(garch_coefficients)
}).sort_values('abs_coefficient', ascending=False)

print("\nTop 10 GARCH 특성 (절대 계수 기준):")
for i, (idx, row) in enumerate(importance_df.head(10).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:25s}: {row['coefficient']:+.6f}")

# 10. 결과 저장
print("\n" + "="*70)
print("💾 결과 저장")
print("="*70)

import json
results = {
    "experiment": "GARCH Integration",
    "date": pd.Timestamp.now().isoformat(),
    "garch_model": "GARCH(1,1)" if garch_available else "EWMA (fallback)",
    "baseline": {
        "features": len(baseline_features),
        "cv_r2_mean": float(baseline_cv_r2),
        "cv_r2_std": float(baseline_cv_std),
        "cv_scores": [float(r2) for r2 in fold_r2_baseline]
    },
    "garch_integrated": {
        "features": len(garch_features),
        "added_features": len(garch_features) - len(baseline_features),
        "cv_r2_mean": float(garch_cv_r2),
        "cv_r2_std": float(garch_cv_std),
        "cv_scores": [float(r2) for r2 in fold_r2_garch]
    },
    "improvement": {
        "cv_r2_delta": float(garch_cv_r2 - baseline_cv_r2),
        "cv_r2_pct": float((garch_cv_r2 / baseline_cv_r2 - 1) * 100),
        "improved_folds": int(improved_folds),
        "total_folds": 5
    },
    "top_garch_features": importance_df.head(10).to_dict('records'),
    "conclusion": "GARCH 통합 효과 검증 완료"
}

with open('data/raw/garch_integration_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ 결과 저장: data/raw/garch_integration_results.json")

# 11. 최종 결론
print("\n" + "="*70)
print("🎯 최종 결론")
print("="*70)

if garch_cv_r2 > baseline_cv_r2:
    improvement_pct = (garch_cv_r2 / baseline_cv_r2 - 1) * 100
    print(f"""
✅ GARCH 통합 성공!

성능 개선:
  - CV R² Mean: {baseline_cv_r2:.4f} → {garch_cv_r2:.4f} ({improvement_pct:+.1f}%)
  - 개선된 Fold: {improved_folds}/5

주요 GARCH 특성:
  - {importance_df.iloc[0]['feature']}
  - {importance_df.iloc[1]['feature']}
  - {importance_df.iloc[2]['feature']}

권장사항:
  ✅ GARCH 특성을 최종 모델에 포함
  ✅ 조건부 이분산성 모델링으로 예측력 향상
  ✅ 다음 단계: LSTM 프로토타입
""")
else:
    decline_pct = (garch_cv_r2 / baseline_cv_r2 - 1) * 100
    print(f"""
❌ GARCH 통합 실패

성능 변화:
  - CV R² Mean: {baseline_cv_r2:.4f} → {garch_cv_r2:.4f} ({decline_pct:+.1f}%)
  - 개선된 Fold: {improved_folds}/5

문제 분석:
  - GARCH 특성이 노이즈로 작용
  - 과적합 또는 정보 중복
  - Feature selection 필요

권장사항:
  ❌ GARCH 특성 제거 또는 선별적 사용
  🔍 다음 단계: LSTM 프로토타입 (마지막 시도)
""")
