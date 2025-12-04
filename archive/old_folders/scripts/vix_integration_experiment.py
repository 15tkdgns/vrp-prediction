#!/usr/bin/env python3
"""
VIX 통합 실험
- VIX 데이터 추가하여 변동성 예측 개선
- Forward-looking 지표로 예측력 향상 기대
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
print("🔬 VIX 통합 실험")
print("="*70)

# 1. SPY 데이터 로드
print("\n1️⃣  SPY 데이터 로드...")
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

# 2. VIX 데이터 로드
print("\n2️⃣  VIX 데이터 로드...")
vix = yf.Ticker("^VIX")
vix_df = vix.history(start="2015-01-01", end="2024-12-31")
vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)

# VIX를 SPY 데이터프레임에 병합
df = df.join(vix_df[['Close']].rename(columns={'Close': 'VIX'}), how='left')

# VIX 결측치 처리 (forward fill)
df['VIX'] = df['VIX'].fillna(method='ffill')

print(f"   VIX 데이터: {df['VIX'].notna().sum()} 샘플")
print(f"   VIX 범위: [{df['VIX'].min():.2f}, {df['VIX'].max():.2f}]")
print(f"   VIX 평균: {df['VIX'].mean():.2f}")

# 3. 기본 특성 생성 (V0 원본)
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

# 4. VIX 기반 특성 생성
print("\n4️⃣  VIX 기반 특성 생성...")

# VIX를 연율화된 변동성으로 변환 (VIX는 %로 표시되므로 /100)
df['vix_annualized'] = df['VIX'] / 100

# VIX 변화율
df['vix_change_1d'] = df['VIX'].pct_change(1)
df['vix_change_5d'] = df['VIX'].pct_change(5)
df['vix_change_10d'] = df['VIX'].pct_change(10)

# VIX 롤링 통계
df['vix_ma_5'] = df['VIX'].rolling(5).mean()
df['vix_ma_20'] = df['VIX'].rolling(20).mean()
df['vix_ma_60'] = df['VIX'].rolling(60).mean()
df['vix_std_20'] = df['VIX'].rolling(20).std()

# VIX 모멘텀 (현재 VIX / 이동평균)
df['vix_momentum_20'] = df['VIX'] / df['vix_ma_20']
df['vix_momentum_60'] = df['VIX'] / df['vix_ma_60']

# VIX Percentile (최근 60일 중 위치)
df['vix_percentile_60d'] = df['VIX'].rolling(60).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
)

# VIX vs Realized Volatility Spread
# VIX는 implied vol (연율화), 일간 변동성을 연율화해서 비교
df['realized_vol_annualized'] = df['volatility_20d'] * np.sqrt(252)
df['vix_rv_spread'] = df['vix_annualized'] - df['realized_vol_annualized']
df['vix_rv_ratio'] = df['vix_annualized'] / df['realized_vol_annualized']

# VIX Regime Indicator
vix_median = df['VIX'].median()
df['vix_regime'] = (df['VIX'] > vix_median).astype(int)  # 0: Low VIX, 1: High VIX

# VIX 급등/급락 감지
df['vix_spike'] = (df['vix_change_1d'] > 0.10).astype(int)  # 10% 이상 상승
df['vix_drop'] = (df['vix_change_1d'] < -0.10).astype(int)  # 10% 이상 하락

print("   추가된 VIX 특성:")
print("   - vix_annualized: VIX를 연율화 변동성으로 변환")
print("   - vix_change_*: VIX 변화율 (1d, 5d, 10d)")
print("   - vix_ma_*: VIX 이동평균 (5, 20, 60)")
print("   - vix_momentum_*: VIX 모멘텀")
print("   - vix_percentile: VIX 분위수")
print("   - vix_rv_spread: VIX vs Realized Vol 스프레드")
print("   - vix_regime: VIX 구간 (High/Low)")
print("   - vix_spike/drop: VIX 급등/급락 감지")

df = df.dropna()

# 5. Baseline 모델 (VIX 없음)
print("\n" + "="*70)
print("📊 Baseline 성능 (VIX 없음)")
print("="*70)

baseline_features = [col for col in df.columns if col not in
                     ['returns', 'target_vol_5d', 'Close', 'Open', 'High', 'Low',
                      'Volume', 'Dividends', 'Stock Splits', 'VIX',
                      'vix_annualized', 'vix_change_1d', 'vix_change_5d', 'vix_change_10d',
                      'vix_ma_5', 'vix_ma_20', 'vix_ma_60', 'vix_std_20',
                      'vix_momentum_20', 'vix_momentum_60', 'vix_percentile_60d',
                      'realized_vol_annualized', 'vix_rv_spread', 'vix_rv_ratio',
                      'vix_regime', 'vix_spike', 'vix_drop']]

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

# 6. VIX 통합 모델
print("\n" + "="*70)
print("🔬 VIX 통합 모델")
print("="*70)

vix_features = baseline_features + [
    'vix_annualized', 'vix_change_1d', 'vix_change_5d', 'vix_change_10d',
    'vix_ma_5', 'vix_ma_20', 'vix_ma_60', 'vix_std_20',
    'vix_momentum_20', 'vix_momentum_60', 'vix_percentile_60d',
    'vix_rv_spread', 'vix_rv_ratio', 'vix_regime', 'vix_spike', 'vix_drop'
]

X_vix = df[vix_features]

print(f"\nVIX 통합 특성: {len(vix_features)}개 (+{len(vix_features) - len(baseline_features)}개)")

fold_r2_vix = []
all_predictions_vix = np.full(len(X_vix), np.nan)
all_actuals_vix = np.full(len(X_vix), np.nan)

for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X_vix, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train = X_vix.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X_vix.iloc[test_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    all_predictions_vix[test_idx] = y_pred
    all_actuals_vix[test_idx] = y_test.values

    fold_r2 = r2_score(y_test, y_pred)
    fold_r2_vix.append(fold_r2)
    print(f"Fold {fold_idx}: R² = {fold_r2:.4f} (Baseline: {fold_r2_baseline[fold_idx-1]:.4f}, "
          f"Δ = {fold_r2 - fold_r2_baseline[fold_idx-1]:+.4f})")

vix_cv_r2 = np.mean(fold_r2_vix)
vix_cv_std = np.std(fold_r2_vix)

print(f"\nVIX 통합 CV R² Mean: {vix_cv_r2:.4f} (±{vix_cv_std:.4f})")

# 7. 결과 비교
print("\n" + "="*70)
print("📈 VIX 통합 효과 분석")
print("="*70)

print(f"\nCV R² Mean:")
print(f"  Baseline (VIX 없음):  {baseline_cv_r2:.4f}")
print(f"  VIX 통합:             {vix_cv_r2:.4f}")
print(f"  Δ:                    {vix_cv_r2 - baseline_cv_r2:+.4f} "
      f"({(vix_cv_r2 / baseline_cv_r2 - 1) * 100:+.1f}%)")

print(f"\nCV R² Std (안정성):")
print(f"  Baseline:  {baseline_cv_std:.4f}")
print(f"  VIX 통합:  {vix_cv_std:.4f}")
print(f"  Δ:         {vix_cv_std - baseline_cv_std:+.4f}")

# Fold별 개선 분석
print("\nFold별 개선 효과:")
for i in range(5):
    improvement = fold_r2_vix[i] - fold_r2_baseline[i]
    print(f"  Fold {i+1}: {improvement:+.4f} {'✅' if improvement > 0 else '❌'}")

improved_folds = sum(1 for i in range(5) if fold_r2_vix[i] > fold_r2_baseline[i])
print(f"\n개선된 Fold: {improved_folds}/5")

# 8. VIX 구간별 성능
print("\n" + "="*70)
print("📊 VIX 구간별 성능 분석")
print("="*70)

test_mask = ~np.isnan(all_predictions_vix)
y_test_all = all_actuals_vix[test_mask]
y_pred_baseline = all_predictions_baseline[test_mask]
y_pred_vix = all_predictions_vix[test_mask]
df_test = df.iloc[test_mask]

# VIX 3분위
vix_terciles = pd.qcut(df_test['VIX'], q=3, labels=['Low VIX', 'Medium VIX', 'High VIX'], duplicates='drop')

for tercile in ['Low VIX', 'Medium VIX', 'High VIX']:
    mask = vix_terciles == tercile

    baseline_r2 = r2_score(y_test_all[mask], y_pred_baseline[mask])
    vix_r2 = r2_score(y_test_all[mask], y_pred_vix[mask])

    print(f"\n{tercile} ({mask.sum()} 샘플):")
    print(f"  VIX 범위: [{df_test['VIX'][mask].min():.2f}, {df_test['VIX'][mask].max():.2f}]")
    print(f"  Baseline R²:  {baseline_r2:7.4f}")
    print(f"  VIX 통합 R²:  {vix_r2:7.4f}")
    print(f"  Δ:            {vix_r2 - baseline_r2:+7.4f} {'✅' if vix_r2 > baseline_r2 else '❌'}")

# 9. Feature Importance (VIX 관련 특성)
print("\n" + "="*70)
print("🔍 VIX 특성 중요도 분석")
print("="*70)

# 마지막 fold 모델로 계수 확인
X_train = X_vix.iloc[train_idx]
y_train = y.iloc[train_idx]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# VIX 관련 특성의 계수
vix_feature_names = [f for f in vix_features if 'vix' in f]
vix_feature_indices = [vix_features.index(f) for f in vix_feature_names]
vix_coefficients = model.coef_[vix_feature_indices]

importance_df = pd.DataFrame({
    'feature': vix_feature_names,
    'coefficient': vix_coefficients,
    'abs_coefficient': np.abs(vix_coefficients)
}).sort_values('abs_coefficient', ascending=False)

print("\nTop 10 VIX 특성 (절대 계수 기준):")
for i, (idx, row) in enumerate(importance_df.head(10).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:25s}: {row['coefficient']:+.6f}")

# 10. 결과 저장
print("\n" + "="*70)
print("💾 결과 저장")
print("="*70)

import json
results = {
    "experiment": "VIX Integration",
    "date": pd.Timestamp.now().isoformat(),
    "baseline": {
        "features": len(baseline_features),
        "cv_r2_mean": float(baseline_cv_r2),
        "cv_r2_std": float(baseline_cv_std),
        "cv_scores": [float(r2) for r2 in fold_r2_baseline]
    },
    "vix_integrated": {
        "features": len(vix_features),
        "added_features": len(vix_features) - len(baseline_features),
        "cv_r2_mean": float(vix_cv_r2),
        "cv_r2_std": float(vix_cv_std),
        "cv_scores": [float(r2) for r2 in fold_r2_vix]
    },
    "improvement": {
        "cv_r2_delta": float(vix_cv_r2 - baseline_cv_r2),
        "cv_r2_pct": float((vix_cv_r2 / baseline_cv_r2 - 1) * 100),
        "improved_folds": int(improved_folds),
        "total_folds": 5
    },
    "top_vix_features": importance_df.head(10).to_dict('records'),
    "conclusion": "VIX 통합 효과 검증 완료"
}

with open('data/raw/vix_integration_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ 결과 저장: data/raw/vix_integration_results.json")

# 11. 최종 결론
print("\n" + "="*70)
print("🎯 최종 결론")
print("="*70)

if vix_cv_r2 > baseline_cv_r2:
    improvement_pct = (vix_cv_r2 / baseline_cv_r2 - 1) * 100
    print(f"""
✅ VIX 통합 성공!

성능 개선:
  - CV R² Mean: {baseline_cv_r2:.4f} → {vix_cv_r2:.4f} ({improvement_pct:+.1f}%)
  - 개선된 Fold: {improved_folds}/5

주요 VIX 특성:
  - {importance_df.iloc[0]['feature']}
  - {importance_df.iloc[1]['feature']}
  - {importance_df.iloc[2]['feature']}

권장사항:
  ✅ VIX 특성을 최종 모델에 포함
  ✅ Forward-looking 정보로 예측력 향상
  ✅ 다음 단계: GARCH 통합 실험
""")
else:
    decline_pct = (vix_cv_r2 / baseline_cv_r2 - 1) * 100
    print(f"""
❌ VIX 통합 실패

성능 변화:
  - CV R² Mean: {baseline_cv_r2:.4f} → {vix_cv_r2:.4f} ({decline_pct:+.1f}%)
  - 개선된 Fold: {improved_folds}/5

문제 분석:
  - VIX 특성이 노이즈로 작용
  - 과적합 또는 정보 중복
  - Feature selection 필요

권장사항:
  ❌ VIX 특성 제거 또는 선별적 사용
  🔍 다른 접근 방법 시도 (GARCH, LSTM)
""")
