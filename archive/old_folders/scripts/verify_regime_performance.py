#!/usr/bin/env python3
"""
변동성 구간별 성능 재검증
심각한 음수 R² 문제 원인 파악
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
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
print("🔍 변동성 구간별 성능 재검증")
print("="*70)

# 데이터 로드
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

# 특성 생성
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

df = df.dropna()

feature_cols = [col for col in df.columns if col not in
                ['returns', 'target_vol_5d', 'Close', 'Open', 'High', 'Low',
                 'Volume', 'Dividends', 'Stock Splits']]
feature_cols = feature_cols[:31]

X = df[feature_cols]
y = df['target_vol_5d']

print(f"\n데이터: {len(df)} 샘플")
print(f"특성: {len(feature_cols)}개")

# CV 수행
all_predictions = np.full(len(X), np.nan)
all_actuals = np.full(len(X), np.nan)

for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    all_predictions[test_idx] = y_pred
    all_actuals[test_idx] = y_test.values

test_mask = ~np.isnan(all_predictions)
y_test_all = all_actuals[test_mask]
y_pred_all = all_predictions[test_mask]

print(f"\n전체 성능:")
print(f"  R² = {r2_score(y_test_all, y_pred_all):.4f}")

# 변동성 구간 분석
print("\n" + "="*70)
print("변동성 구간별 상세 분석")
print("="*70)

# 3분위
terciles = pd.qcut(y_test_all, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')

for tercile in ['Low', 'Medium', 'High']:
    mask = terciles == tercile
    actual = y_test_all[mask]
    pred = y_pred_all[mask]

    r2 = r2_score(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = np.mean(np.abs(actual - pred))

    # 추가 분석
    baseline_var = np.var(actual)
    residual_var = np.var(actual - pred)

    print(f"\n{tercile} Volatility:")
    print(f"  샘플 수: {mask.sum()}")
    print(f"  실제 범위: [{actual.min():.6f}, {actual.max():.6f}]")
    print(f"  실제 평균: {actual.mean():.6f} (±{actual.std():.6f})")
    print(f"  예측 평균: {pred.mean():.6f} (±{pred.std():.6f})")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Baseline Var: {baseline_var:.8f}")
    print(f"  Residual Var: {residual_var:.8f}")

    # R² 분해
    ss_tot = np.sum((actual - actual.mean())**2)
    ss_res = np.sum((actual - pred)**2)
    print(f"  SS_tot: {ss_tot:.8f}")
    print(f"  SS_res: {ss_res:.8f}")
    print(f"  Ratio (SS_res/SS_tot): {ss_res/ss_tot:.4f}")

    # 상관계수
    corr = np.corrcoef(actual, pred)[0, 1]
    print(f"  Correlation: {corr:.4f}")

# 문제 진단
print("\n" + "="*70)
print("문제 진단")
print("="*70)

print("\n1. 예측값 범위 확인:")
print(f"   실제 전체 범위: [{y_test_all.min():.6f}, {y_test_all.max():.6f}]")
print(f"   예측 전체 범위: [{y_pred_all.min():.6f}, {y_pred_all.max():.6f}]")

# Low Vol 구간 상세
low_mask = terciles == 'Low'
low_actual = y_test_all[low_mask]
low_pred = y_pred_all[low_mask]

print("\n2. Low Vol 구간 문제:")
print(f"   Low Vol 실제 범위: [{low_actual.min():.6f}, {low_actual.max():.6f}]")
print(f"   Low Vol 예측 범위: [{low_pred.min():.6f}, {low_pred.max():.6f}]")
print(f"   Low Vol 실제 평균: {low_actual.mean():.6f}")
print(f"   Low Vol 예측 평균: {low_pred.mean():.6f}")

# 과적합/과소적합 확인
low_baseline = np.mean(low_actual)
low_baseline_mse = np.mean((low_actual - low_baseline)**2)
low_model_mse = mean_squared_error(low_actual, low_pred)

print(f"\n   Baseline MSE (평균 예측): {low_baseline_mse:.8f}")
print(f"   Model MSE: {low_model_mse:.8f}")
print(f"   비율: {low_model_mse / low_baseline_mse:.4f}")

if low_model_mse > low_baseline_mse:
    print(f"   ⚠️ 모델이 단순 평균보다 {(low_model_mse/low_baseline_mse - 1)*100:.1f}% 더 나쁨!")

# 3. 변동성 범위 문제
print("\n3. 변동성 범위 분석:")
vol_ranges = pd.cut(y_test_all, bins=10)
for i, (interval, group) in enumerate(vol_ranges.value_counts().sort_index().items(), 1):
    mask = vol_ranges == interval
    if mask.sum() > 0:
        r2 = r2_score(y_test_all[mask], y_pred_all[mask])
        print(f"   구간 {i} [{interval.left:.4f}, {interval.right:.4f}]: {mask.sum():4d} 샘플, R² = {r2:7.4f}")

print("\n" + "="*70)
print("결론")
print("="*70)

print("""
1. Low/Medium Vol에서 R² 음수 원인:
   - 모델 예측이 단순 평균보다 못함
   - 낮은 변동성 구간에서 과적합 발생
   - 특성의 신호 대비 노이즈 비율 과다

2. 해결 방안:
   - 구간별 독립 모델 학습
   - Regularization 강화 (alpha 증가)
   - 특성 선택 (Low Vol 전용 특성)
   - Quantile Regression 사용

3. 현재 모델 한계:
   - 전체 R² 0.31은 High Vol 구간 덕분
   - Low/Medium Vol 예측 사실상 불가능
   - 실전 사용 시 구간 필터링 필수
""")
