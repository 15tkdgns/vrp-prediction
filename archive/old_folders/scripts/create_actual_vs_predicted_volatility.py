#!/usr/bin/env python3
"""
실제 변동성 vs 예측 변동성 비교 그래프
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5):
    """Purged K-Fold Cross-Validation"""
    n_samples = len(X)
    fold_size = n_samples // n_splits

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_splits - 1 else n_samples

        train_indices = []
        for i in range(n_samples):
            if i < test_start - purge_length:
                train_indices.append(i)
            elif i >= test_end + embargo_length:
                train_indices.append(i)

        test_indices = list(range(test_start, test_end))
        yield np.array(train_indices), np.array(test_indices)

print("="*70)
print("📊 실제 변동성 vs 예측 변동성 비교 그래프")
print("="*70)

# 1. 데이터 로드
print("\n1️⃣  데이터 로드...")
df = pd.read_csv('data/leak_free/leak_free_sp500_dataset.csv', parse_dates=['Date'])
df = df.rename(columns={'Date': 'date'})
df = df.sort_values('date').reset_index(drop=True)

# 2. Parkinson Realized Volatility 계산
print("\n2️⃣  Parkinson Realized Volatility 계산...")
df['parkinson_vol'] = np.sqrt(
    1/(4*np.log(2)) * (np.log(df['high']/df['low']))**2
)

# 3. Target 생성
print("\n3️⃣  Target 생성...")
targets_rv = []
target_dates = []

for i in range(len(df)):
    if i + 5 < len(df):
        future_vol = df['parkinson_vol'].iloc[i+1:i+6].mean()
        targets_rv.append(future_vol)
        target_dates.append(df['date'].iloc[i+5])

df_target = df.iloc[:len(targets_rv)].copy()
df_target['target_rv'] = targets_rv
df_target['target_date'] = target_dates

# 4. Feature와 Target 준비
feature_cols = [col for col in df.columns if col not in
                ['date', 'open', 'high', 'low', 'close', 'volume', 'returns',
                 'target_return_1d', 'target_return_5d', 'target_direction_1d', 'target_direction_5d',
                 'parkinson_vol']]

X = df_target[feature_cols].values
y_rv = np.array(targets_rv)
dates = df_target['date'].values

print(f"   Feature 수: {len(feature_cols)}")
print(f"   샘플 수: {len(X)}")

# 5. 전체 시계열 예측 생성
print("\n4️⃣  전체 시계열 예측 생성...")
all_predictions = np.zeros(len(y_rv))
all_actuals = y_rv.copy()
prediction_counts = np.zeros(len(y_rv))

scaler = StandardScaler()

# Purged K-Fold로 예측
for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y_rv, n_splits=5, purge_length=5, embargo_length=5)):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y_rv[train_idx]

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    all_predictions[test_idx] += predictions
    prediction_counts[test_idx] += 1

# 평균 예측값
all_predictions = all_predictions / np.maximum(prediction_counts, 1)

# 6. 그래프 생성
print("\n5️⃣  비교 그래프 생성...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# 6-1. 전체 시계열 (2015-2024)
ax1 = axes[0]
ax1.plot(dates, all_actuals, 'k-', linewidth=1.5, label='Actual Volatility (Parkinson)', alpha=0.8)
ax1.plot(dates, all_predictions, 'g-', linewidth=1.5, label='Predicted Volatility (Ridge)', alpha=0.7)
ax1.set_title('SPY Realized Volatility: Actual vs Predicted (Full Period 2015-2024)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Volatility (5-day)', fontsize=11)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# 통계 정보
r2_total = 1 - np.sum((all_actuals - all_predictions)**2) / np.sum((all_actuals - np.mean(all_actuals))**2)
rmse_total = np.sqrt(np.mean((all_actuals - all_predictions)**2))
corr = np.corrcoef(all_actuals, all_predictions)[0, 1]

stats_text = f"""Overall Performance:
R² = {r2_total:.4f}
RMSE = {rmse_total:.6f}
Correlation = {corr:.4f}
Samples = {len(all_actuals):,}"""

ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=9, family='monospace')

# 6-2. 최근 1년 (2024)
ax2 = axes[1]
recent_mask = dates >= pd.Timestamp('2024-01-01')
recent_dates = dates[recent_mask]
recent_actuals = all_actuals[recent_mask]
recent_predictions = all_predictions[recent_mask]

ax2.plot(recent_dates, recent_actuals, 'k-', linewidth=2, label='Actual Volatility', alpha=0.8, marker='o', markersize=3)
ax2.plot(recent_dates, recent_predictions, 'g-', linewidth=2, label='Predicted Volatility', alpha=0.7, marker='s', markersize=3)
ax2.set_title('Recent 1-Year Performance (2024)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Volatility (5-day)', fontsize=11)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# 최근 1년 통계
r2_recent = 1 - np.sum((recent_actuals - recent_predictions)**2) / np.sum((recent_actuals - np.mean(recent_actuals))**2)
rmse_recent = np.sqrt(np.mean((recent_actuals - recent_predictions)**2))

stats_text_recent = f"""Recent Performance:
R² = {r2_recent:.4f}
RMSE = {rmse_recent:.6f}
Samples = {len(recent_actuals):,}"""

ax2.text(0.02, 0.98, stats_text_recent, transform=ax2.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
         fontsize=9, family='monospace')

# 6-3. 예측 오차 시계열
ax3 = axes[2]
errors = all_actuals - all_predictions
ax3.fill_between(dates, 0, errors, where=(errors >= 0), color='red', alpha=0.3, label='Over-prediction')
ax3.fill_between(dates, 0, errors, where=(errors < 0), color='green', alpha=0.3, label='Under-prediction')
ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date', fontsize=11)
ax3.set_ylabel('Error (Actual - Predicted)', fontsize=11)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

# 오차 통계
mae = np.mean(np.abs(errors))
error_std = np.std(errors)

stats_text_error = f"""Error Statistics:
MAE = {mae:.6f}
Std = {error_std:.6f}
Mean Error = {np.mean(errors):.6f}"""

ax3.text(0.02, 0.98, stats_text_error, transform=ax3.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
         fontsize=9, family='monospace')

plt.tight_layout()

# 7. 저장
output_path = Path('dashboard/figures/actual_vs_predicted_volatility.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n💾 그래프 저장: {output_path}")

# 8. 성능 데이터 저장
performance_data = {
    'full_period': {
        'r2': float(r2_total),
        'rmse': float(rmse_total),
        'mae': float(mae),
        'correlation': float(corr),
        'samples': int(len(all_actuals))
    },
    'recent_year_2024': {
        'r2': float(r2_recent),
        'rmse': float(rmse_recent),
        'samples': int(len(recent_actuals))
    },
    'error_statistics': {
        'mae': float(mae),
        'std': float(error_std),
        'mean_error': float(np.mean(errors))
    }
}

json_path = Path('data/raw/volatility_comparison_metrics.json')
with open(json_path, 'w') as f:
    json.dump(performance_data, f, indent=2)
print(f"💾 성능 데이터 저장: {json_path}")

print("\n" + "="*70)
print("✅ 실제 vs 예측 변동성 비교 그래프 생성 완료")
print("="*70)
print(f"\n📊 전체 성능:")
print(f"   R² = {r2_total:.4f}")
print(f"   RMSE = {rmse_total:.6f}")
print(f"   MAE = {mae:.6f}")
print(f"   Correlation = {corr:.4f}")
print(f"\n📊 최근 1년 (2024) 성능:")
print(f"   R² = {r2_recent:.4f}")
print(f"   RMSE = {rmse_recent:.6f}")
print(f"\n생성 파일:")
print(f"   - {output_path}")
print(f"   - {json_path}")
