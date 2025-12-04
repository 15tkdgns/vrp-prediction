#!/usr/bin/env python3
"""
Realized Volatility 최종 차트 생성
- 실제 vs 예측 시각화
- 성과 요약
- 최종 모델 비교
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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
print("📊 Realized Volatility 최종 차트 생성")
print("="*70)

# 데이터 로드
print("\n1️⃣  데이터 로드 및 모델 학습...")
spy = yf.Ticker("SPY")
df = spy.history(start="2015-01-01", end="2024-12-31")
df.index = pd.to_datetime(df.index).tz_localize(None)
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

# RV 타겟
df['parkinson_vol'] = np.sqrt(
    1/(4*np.log(2)) * (np.log(df['High']/df['Low']))**2
)

targets_rv = []
for i in range(len(df)):
    if i + 5 < len(df):
        future_vol = df['parkinson_vol'].iloc[i+1:i+6].mean()
        targets_rv.append(future_vol)
    else:
        targets_rv.append(np.nan)
df['target_vol_rv'] = targets_rv

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

df['rv_lag_1'] = df['parkinson_vol'].shift(1)
df['rv_lag_5'] = df['parkinson_vol'].shift(5)
df['rv_ma_5'] = df['parkinson_vol'].rolling(5).mean()
df['rv_ma_20'] = df['parkinson_vol'].rolling(20).mean()

df = df.dropna()

feature_cols = [col for col in df.columns if col not in
                ['returns', 'target_vol_rv', 'Close', 'Open', 'High', 'Low',
                 'Volume', 'Dividends', 'Stock Splits', 'parkinson_vol']]

X = df[feature_cols]
y = df['target_vol_rv']

# 모델 학습 및 예측
print("   모델 학습 중...")
all_predictions = np.full(len(X), np.nan)
all_actuals = np.full(len(X), np.nan)
fold_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    all_predictions[test_idx] = y_pred
    all_actuals[test_idx] = y_test.values

    fold_scores.append(r2_score(y_test, y_pred))

test_mask = ~np.isnan(all_predictions)
y_test_all = all_actuals[test_mask]
y_pred_all = all_predictions[test_mask]
df_test = df.iloc[test_mask].copy()
df_test['predictions'] = y_pred_all
df_test['actuals'] = y_test_all

overall_r2 = r2_score(y_test_all, y_pred_all)
cv_r2_mean = np.mean(fold_scores)
cv_r2_std = np.std(fold_scores)
rmse = np.sqrt(mean_squared_error(y_test_all, y_pred_all))
mae = mean_absolute_error(y_test_all, y_pred_all)

print(f"   CV R² Mean: {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
print(f"   Test R² (Overall): {overall_r2:.4f}")

# 최종 차트 생성
print("\n2️⃣  최종 차트 생성...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ===== 메인 차트: 시계열 비교 (최근 1년) =====
ax_main = fig.add_subplot(gs[0, :])
recent_data = df_test.iloc[-250:]

ax_main.plot(recent_data.index, recent_data['actuals'],
             label='Actual Volatility (Parkinson)', color='black', linewidth=2, alpha=0.8)
ax_main.plot(recent_data.index, recent_data['predictions'],
             label='Predicted Volatility (Ridge)', color='#2ecc71', linewidth=2, alpha=0.7)
ax_main.fill_between(recent_data.index, recent_data['actuals'], recent_data['predictions'],
                      alpha=0.2, color='gray')
ax_main.set_ylabel('Volatility (5-day)', fontsize=11)
ax_main.set_title(f'SPY Realized Volatility: Actual vs Predicted (CV R² = {cv_r2_mean:.4f})',
                  fontsize=14, fontweight='bold')
ax_main.legend(loc='upper right', fontsize=10)
ax_main.grid(True, alpha=0.3)

# ===== 산점도 =====
ax1 = fig.add_subplot(gs[1, 0])
ax1.scatter(y_test_all, y_pred_all, alpha=0.4, s=8, color='#2ecc71')

min_val = min(y_test_all.min(), y_pred_all.min())
max_val = max(y_test_all.max(), y_pred_all.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

z = np.polyfit(y_test_all, y_pred_all, 1)
p = np.poly1d(z)
ax1.plot(y_test_all, p(y_test_all), "b-", linewidth=2,
         label=f'Trend (R² = {overall_r2:.4f})')

ax1.set_xlabel('Actual Volatility', fontsize=10)
ax1.set_ylabel('Predicted Volatility', fontsize=10)
ax1.set_title('Prediction Scatter Plot', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# ===== Fold별 성능 =====
ax2 = fig.add_subplot(gs[1, 1])
folds = np.arange(1, 6)
colors = ['#2ecc71' if r2 > 0.3 else '#f39c12' if r2 > 0 else '#e74c3c' for r2 in fold_scores]
bars = ax2.bar(folds, fold_scores, color=colors, alpha=0.8, edgecolor='black')
ax2.axhline(y=cv_r2_mean, color='blue', linestyle='--', linewidth=2,
            label=f'Mean: {cv_r2_mean:.4f}')
ax2.set_xlabel('Fold', fontsize=10)
ax2.set_ylabel('R² Score', fontsize=10)
ax2.set_title('Cross-Validation Performance', fontsize=12, fontweight='bold')
ax2.set_xticks(folds)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

for i, (fold, score) in enumerate(zip(folds, fold_scores)):
    ax2.text(fold, score + 0.02, f'{score:.3f}', ha='center', fontsize=9, fontweight='bold')

# ===== 에러 분포 =====
ax3 = fig.add_subplot(gs[1, 2])
errors = y_test_all - y_pred_all
ax3.hist(errors, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax3.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2,
            label=f'Mean: {errors.mean():.6f}')
ax3.set_xlabel('Prediction Error', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# ===== 변동성 구간별 성능 =====
ax4 = fig.add_subplot(gs[2, 0])
terciles = pd.qcut(y_test_all, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
regime_r2 = []
regime_labels = []

for tercile in ['Low', 'Medium', 'High']:
    mask = terciles == tercile
    if mask.sum() > 0:
        r2 = r2_score(y_test_all[mask], y_pred_all[mask])
        regime_r2.append(r2)
        regime_labels.append(f'{tercile}\n({mask.sum()})')

colors_regime = ['#e74c3c' if r2 < 0 else '#f39c12' if r2 < 0.2 else '#2ecc71' for r2 in regime_r2]
ax4.bar(regime_labels, regime_r2, color=colors_regime, alpha=0.8, edgecolor='black')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.axhline(y=cv_r2_mean, color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_ylabel('R² Score', fontsize=10)
ax4.set_title('Performance by Volatility Regime', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for i, r2 in enumerate(regime_r2):
    ax4.text(i, r2 + 0.05 if r2 > 0 else r2 - 0.05, f'{r2:.3f}',
             ha='center', fontsize=9, fontweight='bold')

# ===== 모델 비교 (HAR 포함) =====
ax5 = fig.add_subplot(gs[2, 1])

# HAR 간단 계산
df_har = df.copy()
df_har['rv_daily'] = df_har['parkinson_vol'].shift(1)
df_har['rv_weekly'] = df_har['parkinson_vol'].shift(1).rolling(5).mean()
df_har['rv_monthly'] = df_har['parkinson_vol'].shift(1).rolling(22).mean()
df_har = df_har.dropna()

X_har = df_har[['rv_daily', 'rv_weekly', 'rv_monthly']]
y_har = df_har['target_vol_rv']

har_scores = []
for train_idx, test_idx in purged_kfold_cv(X_har, y_har, n_splits=5, purge_length=5, embargo_length=5):
    X_train_h, y_train_h = X_har.iloc[train_idx], y_har.iloc[train_idx]
    X_test_h, y_test_h = X_har.iloc[test_idx], y_har.iloc[test_idx]

    scaler_h = StandardScaler()
    X_train_h_scaled = scaler_h.fit_transform(X_train_h)
    X_test_h_scaled = scaler_h.transform(X_test_h)

    model_h = Ridge(alpha=1.0)
    model_h.fit(X_train_h_scaled, y_train_h)
    y_pred_h = model_h.predict(X_test_h_scaled)

    har_scores.append(r2_score(y_test_h, y_pred_h))

har_mean = np.mean(har_scores)

models = ['HAR\n(3 feat)', 'V0\n(26 feat)', 'RV\n(30 feat)']
r2_values = [har_mean, 0.3144, cv_r2_mean]  # V0는 이전 결과
colors_model = ['#e74c3c', '#3498db', '#2ecc71']

bars = ax5.bar(models, r2_values, color=colors_model, alpha=0.8, edgecolor='black')
ax5.set_ylabel('R² Score', fontsize=10)
ax5.set_title('Model Comparison', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, r2_values)):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}',
             ha='center', fontsize=10, fontweight='bold')

    if i > 0:
        improvement = (val / r2_values[0] - 1) * 100
        ax5.text(bar.get_x() + bar.get_width()/2, val/2, f'{improvement:+.0f}%',
                 ha='center', fontsize=9, color='white', fontweight='bold')

# ===== 성능 요약 테이블 =====
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

summary_data = [
    ['Metric', 'Value'],
    ['Model', 'Ridge (RV)'],
    ['CV R² Mean', f'{cv_r2_mean:.4f}'],
    ['CV R² Std', f'{cv_r2_std:.4f}'],
    ['Test R²', f'{overall_r2:.4f}'],
    ['RMSE', f'{rmse:.6f}'],
    ['MAE', f'{mae:.6f}'],
    ['Samples', f'{len(y_test_all)}'],
    ['Features', f'{len(feature_cols)}'],
    ['Target', 'Parkinson RV'],
    ['Validation', 'Purged K-Fold'],
    ['HAR Baseline', f'{har_mean:.4f}'],
    ['vs HAR', f'+{(cv_r2_mean/har_mean - 1)*100:.1f}%'],
]

table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                  colWidths=[0.55, 0.45])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

for i in range(2):
    table[(0, i)].set_facecolor('#2ecc71')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(summary_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

# 전체 제목
fig.suptitle('Realized Volatility Model - Final Performance Analysis',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

output_path = "dashboard/figures/rv_final_chart.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n💾 차트 저장: {output_path}")
plt.close()

# 성능 데이터 저장
print("\n3️⃣  성능 데이터 저장...")

performance_data = {
    "model_name": "Ridge Volatility Predictor (Realized Vol)",
    "model_type": "Ridge",
    "target": "Parkinson Realized Volatility (5-day mean)",
    "cv_r2_mean": float(cv_r2_mean),
    "cv_r2_std": float(cv_r2_std),
    "test_r2": float(overall_r2),
    "test_rmse": float(rmse),
    "test_mae": float(mae),
    "cv_scores": [float(s) for s in fold_scores],
    "validation_method": "Purged K-Fold CV (5-fold)",
    "n_samples": int(len(y_test_all)),
    "n_features": len(feature_cols),
    "har_benchmark": float(har_mean),
    "improvement_vs_har": float((cv_r2_mean / har_mean - 1) * 100),
    "improvement_vs_v0": float((cv_r2_mean / 0.3144 - 1) * 100),
    "timestamp": pd.Timestamp.now().isoformat()
}

with open('data/raw/rv_model_performance.json', 'w') as f:
    json.dump(performance_data, f, indent=2)

print(f"   성능 데이터: data/raw/rv_model_performance.json")

print("\n" + "="*70)
print("✅ 최종 차트 생성 완료")
print("="*70)
print(f"""
📊 Realized Volatility 모델 최종 성능

예측 성능:
  - CV R² Mean:  {cv_r2_mean:.4f} (±{cv_r2_std:.4f})
  - Test R²:     {overall_r2:.4f}
  - RMSE:        {rmse:.6f}
  - MAE:         {mae:.6f}

벤치마크 비교:
  - HAR:         {har_mean:.4f}
  - vs HAR:      +{(cv_r2_mean/har_mean - 1)*100:.1f}% ✅
  - vs V0:       +{(cv_r2_mean/0.3144 - 1)*100:.1f}% ✅

생성 파일:
  - 차트: dashboard/figures/rv_final_chart.png
  - 데이터: data/raw/rv_model_performance.json
""")
