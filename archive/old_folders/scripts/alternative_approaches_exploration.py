#!/usr/bin/env python3
"""
대안적 접근법 탐색
기존 방법과 완전히 다른 관점에서 시도

1. Quantile Regression - 분위수 예측
2. Random Forest - 비선형 앙상블
3. Target 재설계 - Realized Volatility
4. Feature Selection - 최소 특성으로 최대 성능
5. Ensemble Stacking - 여러 모델 조합
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
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
print("🔬 대안적 접근법 탐색")
print("="*70)

# 데이터 로드
print("\n1️⃣  데이터 로드...")
spy = yf.Ticker("SPY")
df = spy.history(start="2015-01-01", end="2024-12-31")
df.index = pd.to_datetime(df.index).tz_localize(None)
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

# 기본 타겟 (V0 방식)
targets = []
for i in range(len(df)):
    if i + 5 < len(df):
        future_returns = df['returns'].iloc[i+1:i+6]
        targets.append(future_returns.std())
    else:
        targets.append(np.nan)
df['target_vol_5d'] = targets

# 기본 특성 생성
print("\n2️⃣  기본 특성 생성...")

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

X = df[feature_cols]
y = df['target_vol_5d']

print(f"   데이터: {len(df)} 샘플")
print(f"   특성: {len(feature_cols)}개")

# Baseline
print("\n" + "="*70)
print("📊 Baseline (Ridge)")
print("="*70)

baseline_scores = []
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

    r2 = r2_score(y_test, y_pred)
    baseline_scores.append(r2)
    print(f"Fold {fold_idx}: R² = {r2:.4f}")

baseline_mean = np.mean(baseline_scores)
print(f"\nBaseline Mean R²: {baseline_mean:.4f} (±{np.std(baseline_scores):.4f})")

# ==================== 대안 1: Quantile Regression ====================
print("\n" + "="*70)
print("🎯 대안 1: Quantile Regression (분위수 예측)")
print("="*70)
print("   전략: 점 예측 대신 분위수 예측 (10%, 50%, 90%)")

quantile_scores = []
for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Median 예측 (quantile=0.5)
    model = QuantileRegressor(quantile=0.5, alpha=1.0, solver='highs')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    quantile_scores.append(r2)
    print(f"Fold {fold_idx}: R² = {r2:.4f}")

quantile_mean = np.mean(quantile_scores)
print(f"\nQuantile Mean R²: {quantile_mean:.4f} (±{np.std(quantile_scores):.4f})")
print(f"vs Baseline: {quantile_mean - baseline_mean:+.4f} ({(quantile_mean/baseline_mean - 1)*100:+.1f}%)")

# ==================== 대안 2: Random Forest ====================
print("\n" + "="*70)
print("🌲 대안 2: Random Forest (비선형 앙상블)")
print("="*70)
print("   전략: 트리 기반 모델로 비선형 관계 포착")

rf_scores = []
for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Random Forest (스케일링 불필요)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rf_scores.append(r2)
    print(f"Fold {fold_idx}: R² = {r2:.4f}")

rf_mean = np.mean(rf_scores)
print(f"\nRandom Forest Mean R²: {rf_mean:.4f} (±{np.std(rf_scores):.4f})")
print(f"vs Baseline: {rf_mean - baseline_mean:+.4f} ({(rf_mean/baseline_mean - 1)*100:+.1f}%)")

# ==================== 대안 3: Gradient Boosting ====================
print("\n" + "="*70)
print("🚀 대안 3: Gradient Boosting")
print("="*70)
print("   전략: Boosting으로 잔차 학습")

gb_scores = []
for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    gb_scores.append(r2)
    print(f"Fold {fold_idx}: R² = {r2:.4f}")

gb_mean = np.mean(gb_scores)
print(f"\nGradient Boosting Mean R²: {gb_mean:.4f} (±{np.std(gb_scores):.4f})")
print(f"vs Baseline: {gb_mean - baseline_mean:+.4f} ({(gb_mean/baseline_mean - 1)*100:+.1f}%)")

# ==================== 대안 4: Feature Selection ====================
print("\n" + "="*70)
print("🎯 대안 4: Feature Selection (최소 특성)")
print("="*70)
print("   전략: 가장 중요한 10개 특성만 사용")

# Feature importance 분석
X_train_full = X.iloc[:int(len(X)*0.8)]
y_train_full = y.iloc[:int(len(y)*0.8)]

selector = SelectKBest(score_func=mutual_info_regression, k=10)
selector.fit(X_train_full, y_train_full)

selected_features = X.columns[selector.get_support()].tolist()
print(f"\n   선택된 10개 특성:")
for i, feat in enumerate(selected_features, 1):
    print(f"   {i:2d}. {feat}")

X_selected = X[selected_features]

fs_scores = []
for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X_selected, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train, y_train = X_selected.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X_selected.iloc[test_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    fs_scores.append(r2)
    print(f"Fold {fold_idx}: R² = {r2:.4f}")

fs_mean = np.mean(fs_scores)
print(f"\nFeature Selection Mean R²: {fs_mean:.4f} (±{np.std(fs_scores):.4f})")
print(f"vs Baseline (26 features): {fs_mean - baseline_mean:+.4f} ({(fs_mean/baseline_mean - 1)*100:+.1f}%)")

# ==================== 대안 5: Simple Ensemble ====================
print("\n" + "="*70)
print("🎭 대안 5: Simple Ensemble (Ridge + RF)")
print("="*70)
print("   전략: Ridge와 Random Forest 평균")

ensemble_scores = []
for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=50,
        min_samples_leaf=20, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # 평균
    y_pred = (ridge_pred + rf_pred) / 2

    r2 = r2_score(y_test, y_pred)
    ensemble_scores.append(r2)
    print(f"Fold {fold_idx}: R² = {r2:.4f}")

ensemble_mean = np.mean(ensemble_scores)
print(f"\nEnsemble Mean R²: {ensemble_mean:.4f} (±{np.std(ensemble_scores):.4f})")
print(f"vs Baseline: {ensemble_mean - baseline_mean:+.4f} ({(ensemble_mean/baseline_mean - 1)*100:+.1f}%)")

# ==================== 대안 6: Realized Volatility Target ====================
print("\n" + "="*70)
print("📐 대안 6: Realized Volatility (고빈도 계산)")
print("="*70)
print("   전략: 일중 High-Low로 변동성 추정")

# Parkinson's volatility (High-Low 기반)
df['parkinson_vol'] = np.sqrt(
    1/(4*np.log(2)) * (np.log(df['High']/df['Low']))**2
)

# 5일 평균 Parkinson volatility를 타겟으로
targets_rv = []
for i in range(len(df)):
    if i + 5 < len(df):
        future_vol = df['parkinson_vol'].iloc[i+1:i+6].mean()
        targets_rv.append(future_vol)
    else:
        targets_rv.append(np.nan)

df['target_rv_5d'] = targets_rv
df_rv = df.dropna()

X_rv = df_rv[feature_cols]
y_rv = df_rv['target_rv_5d']

print(f"   Realized Vol 데이터: {len(df_rv)} 샘플")

rv_scores = []
for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X_rv, y_rv, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train, y_train = X_rv.iloc[train_idx], y_rv.iloc[train_idx]
    X_test, y_test = X_rv.iloc[test_idx], y_rv.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rv_scores.append(r2)
    print(f"Fold {fold_idx}: R² = {r2:.4f}")

rv_mean = np.mean(rv_scores)
print(f"\nRealized Vol Mean R²: {rv_mean:.4f} (±{np.std(rv_scores):.4f})")
print(f"vs Baseline: {rv_mean - baseline_mean:+.4f} ({(rv_mean/baseline_mean - 1)*100:+.1f}%)")

# ==================== 최종 결과 ====================
print("\n" + "="*70)
print("📊 모든 대안 비교")
print("="*70)

results = [
    ("Baseline (Ridge)", baseline_mean, np.std(baseline_scores)),
    ("Quantile Regression", quantile_mean, np.std(quantile_scores)),
    ("Random Forest", rf_mean, np.std(rf_scores)),
    ("Gradient Boosting", gb_mean, np.std(gb_scores)),
    ("Feature Selection (10개)", fs_mean, np.std(fs_scores)),
    ("Ensemble (Ridge+RF)", ensemble_mean, np.std(ensemble_scores)),
    ("Realized Volatility", rv_mean, np.std(rv_scores)),
]

results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

print(f"\n{'방법':<30s} {'R² Mean':>10s} {'Std':>10s} {'vs Baseline':>15s}")
print("-" * 70)
for name, mean, std in results_sorted:
    delta = mean - baseline_mean
    pct = (mean / baseline_mean - 1) * 100 if baseline_mean != 0 else 0
    symbol = "🏆" if mean == max(r[1] for r in results) else "  "
    print(f"{symbol} {name:<28s} {mean:10.4f} {std:10.4f} {delta:+7.4f} ({pct:+6.1f}%)")

# 승자 확인
best = results_sorted[0]
print("\n" + "="*70)
print("🏆 최고 성능 모델")
print("="*70)
print(f"\n모델: {best[0]}")
print(f"R² Mean: {best[1]:.4f} (±{best[2]:.4f})")
print(f"개선: {best[1] - baseline_mean:+.4f} ({(best[1]/baseline_mean - 1)*100:+.1f}%)")

if best[1] > baseline_mean * 1.05:  # 5% 이상 개선
    print("\n✅ 의미 있는 개선 발견!")
    print(f"   {best[0]} 모델을 최종 모델로 고려 가능")
elif best[1] > baseline_mean:
    print("\n⚠️ 미미한 개선")
    print(f"   개선 폭이 작아 Baseline 유지 권장")
else:
    print("\n❌ 개선 없음")
    print("   Baseline (Ridge) 유지 권장")

# 결과 저장
import json
results_dict = {
    "experiment": "Alternative Approaches",
    "date": pd.Timestamp.now().isoformat(),
    "baseline": {
        "model": "Ridge",
        "r2_mean": float(baseline_mean),
        "r2_std": float(np.std(baseline_scores)),
        "scores": [float(s) for s in baseline_scores]
    },
    "alternatives": {
        "quantile_regression": {
            "r2_mean": float(quantile_mean),
            "r2_std": float(np.std(quantile_scores)),
            "improvement_pct": float((quantile_mean/baseline_mean - 1)*100)
        },
        "random_forest": {
            "r2_mean": float(rf_mean),
            "r2_std": float(np.std(rf_scores)),
            "improvement_pct": float((rf_mean/baseline_mean - 1)*100)
        },
        "gradient_boosting": {
            "r2_mean": float(gb_mean),
            "r2_std": float(np.std(gb_scores)),
            "improvement_pct": float((gb_mean/baseline_mean - 1)*100)
        },
        "feature_selection": {
            "r2_mean": float(fs_mean),
            "r2_std": float(np.std(fs_scores)),
            "features": selected_features,
            "improvement_pct": float((fs_mean/baseline_mean - 1)*100)
        },
        "ensemble": {
            "r2_mean": float(ensemble_mean),
            "r2_std": float(np.std(ensemble_scores)),
            "improvement_pct": float((ensemble_mean/baseline_mean - 1)*100)
        },
        "realized_volatility": {
            "r2_mean": float(rv_mean),
            "r2_std": float(np.std(rv_scores)),
            "improvement_pct": float((rv_mean/baseline_mean - 1)*100)
        }
    },
    "best_model": {
        "name": best[0],
        "r2_mean": float(best[1]),
        "r2_std": float(best[2]),
        "improvement": float(best[1] - baseline_mean),
        "improvement_pct": float((best[1]/baseline_mean - 1)*100)
    }
}

with open('data/raw/alternative_approaches_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\n💾 결과 저장: data/raw/alternative_approaches_results.json")
