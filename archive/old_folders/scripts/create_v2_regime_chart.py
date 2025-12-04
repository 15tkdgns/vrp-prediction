#!/usr/bin/env python3
"""
V2 Regime-Switching 모델 실제 예측 차트 생성
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_v2_regime_chart():
    """V2 Regime-Switching 모델로 실제 예측 및 차트 생성"""

    print("="*70)
    print("📊 V2 Regime-Switching 모델 예측 차트 생성")
    print("="*70)

    # 1. 데이터 로드
    print("\n1️⃣  데이터 로드...")
    spy = yf.Ticker("SPY")
    df = spy.history(start="2015-01-01", end="2024-12-31")
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['returns'].rolling(20).std()

    # 타겟: V0 방식 (returns[t+1:t+6].std())
    targets = []
    for i in range(len(df)):
        if i + 5 < len(df):
            future_returns = df['returns'].iloc[i+1:i+6]
            targets.append(future_returns.std())
        else:
            targets.append(np.nan)
    df['target_vol_5d'] = targets

    # 특성
    for lag in [1, 2, 3, 5, 10]:
        df[f'vol_lag_{lag}'] = df['volatility'].shift(lag)

    df = df.dropna()

    print(f"   데이터: {len(df)} 샘플")

    # 2. Regime-Switching 모델 학습
    print("\n2️⃣  Regime-Switching 모델 학습...")

    features = ['volatility', 'vol_lag_1', 'vol_lag_2', 'vol_lag_3', 'vol_lag_5', 'vol_lag_10']
    X = df[features]
    y = df['target_vol_5d']

    # K-means clustering
    vol_values = df['volatility'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['regime'] = kmeans.fit_predict(vol_values)

    # 고변동 = 1, 저변동 = 0
    cluster_means = df.groupby('regime')['volatility'].mean()
    if cluster_means[0] > cluster_means[1]:
        df['regime'] = 1 - df['regime']

    print(f"   고변동 regime: {(df['regime']==1).sum()} 샘플")
    print(f"   저변동 regime: {(df['regime']==0).sum()} 샘플")

    # 3. TimeSeriesSplit으로 예측값 생성
    print("\n3️⃣  TimeSeriesSplit CV로 예측값 생성...")

    tscv = TimeSeriesSplit(n_splits=5)
    all_predictions = np.zeros(len(X))
    all_actuals = np.zeros(len(X))
    test_indices = []

    fold_r2_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train_full = X.iloc[train_idx]
        y_train_full = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Regime 구분 (train data만 사용)
        regime_train = df['regime'].iloc[train_idx]

        # 각 regime별 모델 학습
        models = {}
        scalers = {}

        for regime in [0, 1]:
            regime_mask = (regime_train == regime)
            X_train_regime = X_train_full[regime_mask]
            y_train_regime = y_train_full[regime_mask]

            if len(X_train_regime) < 10:
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_regime)

            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train_regime)

            models[regime] = model
            scalers[regime] = scaler

        # 테스트 예측
        regime_test = df['regime'].iloc[test_idx]
        predictions = []

        for i, idx in enumerate(test_idx):
            regime = regime_test.iloc[i]

            if regime in models:
                X_test_single = X_test.iloc[[i]]
                X_test_scaled = scalers[regime].transform(X_test_single)
                pred = models[regime].predict(X_test_scaled)[0]
                predictions.append(pred)
            else:
                # Fallback: 평균 예측
                predictions.append(y_train_full.mean())

        # 저장
        all_predictions[test_idx] = predictions
        all_actuals[test_idx] = y_test.values
        test_indices.extend(test_idx)

        # Fold R²
        fold_r2 = r2_score(y_test, predictions)
        fold_r2_scores.append(fold_r2)
        print(f"   Fold {fold_idx}: R² = {fold_r2:.4f}")

    # 4. 전체 성능
    print("\n4️⃣  전체 성능 계산...")

    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[test_indices] = True

    y_test_all = all_actuals[test_mask]
    y_pred_all = all_predictions[test_mask]

    r2_total = r2_score(y_test_all, y_pred_all)
    rmse_total = np.sqrt(mean_squared_error(y_test_all, y_pred_all))
    mae_total = mean_absolute_error(y_test_all, y_pred_all)

    print(f"   Test R²: {r2_total:.4f}")
    print(f"   RMSE: {rmse_total:.6f}")
    print(f"   MAE: {mae_total:.6f}")
    print(f"   테스트 샘플: {len(y_test_all)}")

    # 5. 시각화
    print("\n5️⃣  차트 생성...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 최근 1년 데이터만
    recent_data = df[test_mask].iloc[-250:]
    recent_actual = y_test_all[-250:]
    recent_pred = y_pred_all[-250:]
    recent_dates = recent_data.index

    # 5.1 시계열 비교
    ax1 = axes[0]
    ax1.plot(recent_dates, recent_actual, label='Actual Volatility',
             color='black', linewidth=1.5, alpha=0.8)
    ax1.plot(recent_dates, recent_pred, label='Predicted Volatility (Regime-Switching)',
             color='red', linewidth=1.5, alpha=0.7)
    ax1.set_title(f'SPY 5-Day Volatility: Actual vs Predicted (Test R² = {r2_total:.4f})',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Volatility', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 5.2 산점도
    ax2 = axes[1]
    ax2.scatter(y_test_all, y_pred_all, alpha=0.5, s=10, color='blue')

    # 완벽한 예측선
    min_val = min(y_test_all.min(), y_pred_all.min())
    max_val = max(y_test_all.max(), y_pred_all.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--',
             linewidth=2, label='Perfect Prediction')

    # 추세선
    z = np.polyfit(y_test_all, y_pred_all, 1)
    p = np.poly1d(z)
    ax2.plot(y_test_all, p(y_test_all), "g-", linewidth=2,
             label=f'Trend (R² = {r2_total:.4f})')

    ax2.set_xlabel('Actual Volatility', fontsize=12)
    ax2.set_ylabel('Predicted Volatility', fontsize=12)
    ax2.set_title('Actual vs Predicted Scatter Plot', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 통계 정보
    stats_text = f"""Ridge Model Performance:
Train R²: {np.mean(fold_r2_scores):.4f}
Test R²: {r2_total:.4f}
RMSE: {rmse_total:.6f}
MAE: {mae_total:.6f}
Features: {len(features)}
Samples: {len(y_test_all)}"""

    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='bottom')

    plt.tight_layout()

    # 저장
    output_path = "dashboard/figures/volatility_actual_vs_predicted.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 차트 저장: {output_path}")

    plt.close()

    # 6. 결과 요약
    print("\n" + "="*70)
    print("📋 V2 Regime-Switching 모델 최종 성능")
    print("="*70)
    print(f"Test R² Score: {r2_total:.4f}")
    print(f"Train R² (CV Mean): {np.mean(fold_r2_scores):.4f} (±{np.std(fold_r2_scores):.4f})")
    print(f"RMSE: {rmse_total:.6f}")
    print(f"MAE: {mae_total:.6f}")
    print(f"Regime 0 (Low Vol): {(df['regime']==0).sum()} samples")
    print(f"Regime 1 (High Vol): {(df['regime']==1).sum()} samples")
    print("="*70)

    return r2_total, rmse_total, mae_total

if __name__ == "__main__":
    r2, rmse, mae = create_v2_regime_chart()
    print("\n✅ V2 Regime-Switching 차트 생성 완료!")
