#!/usr/bin/env python3
"""
변동성 예측 시각화 (간단 버전)
실제 SPY 변동성과 Ridge 모델 예측 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

def create_volatility_prediction_chart():
    """변동성 예측 차트 생성"""

    print("📊 변동성 예측 차트 생성 중...")

    # 1. SPY 데이터 수집
    spy = yf.Ticker("SPY")
    df = spy.history(start="2020-01-01", end="2024-12-31")
    df.index = pd.to_datetime(df.index).tz_localize(None)

    print(f"✅ SPY 데이터 로드: {len(df)} 샘플")

    # 2. 변동성 계산
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility_5d'] = df['returns'].rolling(5).std()
    df['volatility_20d'] = df['returns'].rolling(20).std()

    # 타겟: 5일 후 변동성
    df['target_vol'] = df['volatility_5d'].shift(-5)

    # 특성
    feature_cols = []
    for lag in [1, 2, 3, 5, 10]:
        df[f'vol_lag_{lag}'] = df['volatility_5d'].shift(lag)
        feature_cols.append(f'vol_lag_{lag}')

    df = df.dropna()

    # 3. 학습/테스트 분할
    X = df[feature_cols]
    y = df['target_vol']

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4. Ridge 모델 학습
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # 5. 예측
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # 성능
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"\n📈 모델 성능:")
    print(f"   학습 R²: {r2_train:.4f}")
    print(f"   테스트 R²: {r2_test:.4f}")
    print(f"   RMSE: {rmse_test:.6f}")
    print(f"   MAE: {mae_test:.6f}")

    # 6. 시각화
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 6.1 테스트 기간 시계열
    ax1 = axes[0]
    test_dates = y_test.index
    ax1.plot(test_dates, y_test.values, label='Actual Volatility', color='black', linewidth=2, alpha=0.8)
    ax1.plot(test_dates, y_pred_test, label='Predicted Volatility (Ridge)', color='red', linewidth=2, alpha=0.7)
    ax1.fill_between(test_dates, y_test.values, y_pred_test, alpha=0.2, color='gray')

    ax1.set_title(f'SPY 5-Day Volatility: Actual vs Predicted (Test R² = {r2_test:.4f})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Volatility', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 6.2 산점도
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred_test, alpha=0.5, s=20, color='blue')

    # 완벽한 예측 라인
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # 추세선
    z = np.polyfit(y_test, y_pred_test, 1)
    p = np.poly1d(z)
    ax2.plot(y_test, p(y_test), "g-", linewidth=2, label=f'Trend (R² = {r2_test:.4f})')

    ax2.set_xlabel('Actual Volatility', fontsize=12)
    ax2.set_ylabel('Predicted Volatility', fontsize=12)
    ax2.set_title('Actual vs Predicted Scatter Plot', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 통계 텍스트
    stats_text = f"""Ridge Model Performance:
Train R²: {r2_train:.4f}
Test R²: {r2_test:.4f}
RMSE: {rmse_test:.6f}
MAE: {mae_test:.6f}
Features: {len(feature_cols)}
Samples: {len(y_test):,}"""

    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
             verticalalignment='bottom')

    plt.tight_layout()

    # 저장
    output_path = "dashboard/figures/volatility_actual_vs_predicted.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 차트 저장: {output_path}")

    plt.close()

    return r2_test

if __name__ == "__main__":
    r2 = create_volatility_prediction_chart()

    print("\n✅ 변동성 예측 차트 생성 완료!")
    print(f"\n💡 핵심 결과:")
    print(f"   변동성 예측 R² = {r2:.4f}")
    print(f"   예측 가능 ✅")
    print(f"\n📁 생성 파일: dashboard/figures/volatility_actual_vs_predicted.png")
