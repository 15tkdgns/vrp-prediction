#!/usr/bin/env python3
"""
V0 Ridge 모델 종합 성능 시각화
- Fold별 성능 분포
- 시간대별 예측 정확도
- 에러 분석
- 특성 중요도
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

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

def create_comprehensive_visualization():
    """종합 성능 시각화 생성"""

    print("="*70)
    print("📊 V0 Ridge 모델 종합 성능 시각화")
    print("="*70)

    # 1. 데이터 로드
    print("\n1️⃣  데이터 로드...")
    spy = yf.Ticker("SPY")
    df = spy.history(start="2015-01-01", end="2024-12-31")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # 타겟 생성
    print("\n2️⃣  타겟 생성...")
    targets = []
    for i in range(len(df)):
        if i + 5 < len(df):
            future_returns = df['returns'].iloc[i+1:i+6]
            targets.append(future_returns.std())
        else:
            targets.append(np.nan)
    df['target_vol_5d'] = targets

    # 특성 생성
    print("\n3️⃣  특성 생성...")

    # 변동성
    for window in [5, 10, 20, 60]:
        df[f'volatility_{window}d'] = df['returns'].rolling(window).std()

    # Lag
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'vol_lag_{lag}'] = df['volatility_20d'].shift(lag)

    # 롤링 통계
    df['vol_mean_5d'] = df['volatility_20d'].rolling(5).mean()
    df['vol_mean_10d'] = df['volatility_20d'].rolling(10).mean()
    df['vol_std_5d'] = df['volatility_20d'].rolling(5).std()
    df['vol_std_10d'] = df['volatility_20d'].rolling(10).std()

    # 모멘텀
    for window in [5, 10, 20]:
        df[f'momentum_{window}d'] = df['returns'].rolling(window).sum()

    # 수익률 통계
    df['returns_mean_5d'] = df['returns'].rolling(5).mean()
    df['returns_mean_10d'] = df['returns'].rolling(10).mean()
    df['returns_std_5d'] = df['returns'].rolling(5).std()
    df['returns_std_10d'] = df['returns'].rolling(10).std()

    # 변동성 변화율
    df['vol_change_5d'] = df['volatility_20d'].pct_change(5)
    df['vol_change_10d'] = df['volatility_20d'].pct_change(10)

    # 극단값
    df['extreme_returns'] = (df['returns'].abs() > 2 * df['volatility_20d']).astype(int)
    df['extreme_count_20d'] = df['extreme_returns'].rolling(20).sum()

    df = df.dropna()

    feature_cols = [col for col in df.columns if col not in
                    ['returns', 'target_vol_5d', 'Close', 'Open', 'High', 'Low',
                     'Volume', 'Dividends', 'Stock Splits']]
    feature_cols = feature_cols[:31]

    X = df[feature_cols]
    y = df['target_vol_5d']

    print(f"   데이터: {len(df)} 샘플")
    print(f"   특성: {len(feature_cols)}개")

    # 4. CV로 예측값 및 메트릭 수집
    print("\n4️⃣  Cross-Validation 수행...")

    all_predictions = np.full(len(X), np.nan)
    all_actuals = np.full(len(X), np.nan)
    fold_metrics = []
    feature_importances = []

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

        # 메트릭 저장
        fold_r2 = r2_score(y_test, y_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        fold_mae = mean_absolute_error(y_test, y_pred)

        fold_metrics.append({
            'fold': fold_idx,
            'r2': fold_r2,
            'rmse': fold_rmse,
            'mae': fold_mae,
            'samples': len(test_idx)
        })

        # 특성 중요도 (계수 절댓값)
        feature_importances.append(np.abs(model.coef_))

        print(f"   Fold {fold_idx}: R² = {fold_r2:.4f}, RMSE = {fold_rmse:.6f}, MAE = {fold_mae:.6f}")

    test_mask = ~np.isnan(all_predictions)
    y_test_all = all_actuals[test_mask]
    y_pred_all = all_predictions[test_mask]
    df_test = df.iloc[test_mask].copy()
    df_test['predictions'] = y_pred_all
    df_test['errors'] = y_test_all - y_pred_all
    df_test['abs_errors'] = np.abs(df_test['errors'])
    df_test['pct_errors'] = (df_test['errors'] / y_test_all) * 100

    # 5. 종합 시각화
    print("\n5️⃣  시각화 생성...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ===== 1. Fold별 성능 비교 (Bar Chart) =====
    ax1 = fig.add_subplot(gs[0, 0])
    fold_df = pd.DataFrame(fold_metrics)
    x_pos = np.arange(len(fold_df))
    bars = ax1.bar(x_pos, fold_df['r2'], color=['#2ecc71' if r2 > 0 else '#e74c3c' for r2 in fold_df['r2']])
    ax1.axhline(y=fold_df['r2'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {fold_df["r2"].mean():.4f}')
    ax1.set_xlabel('Fold', fontsize=10)
    ax1.set_ylabel('R² Score', fontsize=10)
    ax1.set_title('Fold-wise R² Performance', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Fold {i}' for i in fold_df['fold']])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 값 표시
    for i, (idx, row) in enumerate(fold_df.iterrows()):
        ax1.text(i, row['r2'] + 0.02, f"{row['r2']:.3f}", ha='center', fontsize=9)

    # ===== 2. 에러 분포 (Histogram) =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_test['errors'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(x=df_test['errors'].mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {df_test["errors"].mean():.6f}')
    ax2.set_xlabel('Prediction Error', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ===== 3. 시간대별 R² (Rolling Window) =====
    ax3 = fig.add_subplot(gs[0, 2])
    window = 100
    rolling_r2 = []
    rolling_dates = []

    for i in range(window, len(df_test)):
        subset_actual = y_test_all[i-window:i]
        subset_pred = y_pred_all[i-window:i]
        r2 = r2_score(subset_actual, subset_pred)
        rolling_r2.append(r2)
        rolling_dates.append(df_test.index[i])

    ax3.plot(rolling_dates, rolling_r2, color='#9b59b6', linewidth=1.5)
    ax3.axhline(y=fold_df['r2'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Overall CV R²: {fold_df["r2"].mean():.4f}')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('R² Score', fontsize=10)
    ax3.set_title(f'Rolling R² (Window={window})', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # ===== 4. 특성 중요도 (Top 15) =====
    ax4 = fig.add_subplot(gs[1, 0])
    avg_importance = np.mean(feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols[:len(avg_importance)],
        'importance': avg_importance
    }).sort_values('importance', ascending=False).head(15)

    y_pos = np.arange(len(feature_importance_df))
    ax4.barh(y_pos, feature_importance_df['importance'], color='#e67e22')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(feature_importance_df['feature'], fontsize=8)
    ax4.invert_yaxis()
    ax4.set_xlabel('Absolute Coefficient', fontsize=10)
    ax4.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # ===== 5. 실제 vs 예측 분위수 분석 =====
    ax5 = fig.add_subplot(gs[1, 1])
    quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    actual_quantiles = np.quantile(y_test_all, quantiles)
    pred_quantiles = np.quantile(y_pred_all, quantiles)

    x_labels = ['Min', 'Q1', 'Median', 'Q3', 'Max']
    x_pos = np.arange(len(x_labels))
    width = 0.35

    ax5.bar(x_pos - width/2, actual_quantiles, width, label='Actual', color='#3498db', alpha=0.8)
    ax5.bar(x_pos + width/2, pred_quantiles, width, label='Predicted', color='#e74c3c', alpha=0.8)
    ax5.set_xlabel('Quantile', fontsize=10)
    ax5.set_ylabel('Volatility', fontsize=10)
    ax5.set_title('Quantile Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(x_labels)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # ===== 6. 에러 vs 실제 변동성 =====
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(y_test_all, df_test['abs_errors'], alpha=0.3, s=5, color='#e74c3c')
    ax6.set_xlabel('Actual Volatility', fontsize=10)
    ax6.set_ylabel('Absolute Error', fontsize=10)
    ax6.set_title('Error vs Actual Volatility', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 추세선
    z = np.polyfit(y_test_all, df_test['abs_errors'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(y_test_all.min(), y_test_all.max(), 100)
    ax6.plot(x_trend, p(x_trend), 'g--', linewidth=2, label='Trend')
    ax6.legend(fontsize=9)

    # ===== 7. 시계열 에러 패턴 =====
    ax7 = fig.add_subplot(gs[2, :2])
    recent_data = df_test.iloc[-250:]

    ax7.plot(recent_data.index, recent_data['errors'], color='#e74c3c', linewidth=1, alpha=0.7, label='Prediction Error')
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax7.fill_between(recent_data.index, 0, recent_data['errors'],
                      where=(recent_data['errors'] > 0), color='#2ecc71', alpha=0.3, label='Overestimate')
    ax7.fill_between(recent_data.index, 0, recent_data['errors'],
                      where=(recent_data['errors'] < 0), color='#e74c3c', alpha=0.3, label='Underestimate')
    ax7.set_xlabel('Date', fontsize=10)
    ax7.set_ylabel('Prediction Error', fontsize=10)
    ax7.set_title('Time Series Error Pattern (Last 250 days)', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(axis='x', rotation=45)

    # ===== 8. 성능 요약 테이블 =====
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    summary_stats = [
        ['Metric', 'Value'],
        ['CV R² Mean', f"{fold_df['r2'].mean():.4f}"],
        ['CV R² Std', f"{fold_df['r2'].std():.4f}"],
        ['Test R² Total', f"{r2_score(y_test_all, y_pred_all):.4f}"],
        ['RMSE', f"{np.sqrt(mean_squared_error(y_test_all, y_pred_all)):.6f}"],
        ['MAE', f"{mean_absolute_error(y_test_all, y_pred_all):.6f}"],
        ['Mean Error', f"{df_test['errors'].mean():.6f}"],
        ['Error Std', f"{df_test['errors'].std():.6f}"],
        ['Samples', f"{len(y_test_all)}"],
        ['Features', f"{len(feature_cols)}"],
    ]

    table = ax8.table(cellText=summary_stats, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 헤더 스타일
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 교대 행 색상
    for i in range(1, len(summary_stats)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    ax8.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    # 전체 제목
    fig.suptitle('V0 Ridge Model - Comprehensive Performance Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # 저장
    output_path = "dashboard/figures/comprehensive_performance_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 종합 시각화 저장: {output_path}")

    plt.close()

    # 6. 추가 분석: 변동성 구간별 성능
    print("\n6️⃣  변동성 구간별 성능 분석...")

    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 변동성 3분위
    df_test['vol_tercile'] = pd.qcut(y_test_all, q=3, labels=['Low', 'Medium', 'High'])

    # 6.1 구간별 R²
    ax1 = axes[0, 0]
    tercile_r2 = []
    for tercile in ['Low', 'Medium', 'High']:
        mask = df_test['vol_tercile'] == tercile
        if mask.sum() > 0:
            tercile_r2.append(r2_score(y_test_all[mask], y_pred_all[mask]))
        else:
            tercile_r2.append(0)

    bars = ax1.bar(['Low Vol', 'Medium Vol', 'High Vol'], tercile_r2,
                    color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
    ax1.axhline(y=fold_df['r2'].mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Overall: {fold_df["r2"].mean():.4f}')
    ax1.set_ylabel('R² Score', fontsize=10)
    ax1.set_title('Performance by Volatility Regime', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(tercile_r2):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    # 6.2 구간별 샘플 수
    ax2 = axes[0, 1]
    tercile_counts = df_test['vol_tercile'].value_counts().sort_index()
    ax2.pie(tercile_counts, labels=['Low Vol', 'Medium Vol', 'High Vol'],
            autopct='%1.1f%%', colors=['#2ecc71', '#f39c12', '#e74c3c'], startangle=90)
    ax2.set_title('Sample Distribution by Regime', fontsize=12, fontweight='bold')

    # 6.3 구간별 박스플롯
    ax3 = axes[1, 0]
    df_test['vol_tercile_str'] = df_test['vol_tercile'].astype(str)
    df_test.boxplot(column='abs_errors', by='vol_tercile_str', ax=ax3,
                     patch_artist=True, showfliers=False)
    ax3.set_xlabel('Volatility Regime', fontsize=10)
    ax3.set_ylabel('Absolute Error', fontsize=10)
    ax3.set_title('Error Distribution by Regime', fontsize=12, fontweight='bold')
    ax3.get_figure().suptitle('')

    # 6.4 월별 평균 성능
    ax4 = axes[1, 1]
    df_test['year_month'] = df_test.index.to_period('M')
    monthly_r2 = []
    monthly_labels = []

    for ym in df_test['year_month'].unique()[-12:]:
        mask = df_test['year_month'] == ym
        if mask.sum() > 10:
            monthly_r2.append(r2_score(y_test_all[mask], y_pred_all[mask]))
            monthly_labels.append(str(ym))

    ax4.plot(range(len(monthly_r2)), monthly_r2, marker='o', linewidth=2, color='#9b59b6')
    ax4.axhline(y=fold_df['r2'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Overall: {fold_df["r2"].mean():.4f}')
    ax4.set_xlabel('Month', fontsize=10)
    ax4.set_ylabel('R² Score', fontsize=10)
    ax4.set_title('Monthly Performance (Last 12 Months)', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(monthly_labels)))
    ax4.set_xticklabels(monthly_labels, rotation=45, fontsize=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path2 = "dashboard/figures/regime_based_performance_analysis.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"💾 변동성 구간별 분석 저장: {output_path2}")

    plt.close()

    print("\n" + "="*70)
    print("✅ 종합 성능 시각화 완료")
    print("="*70)
    print(f"\n생성된 파일:")
    print(f"  1. {output_path}")
    print(f"  2. {output_path2}")
    print(f"\n주요 인사이트:")
    print(f"  - CV R² Mean: {fold_df['r2'].mean():.4f} (±{fold_df['r2'].std():.4f})")
    print(f"  - Low Vol R²: {tercile_r2[0]:.4f}")
    print(f"  - Medium Vol R²: {tercile_r2[1]:.4f}")
    print(f"  - High Vol R²: {tercile_r2[2]:.4f}")
    print(f"  - Mean Error: {df_test['errors'].mean():.6f}")
    print(f"  - Error Std: {df_test['errors'].std():.6f}")

if __name__ == "__main__":
    create_comprehensive_visualization()
