#!/usr/bin/env python3
"""
Realized Volatility 경제적 백테스트
- 변동성 예측 기반 포지션 사이징
- Transaction cost 포함
- Sharpe Ratio, Drawdown 분석
- V0 모델과 비교
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
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
print("💰 Realized Volatility 경제적 백테스트")
print("="*70)

# 1. 데이터 로드
print("\n1️⃣  데이터 로드...")
spy = yf.Ticker("SPY")
df = spy.history(start="2015-01-01", end="2024-12-31")
df.index = pd.to_datetime(df.index).tz_localize(None)
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

# V0 타겟
targets_v0 = []
for i in range(len(df)):
    if i + 5 < len(df):
        future_returns = df['returns'].iloc[i+1:i+6]
        targets_v0.append(future_returns.std())
    else:
        targets_v0.append(np.nan)
df['target_vol_v0'] = targets_v0

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
print("\n2️⃣  특성 생성...")

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

# RV 특성
df['rv_lag_1'] = df['parkinson_vol'].shift(1)
df['rv_lag_5'] = df['parkinson_vol'].shift(5)
df['rv_ma_5'] = df['parkinson_vol'].rolling(5).mean()
df['rv_ma_20'] = df['parkinson_vol'].rolling(20).mean()

df = df.dropna()

feature_cols = [col for col in df.columns if col not in
                ['returns', 'target_vol_v0', 'target_vol_rv', 'Close', 'Open', 'High', 'Low',
                 'Volume', 'Dividends', 'Stock Splits', 'parkinson_vol']]

X = df[feature_cols]

print(f"   데이터: {len(df)} 샘플")
print(f"   특성: {len(feature_cols)}개")

# 3. V0 모델로 예측값 생성
print("\n" + "="*70)
print("📊 V0 모델 예측")
print("="*70)

y_v0 = df['target_vol_v0']
v0_predictions = np.full(len(X), np.nan)
v0_actuals = np.full(len(X), np.nan)

for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y_v0, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train, y_train = X.iloc[train_idx], y_v0.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y_v0.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    v0_predictions[test_idx] = y_pred
    v0_actuals[test_idx] = y_test.values

    print(f"Fold {fold_idx}: R² = {r2_score(y_test, y_pred):.4f}")

v0_mask = ~np.isnan(v0_predictions)
print(f"V0 Overall R²: {r2_score(v0_actuals[v0_mask], v0_predictions[v0_mask]):.4f}")

# 4. RV 모델로 예측값 생성
print("\n" + "="*70)
print("📊 RV 모델 예측")
print("="*70)

y_rv = df['target_vol_rv']
rv_predictions = np.full(len(X), np.nan)
rv_actuals = np.full(len(X), np.nan)

for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_cv(X, y_rv, n_splits=5, purge_length=5, embargo_length=5), 1):

    X_train, y_train = X.iloc[train_idx], y_rv.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y_rv.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    rv_predictions[test_idx] = y_pred
    rv_actuals[test_idx] = y_test.values

    print(f"Fold {fold_idx}: R² = {r2_score(y_test, y_pred):.4f}")

rv_mask = ~np.isnan(rv_predictions)
print(f"RV Overall R²: {r2_score(rv_actuals[rv_mask], rv_predictions[rv_mask]):.4f}")

# 5. 백테스트 데이터 준비
print("\n" + "="*70)
print("💼 백테스트 전략 설계")
print("="*70)

df_bt = df.copy()
df_bt['v0_pred'] = v0_predictions
df_bt['rv_pred'] = rv_predictions
df_bt = df_bt.dropna(subset=['v0_pred', 'rv_pred'])

print(f"\n백테스트 데이터: {len(df_bt)} 샘플")
print(f"기간: {df_bt.index[0].date()} ~ {df_bt.index[-1].date()}")

# 전략: 예측 변동성의 역수로 포지션 사이징
# 변동성이 낮을 것으로 예측되면 큰 포지션, 높으면 작은 포지션

# 6. 전략 1: Buy & Hold (벤치마크)
print("\n전략 1: Buy & Hold (벤치마크)")
df_bt['bh_equity'] = (1 + df_bt['returns']).cumprod()
df_bt['bh_returns'] = df_bt['returns']

# 7. 전략 2: V0 변동성 타겟팅
print("전략 2: V0 Volatility Targeting")

# 타겟 변동성 (연율화 15%)
target_vol = 0.15 / np.sqrt(252)

# 포지션 사이징: target_vol / predicted_vol
df_bt['v0_position'] = target_vol / df_bt['v0_pred']
df_bt['v0_position'] = df_bt['v0_position'].clip(0.5, 2.0)  # 50% ~ 200%

# Transaction cost (0.1%)
tc = 0.001
df_bt['v0_position_change'] = df_bt['v0_position'].diff().abs()
df_bt['v0_tc'] = df_bt['v0_position_change'] * tc

# 전략 수익률
df_bt['v0_strategy_returns'] = df_bt['v0_position'].shift(1) * df_bt['returns'] - df_bt['v0_tc']
df_bt['v0_equity'] = (1 + df_bt['v0_strategy_returns']).cumprod()

# 8. 전략 3: RV 변동성 타겟팅
print("전략 3: RV Volatility Targeting")

df_bt['rv_position'] = target_vol / df_bt['rv_pred']
df_bt['rv_position'] = df_bt['rv_position'].clip(0.5, 2.0)

df_bt['rv_position_change'] = df_bt['rv_position'].diff().abs()
df_bt['rv_tc'] = df_bt['rv_position_change'] * tc

df_bt['rv_strategy_returns'] = df_bt['rv_position'].shift(1) * df_bt['returns'] - df_bt['rv_tc']
df_bt['rv_equity'] = (1 + df_bt['rv_strategy_returns']).cumprod()

df_bt = df_bt.dropna()

# 9. 성과 지표 계산
print("\n" + "="*70)
print("📈 백테스트 성과 분석")
print("="*70)

def calculate_metrics(returns, name):
    """백테스트 성과 지표 계산"""
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1

    volatility = returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0  # Risk-free = 2%

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).sum() / len(returns)

    print(f"\n{name}:")
    print(f"  총 수익률:       {total_return*100:7.2f}%")
    print(f"  연간 수익률:     {annual_return*100:7.2f}%")
    print(f"  연간 변동성:     {volatility*100:7.2f}%")
    print(f"  Sharpe Ratio:   {sharpe:7.3f}")
    print(f"  최대 낙폭:       {max_drawdown*100:7.2f}%")
    print(f"  승률:           {win_rate*100:7.2f}%")

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

bh_metrics = calculate_metrics(df_bt['bh_returns'], "Buy & Hold")
v0_metrics = calculate_metrics(df_bt['v0_strategy_returns'], "V0 Strategy")
rv_metrics = calculate_metrics(df_bt['rv_strategy_returns'], "RV Strategy")

# 10. 비교 분석
print("\n" + "="*70)
print("⚖️  전략 비교")
print("="*70)

print(f"\n{'지표':<20s} {'Buy&Hold':>12s} {'V0':>12s} {'RV':>12s} {'RV vs BH':>12s}")
print("-" * 70)
print(f"{'연간 수익률':<20s} {bh_metrics['annual_return']*100:11.2f}% {v0_metrics['annual_return']*100:11.2f}% {rv_metrics['annual_return']*100:11.2f}% {(rv_metrics['annual_return']-bh_metrics['annual_return'])*100:+11.2f}%")
print(f"{'연간 변동성':<20s} {bh_metrics['volatility']*100:11.2f}% {v0_metrics['volatility']*100:11.2f}% {rv_metrics['volatility']*100:11.2f}% {(rv_metrics['volatility']-bh_metrics['volatility'])*100:+11.2f}%")
print(f"{'Sharpe Ratio':<20s} {bh_metrics['sharpe']:11.3f}  {v0_metrics['sharpe']:11.3f}  {rv_metrics['sharpe']:11.3f}  {rv_metrics['sharpe']-bh_metrics['sharpe']:+11.3f}")
print(f"{'최대 낙폭':<20s} {bh_metrics['max_drawdown']*100:11.2f}% {v0_metrics['max_drawdown']*100:11.2f}% {rv_metrics['max_drawdown']*100:11.2f}% {(rv_metrics['max_drawdown']-bh_metrics['max_drawdown'])*100:+11.2f}%")

# 11. 시각화
print("\n" + "="*70)
print("📊 시각화 생성")
print("="*70)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# 11.1 누적 수익률
ax1 = axes[0, 0]
ax1.plot(df_bt.index, df_bt['bh_equity'], label='Buy & Hold', linewidth=2, color='#3498db')
ax1.plot(df_bt.index, df_bt['v0_equity'], label='V0 Strategy', linewidth=2, color='#e74c3c')
ax1.plot(df_bt.index, df_bt['rv_equity'], label='RV Strategy', linewidth=2, color='#2ecc71')
ax1.set_ylabel('Equity', fontsize=10)
ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 11.2 Drawdown
ax2 = axes[0, 1]
bh_cumulative = (1 + df_bt['bh_returns']).cumprod()
v0_cumulative = (1 + df_bt['v0_strategy_returns']).cumprod()
rv_cumulative = (1 + df_bt['rv_strategy_returns']).cumprod()

bh_dd = (bh_cumulative - bh_cumulative.expanding().max()) / bh_cumulative.expanding().max() * 100
v0_dd = (v0_cumulative - v0_cumulative.expanding().max()) / v0_cumulative.expanding().max() * 100
rv_dd = (rv_cumulative - rv_cumulative.expanding().max()) / rv_cumulative.expanding().max() * 100

ax2.plot(df_bt.index, bh_dd, label='Buy & Hold', linewidth=1.5, color='#3498db')
ax2.plot(df_bt.index, v0_dd, label='V0 Strategy', linewidth=1.5, color='#e74c3c')
ax2.plot(df_bt.index, rv_dd, label='RV Strategy', linewidth=1.5, color='#2ecc71')
ax2.set_ylabel('Drawdown (%)', fontsize=10)
ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 11.3 포지션 사이징
ax3 = axes[1, 0]
ax3.plot(df_bt.index, df_bt['v0_position'], label='V0 Position', linewidth=1, alpha=0.7, color='#e74c3c')
ax3.plot(df_bt.index, df_bt['rv_position'], label='RV Position', linewidth=1, alpha=0.7, color='#2ecc71')
ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
ax3.set_ylabel('Position Size', fontsize=10)
ax3.set_title('Dynamic Position Sizing', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 11.4 롤링 Sharpe Ratio (252일)
ax4 = axes[1, 1]
window = 252
bh_rolling_sharpe = (df_bt['bh_returns'].rolling(window).mean() * 252 - 0.02) / (df_bt['bh_returns'].rolling(window).std() * np.sqrt(252))
v0_rolling_sharpe = (df_bt['v0_strategy_returns'].rolling(window).mean() * 252 - 0.02) / (df_bt['v0_strategy_returns'].rolling(window).std() * np.sqrt(252))
rv_rolling_sharpe = (df_bt['rv_strategy_returns'].rolling(window).mean() * 252 - 0.02) / (df_bt['rv_strategy_returns'].rolling(window).std() * np.sqrt(252))

ax4.plot(df_bt.index, bh_rolling_sharpe, label='Buy & Hold', linewidth=1.5, alpha=0.8, color='#3498db')
ax4.plot(df_bt.index, v0_rolling_sharpe, label='V0 Strategy', linewidth=1.5, alpha=0.8, color='#e74c3c')
ax4.plot(df_bt.index, rv_rolling_sharpe, label='RV Strategy', linewidth=1.5, alpha=0.8, color='#2ecc71')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax4.set_ylabel('Sharpe Ratio', fontsize=10)
ax4.set_title(f'Rolling Sharpe Ratio ({window}d)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 11.5 성과 지표 비교 (바차트)
ax5 = axes[2, 0]
metrics_names = ['Annual\nReturn', 'Volatility', 'Sharpe', 'Max DD']
bh_values = [bh_metrics['annual_return']*100, bh_metrics['volatility']*100,
             bh_metrics['sharpe'], bh_metrics['max_drawdown']*100]
v0_values = [v0_metrics['annual_return']*100, v0_metrics['volatility']*100,
             v0_metrics['sharpe'], v0_metrics['max_drawdown']*100]
rv_values = [rv_metrics['annual_return']*100, rv_metrics['volatility']*100,
             rv_metrics['sharpe'], rv_metrics['max_drawdown']*100]

x = np.arange(len(metrics_names))
width = 0.25

ax5.bar(x - width, bh_values, width, label='Buy & Hold', color='#3498db', alpha=0.8)
ax5.bar(x, v0_values, width, label='V0', color='#e74c3c', alpha=0.8)
ax5.bar(x + width, rv_values, width, label='RV', color='#2ecc71', alpha=0.8)

ax5.set_xticks(x)
ax5.set_xticklabels(metrics_names)
ax5.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')
ax5.axhline(y=0, color='black', linewidth=0.5)

# 11.6 월별 수익률 히트맵
ax6 = axes[2, 1]
rv_monthly = df_bt['rv_strategy_returns'].resample('M').sum() * 100
bh_monthly = df_bt['bh_returns'].resample('M').sum() * 100
monthly_diff = rv_monthly - bh_monthly

ax6.bar(range(len(monthly_diff[-24:])), monthly_diff[-24:],
        color=['#2ecc71' if x > 0 else '#e74c3c' for x in monthly_diff[-24:]], alpha=0.7)
ax6.axhline(y=0, color='black', linewidth=1)
ax6.set_xlabel('Month (Last 24)', fontsize=10)
ax6.set_ylabel('RV - B&H (%)', fontsize=10)
ax6.set_title('Monthly Excess Return (RV vs Buy&Hold)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = "dashboard/figures/rv_economic_backtest.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n💾 시각화 저장: {output_path}")
plt.close()

# 12. 결과 저장
print("\n" + "="*70)
print("💾 결과 저장")
print("="*70)

import json
results = {
    "experiment": "RV Economic Backtest",
    "date": pd.Timestamp.now().isoformat(),
    "period": {
        "start": str(df_bt.index[0].date()),
        "end": str(df_bt.index[-1].date()),
        "days": len(df_bt)
    },
    "strategy": {
        "target_volatility": 0.15,
        "transaction_cost": 0.001,
        "position_range": [0.5, 2.0]
    },
    "results": {
        "buy_hold": bh_metrics,
        "v0_strategy": v0_metrics,
        "rv_strategy": rv_metrics
    },
    "comparison": {
        "rv_vs_bh_return": float(rv_metrics['annual_return'] - bh_metrics['annual_return']),
        "rv_vs_bh_vol": float(rv_metrics['volatility'] - bh_metrics['volatility']),
        "rv_vs_bh_sharpe": float(rv_metrics['sharpe'] - bh_metrics['sharpe']),
        "rv_vs_v0_sharpe": float(rv_metrics['sharpe'] - v0_metrics['sharpe'])
    }
}

# float 변환
for strategy in ['buy_hold', 'v0_strategy', 'rv_strategy']:
    for key in results['results'][strategy]:
        results['results'][strategy][key] = float(results['results'][strategy][key])

with open('data/raw/rv_economic_backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ 결과 저장: data/raw/rv_economic_backtest_results.json")

# 13. 최종 결론
print("\n" + "="*70)
print("🎯 경제적 백테스트 결론")
print("="*70)

rv_better_sharpe = rv_metrics['sharpe'] > bh_metrics['sharpe']
rv_better_vol = rv_metrics['volatility'] < bh_metrics['volatility']

print(f"""
{'✅' if rv_better_sharpe else '❌'} Sharpe Ratio: RV ({rv_metrics['sharpe']:.3f}) vs B&H ({bh_metrics['sharpe']:.3f})
{'✅' if rv_better_vol else '❌'} 변동성 감소: RV ({rv_metrics['volatility']*100:.2f}%) vs B&H ({bh_metrics['volatility']*100:.2f}%)

경제적 가치:
  - 연간 수익률: {rv_metrics['annual_return']*100:.2f}% (B&H 대비 {(rv_metrics['annual_return']-bh_metrics['annual_return'])*100:+.2f}%)
  - 리스크 조정 수익: Sharpe {rv_metrics['sharpe']:.3f} (B&H 대비 {rv_metrics['sharpe']-bh_metrics['sharpe']:+.3f})
  - 최대 낙폭: {rv_metrics['max_drawdown']*100:.2f}% (B&H 대비 {(rv_metrics['max_drawdown']-bh_metrics['max_drawdown'])*100:+.2f}%)

RV vs V0:
  - Sharpe: RV {rv_metrics['sharpe']:.3f} vs V0 {v0_metrics['sharpe']:.3f} ({rv_metrics['sharpe']-v0_metrics['sharpe']:+.3f})

결론: {"✅ 경제적 가치 입증 - 실전 적용 가능" if rv_better_sharpe else "⚠️ 경제적 가치 제한적 - 추가 검증 필요"}
""")

print("="*70)
