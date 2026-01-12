#!/usr/bin/env python3
"""
논문용 추가 실험 (확장)
========================

1. Out-of-sample 기간 분리 (COVID vs 정상)
2. 예측 지평 분석 (5일/10일/22일)
3. 변수 제외 분석 (Ablation Study)
4. VIX 분위수별 성능
5. 거래비용 반영 수익
6. 월별/요일별 패턴
7. 예측 구간 (Prediction Interval)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def load_data():
    """데이터 로드"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill().bfill()
    spy['returns'] = spy['Close'].pct_change()
    
    # 실현변동성
    spy['RV_1d'] = spy['returns'].abs() * np.sqrt(252) * 100
    spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
    spy['RV_10d'] = spy['returns'].rolling(10).std() * np.sqrt(252) * 100
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VRP
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    
    # 다양한 지평의 미래 RV
    spy['RV_future_5d'] = spy['RV_5d'].shift(-5)
    spy['RV_future_10d'] = spy['RV_10d'].shift(-10)
    spy['RV_future_22d'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future_22d']
    
    # 특성
    spy['VIX_lag1'] = spy['VIX'].shift(1)
    spy['VIX_lag5'] = spy['VIX'].shift(5)
    spy['VIX_change'] = spy['VIX'].pct_change()
    spy['VRP_lag1'] = spy['VRP'].shift(1)
    spy['VRP_lag5'] = spy['VRP'].shift(5)
    spy['VRP_ma5'] = spy['VRP'].rolling(5).mean()
    spy['regime_high'] = (spy['VIX'] >= 25).astype(int)
    spy['return_5d'] = spy['returns'].rolling(5).sum()
    spy['return_22d'] = spy['returns'].rolling(22).sum()
    
    spy = spy.replace([np.inf, -np.inf], np.nan)
    
    return spy


def experiment_1_period_comparison(spy):
    """실험 1: COVID vs 정상 기간 비교"""
    print("\n" + "=" * 60)
    print("[1/7] Out-of-sample 기간 분리 (COVID vs 정상)")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    periods = {
        'COVID (2020.03-2020.12)': ('2020-03-01', '2020-12-31'),
        'Post-COVID (2021)': ('2021-01-01', '2021-12-31'),
        '2022 (고금리)': ('2022-01-01', '2022-12-31'),
        '2023 (회복)': ('2023-01-01', '2023-12-31'),
        '2024-2025': ('2024-01-01', '2025-12-31')
    }
    
    results = {}
    
    print(f"\n  {'기간':<25} | {'샘플':>6} | {'R²':>8} | {'MAE':>8} | {'방향':>8}")
    print("  " + "-" * 65)
    
    for period_name, (start, end) in periods.items():
        spy_period = spy[(spy.index >= start) & (spy.index <= end)].copy()
        spy_period = spy_period.dropna(subset=feature_cols + ['VRP_true'])
        
        if len(spy_period) < 50:
            print(f"  {period_name:<25} | 샘플 부족")
            continue
        
        X = spy_period[feature_cols].values
        y_rv = spy_period['RV_future_22d'].values
        y_vrp = spy_period['VRP_true'].values
        vix = spy_period['VIX'].values
        
        split_idx = int(len(spy_period) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_rv[:split_idx], y_rv[split_idx:]
        vix_test = vix[split_idx:]
        y_vrp_test = y_vrp[split_idx:]
        
        if len(X_test) < 20:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        rv_pred = en.predict(X_test_s)
        vrp_pred = vix_test - rv_pred
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        mae = mean_absolute_error(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results[period_name] = {
            'n_samples': len(spy_period),
            'r2': float(r2),
            'mae': float(mae),
            'direction_accuracy': float(dir_acc)
        }
        
        print(f"  {period_name:<25} | {len(spy_period):>6} | {r2:>8.4f} | {mae:>8.2f} | {dir_acc*100:>7.1f}%")
    
    return results


def experiment_2_horizon_analysis(spy):
    """실험 2: 예측 지평 분석"""
    print("\n" + "=" * 60)
    print("[2/7] 예측 지평 분석 (5일/10일/22일)")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    horizons = {
        '5일': 'RV_future_5d',
        '10일': 'RV_future_10d',
        '22일': 'RV_future_22d'
    }
    
    results = {}
    
    print(f"\n  {'지평':<10} | {'RV R²':>8} | {'VRP R²':>8} | {'MAE':>8} | {'방향':>8}")
    print("  " + "-" * 55)
    
    for horizon_name, target_col in horizons.items():
        spy_h = spy.dropna(subset=feature_cols + [target_col])
        
        if len(spy_h) < 100:
            continue
        
        X = spy_h[feature_cols].values
        y = spy_h[target_col].values
        vix = spy_h['VIX'].values
        
        split_idx = int(len(spy_h) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        vix_test = vix[split_idx:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        rv_pred = en.predict(X_test_s)
        vrp_pred = vix_test - rv_pred
        
        rv_r2 = r2_score(y_test, rv_pred)
        y_vrp_test = vix_test - y_test
        vrp_r2 = r2_score(y_vrp_test, vrp_pred)
        mae = mean_absolute_error(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results[horizon_name] = {
            'rv_r2': float(rv_r2),
            'vrp_r2': float(vrp_r2),
            'mae': float(mae),
            'direction_accuracy': float(dir_acc)
        }
        
        print(f"  {horizon_name:<10} | {rv_r2:>8.4f} | {vrp_r2:>8.4f} | {mae:>8.2f} | {dir_acc*100:>7.1f}%")
    
    return results


def experiment_3_ablation(spy):
    """실험 3: 변수 제외 분석 (Ablation Study)"""
    print("\n" + "=" * 60)
    print("[3/7] 변수 제외 분석 (Ablation Study)")
    print("=" * 60)
    
    all_features = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=all_features + ['RV_future_22d'])
    
    X_all = spy_clean[all_features].values
    y = spy_clean['RV_future_22d'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    # 전체 모델 성능
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_all[:split_idx])
    X_test_s = scaler.transform(X_all[split_idx:])
    
    en_full = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en_full.fit(X_train_s, y[:split_idx])
    vrp_pred_full = vix_test - en_full.predict(X_test_s)
    r2_full = r2_score(y_vrp_test, vrp_pred_full)
    
    print(f"\n  전체 모델 (12개 특성): R² = {r2_full:.4f}")
    print(f"\n  {'제외 변수':<15} | {'R²':>8} | {'변화':>10}")
    print("  " + "-" * 40)
    
    results = {'full_model': float(r2_full)}
    
    for exclude_feat in all_features:
        features = [f for f in all_features if f != exclude_feat]
        
        X = spy_clean[features].values
        X_train_s = scaler.fit_transform(X[:split_idx])
        X_test_s = scaler.transform(X[split_idx:])
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y[:split_idx])
        vrp_pred = vix_test - en.predict(X_test_s)
        r2 = r2_score(y_vrp_test, vrp_pred)
        
        change = (r2 - r2_full) / abs(r2_full) * 100
        
        results[f'without_{exclude_feat}'] = float(r2)
        
        marker = "⬇️" if change < -5 else "⬆️" if change > 5 else "➡️"
        print(f"  {exclude_feat:<15} | {r2:>8.4f} | {change:>+9.1f}% {marker}")
    
    return results


def experiment_4_vix_quantile(spy):
    """실험 4: VIX 분위수별 성능"""
    print("\n" + "=" * 60)
    print("[4/7] VIX 분위수별 성능")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future_22d'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future_22d'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    
    vrp_pred = vix[split_idx:] - en.predict(X_test_s)
    y_vrp_test = y_vrp[split_idx:]
    vix_test = vix[split_idx:]
    
    # VIX 분위수
    quantiles = {
        'Q1 (0-25%)': (0, np.percentile(vix_test, 25)),
        'Q2 (25-50%)': (np.percentile(vix_test, 25), np.percentile(vix_test, 50)),
        'Q3 (50-75%)': (np.percentile(vix_test, 50), np.percentile(vix_test, 75)),
        'Q4 (75-100%)': (np.percentile(vix_test, 75), 100)
    }
    
    results = {}
    
    print(f"\n  {'분위수':<15} | {'VIX 범위':>15} | {'샘플':>6} | {'R²':>8} | {'방향':>8}")
    print("  " + "-" * 65)
    
    for q_name, (low, high) in quantiles.items():
        mask = (vix_test >= low) & (vix_test < high)
        
        if mask.sum() >= 10:
            r2 = r2_score(y_vrp_test[mask], vrp_pred[mask])
            dir_acc = ((y_vrp_test[mask] > y_vrp_test.mean()) == 
                      (vrp_pred[mask] > y_vrp_test.mean())).mean()
            
            results[q_name] = {
                'vix_range': f"{low:.1f}-{high:.1f}",
                'n_samples': int(mask.sum()),
                'r2': float(r2),
                'direction_accuracy': float(dir_acc)
            }
            
            print(f"  {q_name:<15} | {low:>6.1f}-{high:>6.1f} | {mask.sum():>6} | "
                  f"{r2:>8.4f} | {dir_acc*100:>7.1f}%")
    
    return results


def experiment_5_trading_cost(spy):
    """실험 5: 거래비용 반영 수익"""
    print("\n" + "=" * 60)
    print("[5/7] 거래비용 반영 수익")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future_22d'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future_22d'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    
    vrp_pred = vix[split_idx:] - en.predict(X_test_s)
    y_vrp_test = y_vrp[split_idx:]
    
    vrp_mean = y_vrp_test.mean()
    positions = (vrp_pred > vrp_mean).astype(int)
    
    # 거래비용 시나리오
    costs = [0.0, 0.05, 0.10, 0.20, 0.50]  # % per trade
    
    results = {}
    
    print(f"\n  {'거래비용':>10} | {'총 수익':>10} | {'순 수익':>10} | {'거래 횟수':>10}")
    print("  " + "-" * 50)
    
    for cost in costs:
        gross_returns = positions * y_vrp_test
        
        # 포지션 변경 시 거래비용
        position_changes = np.abs(np.diff(positions, prepend=0))
        total_cost = position_changes.sum() * cost
        
        gross_total = gross_returns.sum()
        net_total = gross_total - total_cost
        n_trades = positions.sum()
        
        results[f'cost_{cost}'] = {
            'cost_pct': float(cost),
            'gross_return': float(gross_total),
            'net_return': float(net_total),
            'total_cost': float(total_cost),
            'n_trades': int(n_trades)
        }
        
        print(f"  {cost:>9.2f}% | {gross_total:>9.2f}% | {net_total:>9.2f}% | {n_trades:>10}")
    
    return results


def experiment_6_seasonality(spy):
    """실험 6: 월별/요일별 패턴"""
    print("\n" + "=" * 60)
    print("[6/7] 월별/요일별 패턴")
    print("=" * 60)
    
    spy_clean = spy.dropna(subset=['VRP_true'])
    
    # 월별
    spy_clean['month'] = spy_clean.index.month
    monthly = spy_clean.groupby('month')['VRP_true'].agg(['mean', 'std', 'count'])
    
    print(f"\n  📊 월별 VRP 평균:")
    print(f"  {'월':>4} | {'평균':>8} | {'표준편차':>8} | {'샘플':>6}")
    print("  " + "-" * 35)
    
    month_results = {}
    for month in range(1, 13):
        if month in monthly.index:
            row = monthly.loc[month]
            month_results[month] = {
                'mean': float(row['mean']),
                'std': float(row['std']),
                'count': int(row['count'])
            }
            print(f"  {month:>4} | {row['mean']:>8.2f} | {row['std']:>8.2f} | {row['count']:>6}")
    
    # 요일별
    spy_clean['weekday'] = spy_clean.index.weekday
    weekday = spy_clean.groupby('weekday')['VRP_true'].agg(['mean', 'std', 'count'])
    weekday_names = ['월', '화', '수', '목', '금']
    
    print(f"\n  📊 요일별 VRP 평균:")
    print(f"  {'요일':>4} | {'평균':>8} | {'표준편차':>8} | {'샘플':>6}")
    print("  " + "-" * 35)
    
    weekday_results = {}
    for wd in range(5):
        if wd in weekday.index:
            row = weekday.loc[wd]
            weekday_results[weekday_names[wd]] = {
                'mean': float(row['mean']),
                'std': float(row['std']),
                'count': int(row['count'])
            }
            print(f"  {weekday_names[wd]:>4} | {row['mean']:>8.2f} | {row['std']:>8.2f} | {row['count']:>6}")
    
    return {'monthly': month_results, 'weekday': weekday_results}


def experiment_7_prediction_interval(spy):
    """실험 7: 예측 구간 (Prediction Interval)"""
    print("\n" + "=" * 60)
    print("[7/7] 예측 구간 (Prediction Interval)")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future_22d'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future_22d'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    
    rv_pred = en.predict(X_test_s)
    vrp_pred = vix[split_idx:] - rv_pred
    y_vrp_test = y_vrp[split_idx:]
    
    # 잔차 기반 예측 구간
    residuals = y_vrp_test - vrp_pred
    residual_std = residuals.std()
    
    # 신뢰 수준별 예측 구간 커버리지
    confidence_levels = [0.50, 0.80, 0.90, 0.95]
    
    results = {}
    
    print(f"\n  📊 예측 구간 커버리지:")
    print(f"  {'신뢰수준':>10} | {'구간 폭':>10} | {'실제 커버리지':>12}")
    print("  " + "-" * 40)
    
    from scipy import stats
    
    for conf in confidence_levels:
        z = stats.norm.ppf((1 + conf) / 2)
        interval_width = 2 * z * residual_std
        
        lower = vrp_pred - z * residual_std
        upper = vrp_pred + z * residual_std
        
        coverage = ((y_vrp_test >= lower) & (y_vrp_test <= upper)).mean()
        
        results[f'conf_{int(conf*100)}'] = {
            'confidence': float(conf),
            'interval_width': float(interval_width),
            'actual_coverage': float(coverage)
        }
        
        match = "✓" if abs(coverage - conf) < 0.05 else "✗"
        print(f"  {conf*100:>9.0f}% | {interval_width:>9.2f}% | {coverage*100:>11.1f}% {match}")
    
    results['residual_std'] = float(residual_std)
    
    return results


def main():
    print("\n" + "🔬" * 30)
    print("논문용 추가 실험 (확장)")
    print("🔬" * 30)
    
    # 데이터 로드
    print("\n데이터 로드...")
    spy = load_data()
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    # 실험 실행
    results = {}
    
    results['period_comparison'] = experiment_1_period_comparison(spy)
    results['horizon_analysis'] = experiment_2_horizon_analysis(spy)
    results['ablation_study'] = experiment_3_ablation(spy)
    results['vix_quantile'] = experiment_4_vix_quantile(spy)
    results['trading_cost'] = experiment_5_trading_cost(spy)
    results['seasonality'] = experiment_6_seasonality(spy)
    results['prediction_interval'] = experiment_7_prediction_interval(spy)
    
    # 저장
    results['timestamp'] = datetime.now().isoformat()
    
    with open('paper/extended_experiments.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 실험 완료")
    print("=" * 60)
    print(f"  ✓ 결과 저장: paper/extended_experiments.json")


if __name__ == '__main__':
    main()
