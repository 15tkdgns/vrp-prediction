#!/usr/bin/env python3
"""
논문용 테이블 및 검증 데이터 생성
=================================

1. 기술통계량 테이블
2. 상관관계 행렬
3. 모델 비교 테이블 (확장)
4. 특성 중요도 테이블
5. 강건성 검증 (복수 시드)
6. Walk-Forward 검증
7. 표본외 성능 테이블
8. 트레이딩 성과 테이블
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
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
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VRP
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
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


def table_1_descriptive_stats(spy):
    """테이블 1: 기술통계량"""
    print("\n" + "=" * 70)
    print("📊 Table 1: Descriptive Statistics")
    print("=" * 70)
    
    spy_clean = spy.dropna(subset=['VIX', 'RV_22d', 'VRP', 'VRP_true', 'returns'])
    
    variables = {
        'VIX': spy_clean['VIX'],
        'RV_22d': spy_clean['RV_22d'],
        'VRP': spy_clean['VRP'],
        'VRP_true': spy_clean['VRP_true'],
        'Daily Returns (%)': spy_clean['returns'] * 100
    }
    
    stats_data = []
    for name, series in variables.items():
        stats_data.append({
            'Variable': name,
            'N': len(series),
            'Mean': f"{series.mean():.4f}",
            'Std': f"{series.std():.4f}",
            'Min': f"{series.min():.4f}",
            'Q25': f"{series.quantile(0.25):.4f}",
            'Median': f"{series.median():.4f}",
            'Q75': f"{series.quantile(0.75):.4f}",
            'Max': f"{series.max():.4f}",
            'Skewness': f"{series.skew():.4f}",
            'Kurtosis': f"{series.kurtosis():.4f}"
        })
    
    df = pd.DataFrame(stats_data)
    print(df.to_string(index=False))
    
    return df.to_dict('records')


def table_2_correlation_matrix(spy):
    """테이블 2: 상관관계 행렬"""
    print("\n" + "=" * 70)
    print("📊 Table 2: Correlation Matrix")
    print("=" * 70)
    
    spy_clean = spy.dropna(subset=['VIX', 'RV_22d', 'VRP', 'VRP_true', 'VIX_lag1', 'VRP_lag1'])
    
    cols = ['VIX', 'RV_22d', 'VRP', 'VRP_true', 'VIX_lag1', 'VRP_lag1']
    corr_matrix = spy_clean[cols].corr()
    
    print(corr_matrix.round(4).to_string())
    
    return corr_matrix.round(4).to_dict()


def table_3_model_comparison(spy):
    """테이블 3: 모델 비교 (확장)"""
    print("\n" + "=" * 70)
    print("📊 Table 3: Model Comparison")
    print("=" * 70)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # HAR-RV 특성
    har_idx = [feature_cols.index(f) for f in ['RV_1d', 'RV_5d', 'RV_22d']]
    
    models = {
        'HAR-RV': (LinearRegression(), har_idx),
        'Ridge': (Ridge(alpha=1.0, random_state=SEED), None),
        'Lasso': (Lasso(alpha=0.01, random_state=SEED, max_iter=10000), None),
        'ElasticNet': (ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000), None),
        'Random Forest': (RandomForestRegressor(n_estimators=100, max_depth=6, random_state=SEED), None),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=SEED), None)
    }
    
    results = []
    
    for name, (model, feat_idx) in models.items():
        if feat_idx:
            X_tr = X_train_s[:, feat_idx]
            X_te = X_test_s[:, feat_idx]
        else:
            X_tr = X_train_s
            X_te = X_test_s
        
        model.fit(X_tr, y_train)
        rv_pred = model.predict(X_te)
        vrp_pred = vix_test - rv_pred
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        rmse = np.sqrt(mean_squared_error(y_vrp_test, vrp_pred))
        mae = mean_absolute_error(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results.append({
            'Model': name,
            'R²': f"{r2:.4f}",
            'RMSE': f"{rmse:.4f}",
            'MAE': f"{mae:.4f}",
            'Direction Acc.': f"{dir_acc*100:.2f}%"
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return results


def table_4_feature_importance(spy):
    """테이블 4: 특성 중요도"""
    print("\n" + "=" * 70)
    print("📊 Table 4: Feature Importance")
    print("=" * 70)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': np.abs(en.coef_)
    }).sort_values('Coefficient', ascending=False)
    
    importance['Rank'] = range(1, len(importance) + 1)
    importance = importance[['Rank', 'Feature', 'Coefficient']]
    importance['Coefficient'] = importance['Coefficient'].apply(lambda x: f"{x:.4f}")
    
    print(importance.to_string(index=False))
    
    return importance.to_dict('records')


def table_5_robustness_seeds(spy):
    """테이블 5: 복수 시드 강건성"""
    print("\n" + "=" * 70)
    print("📊 Table 5: Robustness Check (Multiple Seeds)")
    print("=" * 70)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    seeds = [0, 21, 42, 123, 456, 789, 1000, 2023, 2024, 2025]
    
    results = []
    
    for seed in seeds:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[:split_idx])
        X_test_s = scaler.transform(X[split_idx:])
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed, max_iter=10000)
        en.fit(X_train_s, y[:split_idx])
        vrp_pred = vix_test - en.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results.append({'Seed': seed, 'R²': r2, 'Direction Acc.': dir_acc})
    
    df = pd.DataFrame(results)
    
    print(f"  {'Seed':>6} | {'R²':>8} | {'Direction Acc.':>14}")
    print("  " + "-" * 35)
    for _, row in df.iterrows():
        print(f"  {row['Seed']:>6} | {row['R²']:>8.4f} | {row['Direction Acc.']*100:>13.2f}%")
    
    print(f"\n  Mean R²: {df['R²'].mean():.4f} ± {df['R²'].std():.4f}")
    print(f"  Mean Direction: {df['Direction Acc.'].mean()*100:.2f}% ± {df['Direction Acc.'].std()*100:.2f}%")
    
    return {
        'individual': results,
        'mean_r2': float(df['R²'].mean()),
        'std_r2': float(df['R²'].std()),
        'mean_direction': float(df['Direction Acc.'].mean()),
        'std_direction': float(df['Direction Acc.'].std())
    }


def table_6_walk_forward(spy):
    """테이블 6: Walk-Forward 검증"""
    print("\n" + "=" * 70)
    print("📊 Table 6: Walk-Forward Validation")
    print("=" * 70)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    dates = spy_clean.index
    
    train_size = 252  # 1년
    test_size = 63    # 1분기
    
    results = []
    
    i = train_size
    while i + test_size <= len(X):
        X_train = X[i-train_size:i]
        y_train = y[i-train_size:i]
        X_test = X[i:i+test_size]
        y_test = y[i:i+test_size]
        vix_test = vix[i:i+test_size]
        y_vrp_test = y_vrp[i:i+test_size]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        vrp_pred = vix_test - en.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results.append({
            'Period': f"{dates[i].strftime('%Y-%m')} - {dates[min(i+test_size-1, len(dates)-1)].strftime('%Y-%m')}",
            'R²': r2,
            'Direction Acc.': dir_acc
        })
        
        i += test_size
    
    print(f"  {'Period':>20} | {'R²':>8} | {'Direction Acc.':>14}")
    print("  " + "-" * 50)
    for row in results:
        print(f"  {row['Period']:>20} | {row['R²']:>8.4f} | {row['Direction Acc.']*100:>13.2f}%")
    
    r2_values = [r['R²'] for r in results]
    dir_values = [r['Direction Acc.'] for r in results]
    
    print(f"\n  Mean R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
    print(f"  Positive R² periods: {sum(1 for r in r2_values if r > 0)}/{len(r2_values)}")
    
    return results


def table_7_out_of_sample(spy):
    """테이블 7: 표본외 성능 (다양한 분할)"""
    print("\n" + "=" * 70)
    print("📊 Table 7: Out-of-Sample Performance")
    print("=" * 70)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    splits = [0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    for split in splits:
        split_idx = int(len(spy_clean) * split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        vix_test = vix[split_idx:]
        y_vrp_test = y_vrp[split_idx:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        vrp_pred = vix_test - en.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        rmse = np.sqrt(mean_squared_error(y_vrp_test, vrp_pred))
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results.append({
            'Train/Test Split': f"{int(split*100)}/{int((1-split)*100)}",
            'Train Size': split_idx,
            'Test Size': len(X) - split_idx,
            'R²': f"{r2:.4f}",
            'RMSE': f"{rmse:.4f}",
            'Direction Acc.': f"{dir_acc*100:.2f}%"
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return results


def table_8_trading_performance(spy):
    """테이블 8: 트레이딩 성과"""
    print("\n" + "=" * 70)
    print("📊 Table 8: Trading Performance")
    print("=" * 70)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    vrp_pred = vix_test - en.predict(X_test_s)
    
    vrp_mean = y_vrp_test.mean()
    
    strategies = {
        'Buy & Hold': np.ones(len(y_vrp_test)),
        'Prediction-based': (vrp_pred > vrp_mean).astype(int),
        'VIX > 20': (vix_test > 20).astype(int),
        'VIX > 25': (vix_test > 25).astype(int)
    }
    
    results = []
    
    for name, positions in strategies.items():
        returns = positions * y_vrp_test
        
        n_trades = int(positions.sum())
        total_return = returns.sum()
        avg_return = returns[positions == 1].mean() if n_trades > 0 else 0
        std_return = returns[positions == 1].std() if n_trades > 1 else 0
        sharpe = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        win_rate = (returns[positions == 1] > 0).mean() if n_trades > 0 else 0
        max_drawdown = pd.Series(returns).cumsum().cummax() - pd.Series(returns).cumsum()
        max_dd = max_drawdown.max()
        
        results.append({
            'Strategy': name,
            'N Trades': n_trades,
            'Total Return': f"{total_return:.2f}%",
            'Avg Return': f"{avg_return:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Win Rate': f"{win_rate*100:.1f}%",
            'Max Drawdown': f"{max_dd:.2f}%"
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return results


def main():
    print("\n" + "📋" * 30)
    print("논문용 테이블 및 검증 데이터 생성")
    print("📋" * 30)
    
    # 데이터 로드
    print("\n데이터 로드...")
    spy = load_data()
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    # 테이블 생성
    tables = {}
    
    tables['table1_descriptive'] = table_1_descriptive_stats(spy)
    tables['table2_correlation'] = table_2_correlation_matrix(spy)
    tables['table3_model_comparison'] = table_3_model_comparison(spy)
    tables['table4_feature_importance'] = table_4_feature_importance(spy)
    tables['table5_robustness'] = table_5_robustness_seeds(spy)
    tables['table6_walk_forward'] = table_6_walk_forward(spy)
    tables['table7_out_of_sample'] = table_7_out_of_sample(spy)
    tables['table8_trading'] = table_8_trading_performance(spy)
    
    # 저장
    tables['timestamp'] = datetime.now().isoformat()
    
    with open('paper/paper_tables.json', 'w') as f:
        json.dump(tables, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("📊 테이블 생성 완료")
    print("=" * 70)
    print(f"  ✓ 저장: paper/paper_tables.json")
    print(f"  ✓ 8개 테이블 생성됨")


if __name__ == '__main__':
    main()
