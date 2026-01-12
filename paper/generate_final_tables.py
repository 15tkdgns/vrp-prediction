#!/usr/bin/env python3
"""
논문용 최종 테이블 생성
======================

Table 1: 기술통계량
Table 2: 상관관계 행렬
Table 3: 모델 비교 (전체)
Table 4: 카테고리별 최고 모델
Table 5: 특성 중요도
Table 6: 강건성 검증
Table 7: 트레이딩 성과
Table 8: 다중 자산 분석
Table 9: 통계적 유의성
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)

# 추가 라이브러리
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

try:
    from arch import arch_model
    HAS_ARCH = True
except:
    HAS_ARCH = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(SEED)
    HAS_TF = True
except:
    HAS_TF = False


def load_and_prepare_data():
    """데이터 로드 및 특성 생성"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    if csv_path.exists():
        spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
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
    
    spy = spy.replace([np.inf, -np.inf], np.nan).dropna()
    
    return spy


def table_1_descriptive_statistics(spy):
    """Table 1: 기술통계량"""
    print("\n" + "=" * 80)
    print("Table 1: Descriptive Statistics")
    print("=" * 80)
    
    variables = {
        'VIX': spy['VIX'],
        'RV (22-day)': spy['RV_22d'],
        'VRP': spy['VRP'],
        'VRP (True)': spy['VRP_true'],
        'Daily Return (%)': spy['returns'] * 100
    }
    
    stats_list = []
    for name, series in variables.items():
        stats_list.append({
            'Variable': name,
            'N': len(series),
            'Mean': round(series.mean(), 4),
            'Std': round(series.std(), 4),
            'Min': round(series.min(), 4),
            'P25': round(series.quantile(0.25), 4),
            'Median': round(series.median(), 4),
            'P75': round(series.quantile(0.75), 4),
            'Max': round(series.max(), 4),
            'Skewness': round(series.skew(), 4),
            'Kurtosis': round(series.kurtosis(), 4)
        })
    
    df = pd.DataFrame(stats_list)
    print(df.to_string(index=False))
    
    return df.to_dict('records')


def table_2_correlation_matrix(spy):
    """Table 2: 상관관계 행렬"""
    print("\n" + "=" * 80)
    print("Table 2: Correlation Matrix")
    print("=" * 80)
    
    cols = ['VIX', 'RV_22d', 'VRP', 'VRP_true', 'VIX_lag1', 'VRP_lag1']
    corr = spy[cols].corr().round(4)
    
    print(corr.to_string())
    
    return corr.to_dict()


def table_3_model_comparison(spy):
    """Table 3: 모델 비교 (전체)"""
    print("\n" + "=" * 80)
    print("Table 3: Model Comparison (All Models)")
    print("=" * 80)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vix = spy['VIX'].values
    y_vrp = spy['VRP_true'].values
    
    split_idx = int(len(spy) * 0.8)
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    # 모델 정의
    models = {
        # 전통 모델
        'HAR-RV': ('Traditional', LinearRegression(), [0, 1, 2]),  # RV_1d, RV_5d, RV_22d
        'EWMA': ('Traditional', None, None),
        # 선형 모델
        'OLS': ('Linear', LinearRegression(), None),
        'Ridge': ('Linear', Ridge(alpha=1.0, random_state=SEED), None),
        'Lasso': ('Linear', Lasso(alpha=0.01, random_state=SEED, max_iter=10000), None),
        'ElasticNet (best)': ('Linear', ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000), None),
        # 트리 기반
        'Random Forest': ('Tree', RandomForestRegressor(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1), None),
        'GradientBoosting': ('Tree', GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=SEED), None),
    }
    
    if HAS_XGB:
        models['XGBoost'] = ('Tree', xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=SEED, verbosity=0), None)
    
    if HAS_LGB:
        models['LightGBM'] = ('Tree', lgb.LGBMRegressor(n_estimators=100, max_depth=4, random_state=SEED, verbosity=-1), None)
    
    # MLP
    models['MLP (128,64)'] = ('Deep Learning', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED, early_stopping=True), None)
    
    # 앙상블
    models['Stacking (best)'] = ('Ensemble', StackingRegressor(
        estimators=[
            ('en', ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)),
            ('ridge', Ridge(alpha=1.0, random_state=SEED))
        ],
        final_estimator=Ridge(alpha=0.1, random_state=SEED)
    ), None)
    
    results = []
    
    for name, (category, model, feat_idx) in models.items():
        try:
            if name == 'EWMA':
                # EWMA 직접 계산
                lambda_param = 0.94
                returns = spy['returns'].values * 100
                ewma_var = np.zeros(len(returns))
                ewma_var[0] = returns[0]**2
                for i in range(1, len(returns)):
                    ewma_var[i] = lambda_param * ewma_var[i-1] + (1 - lambda_param) * returns[i-1]**2
                ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)
                vrp_pred = vix[split_idx:] - ewma_vol[split_idx:]
            elif feat_idx is not None:
                # HAR-RV
                X_tr = X_train_s[:, feat_idx]
                X_te = X_test_s[:, feat_idx]
                model.fit(X_tr, y[:split_idx])
                vrp_pred = vix_test - model.predict(X_te)
            else:
                model.fit(X_train_s, y[:split_idx])
                vrp_pred = vix_test - model.predict(X_test_s)
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            rmse = np.sqrt(mean_squared_error(y_vrp_test, vrp_pred))
            mae = mean_absolute_error(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            results.append({
                'Category': category,
                'Model': name,
                'R²': round(r2, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'Direction (%)': round(dir_acc * 100, 2)
            })
        except Exception as e:
            pass
    
    df = pd.DataFrame(results)
    df = df.sort_values('R²', ascending=False)
    print(df.to_string(index=False))
    
    return df.to_dict('records')


def table_4_category_best(results_table3):
    """Table 4: 카테고리별 최고 모델"""
    print("\n" + "=" * 80)
    print("Table 4: Best Model by Category")
    print("=" * 80)
    
    df = pd.DataFrame(results_table3)
    best_models = df.loc[df.groupby('Category')['R²'].idxmax()]
    best_models = best_models.sort_values('R²', ascending=False)
    
    print(best_models[['Category', 'Model', 'R²', 'Direction (%)']].to_string(index=False))
    
    return best_models.to_dict('records')


def table_5_feature_importance(spy):
    """Table 5: 특성 중요도"""
    print("\n" + "=" * 80)
    print("Table 5: Feature Importance (ElasticNet Coefficients)")
    print("=" * 80)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    
    split_idx = int(len(spy) * 0.8)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    
    en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': en.coef_,
        'Abs_Coefficient': np.abs(en.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    importance['Rank'] = range(1, len(importance) + 1)
    importance = importance[['Rank', 'Feature', 'Coefficient', 'Abs_Coefficient']]
    importance['Coefficient'] = importance['Coefficient'].round(4)
    importance['Abs_Coefficient'] = importance['Abs_Coefficient'].round(4)
    
    print(importance.to_string(index=False))
    
    return importance.to_dict('records')


def table_6_robustness(spy):
    """Table 6: 강건성 검증"""
    print("\n" + "=" * 80)
    print("Table 6: Robustness Check")
    print("=" * 80)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vix = spy['VIX'].values
    y_vrp = spy['VRP_true'].values
    
    results = []
    
    # 다양한 분할 비율
    for split_ratio in [0.6, 0.7, 0.8, 0.9]:
        split_idx = int(len(spy) * split_ratio)
        vix_test = vix[split_idx:]
        y_vrp_test = y_vrp[split_idx:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[:split_idx])
        X_test_s = scaler.transform(X[split_idx:])
        
        en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y[:split_idx])
        vrp_pred = vix_test - en.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results.append({
            'Test': f'{int(split_ratio*100)}/{int((1-split_ratio)*100)} Split',
            'Train Size': split_idx,
            'Test Size': len(spy) - split_idx,
            'R²': round(r2, 4),
            'Direction (%)': round(dir_acc * 100, 2)
        })
    
    # 연도별 분석
    for year in range(2021, 2025):
        mask = spy.index.year == year
        if mask.sum() < 50:
            continue
        
        spy_year = spy[mask]
        X_year = spy_year[feature_cols].values
        y_year = spy_year['RV_future'].values
        vix_year = spy_year['VIX'].values
        y_vrp_year = spy_year['VRP_true'].values
        
        split_idx = int(len(spy_year) * 0.7)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_year[:split_idx])
        X_test_s = scaler.transform(X_year[split_idx:])
        
        en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_year[:split_idx])
        vrp_pred = vix_year[split_idx:] - en.predict(X_test_s)
        y_vrp_test = y_vrp_year[split_idx:]
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results.append({
            'Test': f'Year {year}',
            'Train Size': split_idx,
            'Test Size': len(spy_year) - split_idx,
            'R²': round(r2, 4),
            'Direction (%)': round(dir_acc * 100, 2)
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return results


def table_7_trading_performance(spy):
    """Table 7: 트레이딩 성과"""
    print("\n" + "=" * 80)
    print("Table 7: Trading Performance")
    print("=" * 80)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vix = spy['VIX'].values
    y_vrp = spy['VRP_true'].values
    
    split_idx = int(len(spy) * 0.8)
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
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
        
        results.append({
            'Strategy': name,
            'N Trades': n_trades,
            'Total Return (%)': round(total_return, 2),
            'Avg Return (%)': round(avg_return, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Win Rate (%)': round(win_rate * 100, 1)
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return results


def table_8_multi_asset():
    """Table 8: 다중 자산 분석"""
    print("\n" + "=" * 80)
    print("Table 8: Multi-Asset Analysis")
    print("=" * 80)
    
    assets = [
        ('SPY', '^VIX', 'S&P 500'),
        ('EFA', '^VIX', 'EAFE (Developed)'),
        ('EEM', '^VIX', 'Emerging Markets'),
        ('GLD', '^VIX', 'Gold')
    ]
    
    results = []
    
    for ticker, vol_ticker, name in assets:
        try:
            asset = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
            vol = yf.download(vol_ticker, start='2015-01-01', end='2025-01-01', progress=False)
            
            if isinstance(asset.columns, pd.MultiIndex):
                asset.columns = asset.columns.get_level_values(0)
            if isinstance(vol.columns, pd.MultiIndex):
                vol.columns = vol.columns.get_level_values(0)
            
            asset['Vol'] = vol['Close'].reindex(asset.index).ffill().bfill()
            asset['returns'] = asset['Close'].pct_change()
            asset['RV_22d'] = asset['returns'].rolling(22).std() * np.sqrt(252) * 100
            asset['VRP'] = asset['Vol'] - asset['RV_22d']
            asset['RV_future'] = asset['RV_22d'].shift(-22)
            asset['VRP_true'] = asset['Vol'] - asset['RV_future']
            
            asset['RV_1d'] = asset['returns'].abs() * np.sqrt(252) * 100
            asset['RV_5d'] = asset['returns'].rolling(5).std() * np.sqrt(252) * 100
            asset['Vol_lag1'] = asset['Vol'].shift(1)
            asset['Vol_lag5'] = asset['Vol'].shift(5)
            asset['Vol_change'] = asset['Vol'].pct_change()
            asset['VRP_lag1'] = asset['VRP'].shift(1)
            asset['VRP_lag5'] = asset['VRP'].shift(5)
            asset['VRP_ma5'] = asset['VRP'].rolling(5).mean()
            asset['regime_high'] = (asset['Vol'] >= 25).astype(int)
            asset['return_5d'] = asset['returns'].rolling(5).sum()
            asset['return_22d'] = asset['returns'].rolling(22).sum()
            
            asset = asset.replace([np.inf, -np.inf], np.nan).dropna()
            
            feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5', 
                           'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                           'regime_high', 'return_5d', 'return_22d']
            
            X = asset[feature_cols].values
            y = asset['RV_future'].values
            vol_vals = asset['Vol'].values
            y_vrp = asset['VRP_true'].values
            
            split_idx = int(len(asset) * 0.8)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X[:split_idx])
            X_test_s = scaler.transform(X[split_idx:])
            
            en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
            en.fit(X_train_s, y[:split_idx])
            vrp_pred = vol_vals[split_idx:] - en.predict(X_test_s)
            y_vrp_test = y_vrp[split_idx:]
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            # VIX-RV 상관
            vix_rv_corr = asset['Vol'].corr(asset['RV_22d'])
            
            results.append({
                'Asset': name,
                'Ticker': ticker,
                'N': len(asset),
                'VIX-RV Corr': round(vix_rv_corr, 4),
                'R²': round(r2, 4),
                'Direction (%)': round(dir_acc * 100, 2)
            })
        except Exception as e:
            pass
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return results


def table_9_statistical_tests(spy):
    """Table 9: 통계적 유의성"""
    print("\n" + "=" * 80)
    print("Table 9: Statistical Significance Tests")
    print("=" * 80)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vix = spy['VIX'].values
    y_vrp = spy['VRP_true'].values
    
    split_idx = int(len(spy) * 0.8)
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    vrp_pred = vix_test - en.predict(X_test_s)
    
    errors = y_vrp_test - vrp_pred
    
    results = []
    
    # 1. Bias test
    t_stat, p_value = stats.ttest_1samp(errors, 0)
    results.append({
        'Test': 'Bias (t-test)',
        'Statistic': round(t_stat, 4),
        'p-value': f"{p_value:.6f}",
        'Conclusion': 'No Bias' if p_value > 0.05 else 'Biased'
    })
    
    # 2. F-test for R²
    n = len(y_vrp_test)
    k = len(feature_cols)
    r2 = r2_score(y_vrp_test, vrp_pred)
    f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
    p_value_f = 1 - stats.f.cdf(f_stat, k, n - k - 1)
    results.append({
        'Test': 'R² (F-test)',
        'Statistic': round(f_stat, 4),
        'p-value': f"{p_value_f:.6f}",
        'Conclusion': 'Significant' if p_value_f < 0.05 else 'Not Significant'
    })
    
    # 3. Direction accuracy (Binomial test)
    vrp_mean = y_vrp_test.mean()
    correct = ((y_vrp_test > vrp_mean) == (vrp_pred > vrp_mean)).sum()
    binom_result = stats.binomtest(correct, n, 0.5, alternative='greater')
    results.append({
        'Test': 'Direction (Binomial)',
        'Statistic': f"{correct}/{n}",
        'p-value': f"{binom_result.pvalue:.6f}",
        'Conclusion': 'Significant' if binom_result.pvalue < 0.05 else 'Not Significant'
    })
    
    # 4. Normality test
    _, p_normal = stats.shapiro(errors[:500])  # Shapiro max 5000
    results.append({
        'Test': 'Normality (Shapiro-Wilk)',
        'Statistic': 'N/A',
        'p-value': f"{p_normal:.6f}",
        'Conclusion': 'Normal' if p_normal > 0.05 else 'Non-Normal'
    })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return results


def main():
    print("\n" + "📊" * 30)
    print("논문용 최종 테이블 생성")
    print("📊" * 30)
    
    # 데이터 로드
    print("\n데이터 로드...")
    spy = load_and_prepare_data()
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    all_tables = {}
    
    # 테이블 생성
    all_tables['table1_descriptive'] = table_1_descriptive_statistics(spy)
    all_tables['table2_correlation'] = table_2_correlation_matrix(spy)
    all_tables['table3_model_comparison'] = table_3_model_comparison(spy)
    all_tables['table4_category_best'] = table_4_category_best(all_tables['table3_model_comparison'])
    all_tables['table5_feature_importance'] = table_5_feature_importance(spy)
    all_tables['table6_robustness'] = table_6_robustness(spy)
    all_tables['table7_trading'] = table_7_trading_performance(spy)
    all_tables['table8_multi_asset'] = table_8_multi_asset()
    all_tables['table9_statistical'] = table_9_statistical_tests(spy)
    
    # 저장
    all_tables['timestamp'] = datetime.now().isoformat()
    
    with open('paper/final_tables.json', 'w') as f:
        json.dump(all_tables, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("📊 테이블 생성 완료")
    print("=" * 80)
    print(f"  ✓ 9개 테이블 생성")
    print(f"  ✓ 저장: paper/final_tables.json")


if __name__ == '__main__':
    main()
