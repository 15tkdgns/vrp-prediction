#!/usr/bin/env python3
"""
논문용 추가 실험 및 데이터 수집
================================

1. 모델 상세 비교 (통계적 유의성)
2. 시계열 Cross-Validation
3. Diebold-Mariano 검정
4. 예측 오차 분석
5. 예측 vs 실제 시각화 데이터
6. 경제적 유의성 (Sharpe Ratio)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
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
    spy = spy.dropna()
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    return spy, feature_cols


def experiment_1_model_comparison(spy, feature_cols):
    """실험 1: 상세 모델 비교 (통계 검정 포함)"""
    print("\n" + "=" * 60)
    print("[1/6] 상세 모델 비교")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vix = spy['VIX'].values
    y_vrp = spy['VRP_true'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {
        'HAR-RV': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=SEED),
        'Lasso': Lasso(alpha=0.01, random_state=SEED, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, 
                                                       learning_rate=0.05, random_state=SEED)
    }
    
    results = {}
    predictions = {}
    errors = {}
    
    # HAR-RV는 다른 특성 사용
    har_features = ['RV_1d', 'RV_5d', 'RV_22d']
    har_idx = [feature_cols.index(f) for f in har_features]
    
    for name, model in models.items():
        if name == 'HAR-RV':
            X_tr = X_train_s[:, har_idx]
            X_te = X_test_s[:, har_idx]
        else:
            X_tr = X_train_s
            X_te = X_test_s
        
        model.fit(X_tr, y_train)
        rv_pred = model.predict(X_te)
        vrp_pred = vix_test - rv_pred
        
        predictions[name] = vrp_pred
        errors[name] = y_vrp_test - vrp_pred
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        rmse = np.sqrt(mean_squared_error(y_vrp_test, vrp_pred))
        mae = mean_absolute_error(y_vrp_test, vrp_pred)
        
        # 방향 정확도
        vrp_mean = y_vrp_test.mean()
        dir_acc = ((y_vrp_test > vrp_mean) == (vrp_pred > vrp_mean)).mean()
        
        results[name] = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'direction_accuracy': float(dir_acc)
        }
    
    print(f"\n  {'Model':<20} | {'R²':>8} | {'RMSE':>8} | {'MAE':>8} | {'방향':>8}")
    print("  " + "-" * 60)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
        print(f"  {name:<20} | {metrics['r2']:>8.4f} | {metrics['rmse']:>8.2f} | "
              f"{metrics['mae']:>8.2f} | {metrics['direction_accuracy']*100:>7.1f}%")
    
    return results, predictions, errors, y_vrp_test


def experiment_2_time_series_cv(spy, feature_cols):
    """실험 2: 시계열 Cross-Validation"""
    print("\n" + "=" * 60)
    print("[2/6] 시계열 Cross-Validation (5-Fold)")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vix = spy['VIX'].values
    y_vrp = spy['VRP_true'].values
    
    n_splits = 5
    fold_size = len(X) // (n_splits + 1)
    
    cv_results = []
    
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_end = train_end + fold_size
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        vix_test = vix[train_end:test_end]
        y_vrp_test = y_vrp[train_end:test_end]
        
        if len(X_test) < 50:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        rv_pred = en.predict(X_test_s)
        vrp_pred = vix_test - rv_pred
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        cv_results.append({
            'fold': i + 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'r2': float(r2),
            'direction_accuracy': float(dir_acc)
        })
        
        print(f"  Fold {i+1}: Train={len(X_train):4d}, Test={len(X_test):3d}, "
              f"R²={r2:.4f}, 방향={dir_acc*100:.1f}%")
    
    avg_r2 = np.mean([r['r2'] for r in cv_results])
    std_r2 = np.std([r['r2'] for r in cv_results])
    print(f"\n  평균 R²: {avg_r2:.4f} ± {std_r2:.4f}")
    
    return cv_results


def experiment_3_diebold_mariano(predictions, errors, y_vrp_test):
    """실험 3: Diebold-Mariano 검정"""
    print("\n" + "=" * 60)
    print("[3/6] Diebold-Mariano 검정 (ElasticNet vs 다른 모델)")
    print("=" * 60)
    
    base_model = 'ElasticNet'
    base_errors = errors[base_model]
    
    dm_results = {}
    
    for name, error in errors.items():
        if name == base_model:
            continue
        
        # MSE 차이
        d = base_errors**2 - error**2
        n = len(d)
        
        # 평균과 분산
        d_mean = d.mean()
        d_var = d.var()
        
        # DM 통계량
        dm_stat = d_mean / np.sqrt(d_var / n)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        better = "ElasticNet" if dm_stat < 0 else name
        significant = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
        
        dm_results[name] = {
            'dm_statistic': float(dm_stat),
            'p_value': float(p_value),
            'better_model': better
        }
        
        print(f"  ElasticNet vs {name:<15}: DM={dm_stat:>7.3f}, p={p_value:.4f} {significant}")
    
    return dm_results


def experiment_4_error_analysis(errors, y_vrp_test, spy):
    """실험 4: 예측 오차 분석"""
    print("\n" + "=" * 60)
    print("[4/6] 예측 오차 분석")
    print("=" * 60)
    
    error = errors['ElasticNet']
    
    split_idx = int(len(spy) * 0.8)
    vix_test = spy['VIX'].values[split_idx:]
    
    # 오류 통계
    print(f"\n  📊 ElasticNet 오차 통계:")
    print(f"     평균 오차 (ME):     {error.mean():>8.4f}")
    print(f"     평균 절대 오차:     {np.abs(error).mean():>8.4f}")
    print(f"     오차 표준편차:      {error.std():>8.4f}")
    print(f"     최대 과대예측:      {error.min():>8.4f}")
    print(f"     최대 과소예측:      {error.max():>8.4f}")
    
    # Regime별 오차
    print(f"\n  📊 Regime별 오차 (MAE):")
    
    regimes = {
        'Low Vol (VIX<20)': vix_test < 20,
        'Normal (20≤VIX<25)': (vix_test >= 20) & (vix_test < 25),
        'High Vol (VIX≥25)': vix_test >= 25
    }
    
    regime_errors = {}
    for regime, mask in regimes.items():
        if mask.sum() >= 10:
            mae = np.abs(error[mask]).mean()
            regime_errors[regime] = float(mae)
            print(f"     {regime:<20}: MAE = {mae:>6.2f}")
    
    return {
        'mean_error': float(error.mean()),
        'mae': float(np.abs(error).mean()),
        'std_error': float(error.std()),
        'min_error': float(error.min()),
        'max_error': float(error.max()),
        'regime_errors': regime_errors
    }


def experiment_5_prediction_data(predictions, y_vrp_test, spy):
    """실험 5: 예측 vs 실제 데이터 (시각화용)"""
    print("\n" + "=" * 60)
    print("[5/6] 예측 vs 실제 데이터")
    print("=" * 60)
    
    split_idx = int(len(spy) * 0.8)
    dates = spy.index[split_idx:].strftime('%Y-%m-%d').tolist()
    
    pred_data = {
        'dates': dates,
        'actual': y_vrp_test.tolist(),
        'predicted': predictions['ElasticNet'].tolist()
    }
    
    print(f"  ✓ 저장된 데이터 포인트: {len(dates)}개")
    print(f"  ✓ 기간: {dates[0]} ~ {dates[-1]}")
    
    # 20일 이동 평균
    actual_ma = pd.Series(y_vrp_test).rolling(20).mean().tolist()
    pred_ma = pd.Series(predictions['ElasticNet']).rolling(20).mean().tolist()
    
    pred_data['actual_ma20'] = actual_ma
    pred_data['predicted_ma20'] = pred_ma
    
    return pred_data


def experiment_6_economic_significance(predictions, y_vrp_test):
    """실험 6: 경제적 유의성 (Sharpe Ratio)"""
    print("\n" + "=" * 60)
    print("[6/6] 경제적 유의성")
    print("=" * 60)
    
    vrp_mean = y_vrp_test.mean()
    
    strategies = {
        'Buy & Hold': np.ones(len(y_vrp_test)),  # 항상 매도
        'ElasticNet': (predictions['ElasticNet'] > vrp_mean).astype(int),
        'HAR-RV': (predictions['HAR-RV'] > vrp_mean).astype(int),
        'GradientBoosting': (predictions['GradientBoosting'] > vrp_mean).astype(int)
    }
    
    results = {}
    
    print(f"\n  {'Strategy':<20} | {'수익률':>10} | {'표준편차':>10} | {'Sharpe':>8} | {'승률':>8}")
    print("  " + "-" * 65)
    
    for name, positions in strategies.items():
        returns = positions * y_vrp_test
        
        total_return = returns.sum()
        avg_return = returns[positions == 1].mean() if positions.sum() > 0 else 0
        std_return = returns[positions == 1].std() if positions.sum() > 1 else 0
        sharpe = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        win_rate = (returns[positions == 1] > 0).mean() if positions.sum() > 0 else 0
        
        results[name] = {
            'total_return': float(total_return),
            'avg_return': float(avg_return),
            'std_return': float(std_return),
            'sharpe_ratio': float(sharpe),
            'win_rate': float(win_rate),
            'n_trades': int(positions.sum())
        }
        
        print(f"  {name:<20} | {avg_return:>9.2f}% | {std_return:>9.2f}% | "
              f"{sharpe:>8.2f} | {win_rate*100:>7.1f}%")
    
    return results


def main():
    print("\n" + "📊" * 30)
    print("논문용 추가 실험 및 데이터 수집")
    print("📊" * 30)
    
    # 데이터 로드
    print("\n데이터 로드...")
    spy, feature_cols = load_data()
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    # 실험 실행
    model_results, predictions, errors, y_vrp_test = experiment_1_model_comparison(spy, feature_cols)
    cv_results = experiment_2_time_series_cv(spy, feature_cols)
    dm_results = experiment_3_diebold_mariano(predictions, errors, y_vrp_test)
    error_analysis = experiment_4_error_analysis(errors, y_vrp_test, spy)
    pred_data = experiment_5_prediction_data(predictions, y_vrp_test, spy)
    economic_results = experiment_6_economic_significance(predictions, y_vrp_test)
    
    # 결과 저장
    print("\n" + "=" * 60)
    print("📊 결과 저장")
    print("=" * 60)
    
    output = {
        'model_comparison': model_results,
        'cv_results': cv_results,
        'dm_test': dm_results,
        'error_analysis': error_analysis,
        'economic_significance': economic_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/additional_experiments.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    with open('paper/prediction_data.json', 'w') as f:
        json.dump(pred_data, f)
    
    print(f"  ✓ paper/additional_experiments.json")
    print(f"  ✓ paper/prediction_data.json")
    
    # 요약
    print("\n" + "=" * 60)
    print("📊 실험 요약")
    print("=" * 60)
    
    best_model = max(model_results.items(), key=lambda x: x[1]['r2'])
    avg_cv_r2 = np.mean([r['r2'] for r in cv_results])
    best_strategy = max(economic_results.items(), key=lambda x: x[1]['sharpe_ratio'] if x[0] != 'Buy & Hold' else -999)
    
    print(f"""
    🏆 최고 모델: {best_model[0]}
       R² = {best_model[1]['r2']:.4f}
       방향 정확도 = {best_model[1]['direction_accuracy']*100:.1f}%
    
    📊 5-Fold CV 평균 R²: {avg_cv_r2:.4f}
    
    💰 최고 전략: {best_strategy[0]}
       Sharpe Ratio = {best_strategy[1]['sharpe_ratio']:.2f}
       승률 = {best_strategy[1]['win_rate']*100:.1f}%
    """)


if __name__ == '__main__':
    main()
