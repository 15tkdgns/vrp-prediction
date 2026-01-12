#!/usr/bin/env python3
"""
SCI 출판용 보완 실험
====================

1. 딥러닝 모델 (LSTM, MLP)
2. 장기 데이터 분석 (2010-2025)
3. 다중 시장 분석 (유럽, 일본)
4. 앙상블 모델 (ML + 전통)
5. 시장 마이크로스트럭처 특성
6. 경제적 유의성 강화
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def prepare_data(ticker, vol_ticker, start='2010-01-01', end='2025-01-01'):
    """데이터 준비"""
    asset = yf.download(ticker, start=start, end=end, progress=False)
    vol = yf.download(vol_ticker, start=start, end=end, progress=False)
    
    if isinstance(asset.columns, pd.MultiIndex):
        asset.columns = asset.columns.get_level_values(0)
    if isinstance(vol.columns, pd.MultiIndex):
        vol.columns = vol.columns.get_level_values(0)
    
    asset['Vol'] = vol['Close'].reindex(asset.index).ffill().bfill()
    asset['returns'] = asset['Close'].pct_change()
    
    # 실현변동성
    asset['RV_1d'] = asset['returns'].abs() * np.sqrt(252) * 100
    asset['RV_5d'] = asset['returns'].rolling(5).std() * np.sqrt(252) * 100
    asset['RV_22d'] = asset['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VRP
    asset['VRP'] = asset['Vol'] - asset['RV_22d']
    asset['RV_future'] = asset['RV_22d'].shift(-22)
    asset['VRP_true'] = asset['Vol'] - asset['RV_future']
    
    # 특성
    asset['Vol_lag1'] = asset['Vol'].shift(1)
    asset['Vol_lag5'] = asset['Vol'].shift(5)
    asset['Vol_change'] = asset['Vol'].pct_change()
    asset['VRP_lag1'] = asset['VRP'].shift(1)
    asset['VRP_lag5'] = asset['VRP'].shift(5)
    asset['VRP_ma5'] = asset['VRP'].rolling(5).mean()
    asset['regime_high'] = (asset['Vol'] >= 25).astype(int)
    asset['return_5d'] = asset['returns'].rolling(5).sum()
    asset['return_22d'] = asset['returns'].rolling(22).sum()
    
    # 추가 특성
    asset['Vol_ma5'] = asset['Vol'].rolling(5).mean()
    asset['Vol_ma22'] = asset['Vol'].rolling(22).mean()
    asset['Vol_std5'] = asset['Vol'].rolling(5).std()
    asset['RV_ratio'] = asset['RV_5d'] / (asset['RV_22d'] + 1e-8)
    
    asset = asset.replace([np.inf, -np.inf], np.nan).dropna()
    
    return asset


def experiment_1_deep_learning():
    """보완 1: 딥러닝 모델"""
    print("\n" + "=" * 70)
    print("[1/6] 딥러닝 모델 (MLP Neural Network)")
    print("=" * 70)
    
    spy = prepare_data('SPY', '^VIX', '2015-01-01', '2025-01-01')
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5', 
                   'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d',
                   'Vol_ma5', 'Vol_ma22', 'Vol_std5', 'RV_ratio']
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vol = spy['Vol'].values
    y_vrp = spy['VRP_true'].values
    
    split_idx = int(len(spy) * 0.8)
    vix_test = vol[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    models = {
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000),
        'MLP (64,32)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, 
                                     random_state=SEED, early_stopping=True),
        'MLP (128,64,32)': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                         random_state=SEED, early_stopping=True),
        'MLP (256,128,64)': MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                                          random_state=SEED, early_stopping=True)
    }
    
    results = {}
    
    print(f"\n  {'Model':25s} | {'R²':>10} | {'MAE':>10} | {'방향':>10}")
    print("  " + "-" * 65)
    
    for name, model in models.items():
        try:
            model.fit(X_train_s, y[:split_idx])
            vrp_pred = vix_test - model.predict(X_test_s)
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            mae = mean_absolute_error(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            results[name] = {
                'r2': float(r2),
                'mae': float(mae),
                'direction_accuracy': float(dir_acc)
            }
            
            print(f"  {name:25s} | {r2:>10.4f} | {mae:>10.4f} | {dir_acc*100:>9.1f}%")
        except Exception as e:
            print(f"  {name:25s} | 오류: {str(e)[:30]}")
    
    return results


def experiment_2_long_term():
    """보완 2: 장기 데이터 분석 (2010-2025)"""
    print("\n" + "=" * 70)
    print("[2/6] 장기 데이터 분석 (2010-2025)")
    print("=" * 70)
    
    spy = prepare_data('SPY', '^VIX', '2010-01-01', '2025-01-01')
    
    print(f"\n  📊 데이터: {len(spy)} 거래일 ({spy.index[0].date()} ~ {spy.index[-1].date()})")
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5', 
                   'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vol = spy['Vol'].values
    y_vrp = spy['VRP_true'].values
    dates = spy.index
    
    # 롤링 윈도우 평가
    window = 504  # 2년
    step = 252    # 1년씩 이동
    
    results = []
    
    print(f"\n  📊 롤링 윈도우 평가 (2년 학습, 1년 테스트):")
    print(f"  {'기간':25s} | {'R²':>10} | {'방향':>10}")
    print("  " + "-" * 50)
    
    for start in range(0, len(X) - window - step, step):
        train_end = start + window
        test_end = min(train_end + step, len(X))
        
        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        vix_test = vol[train_end:test_end]
        y_vrp_test = y_vrp[train_end:test_end]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        vrp_pred = vix_test - en.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        period = f"{dates[train_end].strftime('%Y')}-{dates[test_end-1].strftime('%Y')}"
        
        results.append({
            'period': period,
            'r2': float(r2),
            'direction_accuracy': float(dir_acc)
        })
        
        print(f"  {period:25s} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
    
    # 요약
    r2_values = [r['r2'] for r in results]
    dir_values = [r['direction_accuracy'] for r in results]
    
    print(f"\n  📊 장기 성능 요약:")
    print(f"     R² 평균: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
    print(f"     방향 평균: {np.mean(dir_values)*100:.1f}%")
    print(f"     양수 R² 비율: {sum(1 for r in r2_values if r > 0)}/{len(r2_values)}")
    
    return {
        'periods': results,
        'mean_r2': float(np.mean(r2_values)),
        'std_r2': float(np.std(r2_values)),
        'positive_ratio': sum(1 for r in r2_values if r > 0) / len(r2_values)
    }


def experiment_3_multi_market():
    """보완 3: 다중 시장 분석"""
    print("\n" + "=" * 70)
    print("[3/6] 다중 시장 분석")
    print("=" * 70)
    
    markets = [
        ('SPY (S&P 500)', 'SPY', '^VIX'),
        ('EFA (EAFE)', 'EFA', '^VIX'),
        ('EEM (Emerging)', 'EEM', '^VIX'),
        ('GLD (Gold)', 'GLD', '^VIX'),
    ]
    
    results = {}
    
    print(f"\n  {'Market':25s} | {'샘플':>8} | {'R²':>10} | {'방향':>10}")
    print("  " + "-" * 60)
    
    for name, ticker, vol_ticker in markets:
        try:
            data = prepare_data(ticker, vol_ticker, '2015-01-01', '2025-01-01')
            
            if len(data) < 500:
                print(f"  {name:25s} | 데이터 부족")
                continue
            
            feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5', 
                           'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                           'regime_high', 'return_5d', 'return_22d']
            
            X = data[feature_cols].values
            y = data['RV_future'].values
            vol = data['Vol'].values
            y_vrp = data['VRP_true'].values
            
            split_idx = int(len(data) * 0.8)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X[:split_idx])
            X_test_s = scaler.transform(X[split_idx:])
            
            en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
            en.fit(X_train_s, y[:split_idx])
            vrp_pred = vol[split_idx:] - en.predict(X_test_s)
            y_vrp_test = y_vrp[split_idx:]
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            results[name] = {
                'n_samples': len(data),
                'r2': float(r2),
                'direction_accuracy': float(dir_acc)
            }
            
            print(f"  {name:25s} | {len(data):>8} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
            
        except Exception as e:
            print(f"  {name:25s} | 오류: {str(e)[:30]}")
    
    return results


def experiment_4_ensemble():
    """보완 4: 앙상블 모델"""
    print("\n" + "=" * 70)
    print("[4/6] 앙상블 모델 (ML + 전통)")
    print("=" * 70)
    
    spy = prepare_data('SPY', '^VIX', '2015-01-01', '2025-01-01')
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5', 
                   'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vol = spy['Vol'].values
    y_vrp = spy['VRP_true'].values
    
    split_idx = int(len(spy) * 0.8)
    vix_test = vol[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    # 개별 모델
    en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
    ridge = Ridge(alpha=1.0, random_state=SEED)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED, early_stopping=True)
    
    en.fit(X_train_s, y[:split_idx])
    ridge.fit(X_train_s, y[:split_idx])
    mlp.fit(X_train_s, y[:split_idx])
    
    pred_en = en.predict(X_test_s)
    pred_ridge = ridge.predict(X_test_s)
    pred_mlp = mlp.predict(X_test_s)
    
    # 앙상블 조합
    ensembles = {
        'ElasticNet Only': pred_en,
        'Ridge Only': pred_ridge,
        'MLP Only': pred_mlp,
        'Simple Average': (pred_en + pred_ridge + pred_mlp) / 3,
        'Weighted (EN 50%, Ridge 30%, MLP 20%)': 0.5*pred_en + 0.3*pred_ridge + 0.2*pred_mlp,
        'Weighted (EN 60%, MLP 40%)': 0.6*pred_en + 0.4*pred_mlp
    }
    
    results = {}
    
    print(f"\n  {'Ensemble':40s} | {'R²':>10} | {'방향':>10}")
    print("  " + "-" * 65)
    
    for name, pred in ensembles.items():
        vrp_pred = vix_test - pred
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results[name] = {
            'r2': float(r2),
            'direction_accuracy': float(dir_acc)
        }
        
        print(f"  {name:40s} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
    
    return results


def experiment_5_microstructure():
    """보완 5: 시장 마이크로스트럭처"""
    print("\n" + "=" * 70)
    print("[5/6] 시장 마이크로스트럭처 특성")
    print("=" * 70)
    
    spy = prepare_data('SPY', '^VIX', '2015-01-01', '2025-01-01')
    
    # 추가 마이크로스트럭처 특성
    spy['Volume_ma5'] = spy['Volume'].rolling(5).mean()
    spy['Volume_ratio'] = spy['Volume'] / spy['Volume_ma5']
    spy['High_Low_range'] = (spy['High'] - spy['Low']) / spy['Close'] * 100
    spy['Close_Open'] = (spy['Close'] - spy['Open']) / spy['Open'] * 100
    spy['Garman_Klass'] = 0.5 * np.log(spy['High']/spy['Low'])**2 - (2*np.log(2)-1)*np.log(spy['Close']/spy['Open'])**2
    
    spy = spy.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 기본 특성
    base_features = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5', 
                    'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                    'regime_high', 'return_5d', 'return_22d']
    
    # 확장 특성
    extended_features = base_features + ['Volume_ratio', 'High_Low_range', 'Close_Open', 'Garman_Klass']
    
    y = spy['RV_future'].values
    vol = spy['Vol'].values
    y_vrp = spy['VRP_true'].values
    
    split_idx = int(len(spy) * 0.8)
    vix_test = vol[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    results = {}
    
    for name, features in [('기본 특성', base_features), ('확장 특성 (마이크로)', extended_features)]:
        X = spy[features].values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[:split_idx])
        X_test_s = scaler.transform(X[split_idx:])
        
        en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y[:split_idx])
        vrp_pred = vix_test - en.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        
        results[name] = {'n_features': len(features), 'r2': float(r2)}
    
    print(f"\n  📊 마이크로스트럭처 특성 효과:")
    for name, r in results.items():
        print(f"     {name}: {r['n_features']}개 특성 → R² = {r['r2']:.4f}")
    
    improvement = (results['확장 특성 (마이크로)']['r2'] - results['기본 특성']['r2']) / abs(results['기본 특성']['r2']) * 100
    print(f"\n  💡 개선율: {improvement:+.1f}%")
    
    return results


def experiment_6_economic_significance():
    """보완 6: 경제적 유의성 강화"""
    print("\n" + "=" * 70)
    print("[6/6] 경제적 유의성 강화")
    print("=" * 70)
    
    spy = prepare_data('SPY', '^VIX', '2015-01-01', '2025-01-01')
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5', 
                   'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = spy[feature_cols].values
    y = spy['RV_future'].values
    vol = spy['Vol'].values
    y_vrp = spy['VRP_true'].values
    
    split_idx = int(len(spy) * 0.8)
    vix_test = vol[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    vrp_pred = vix_test - en.predict(X_test_s)
    
    vrp_mean = y_vrp_test.mean()
    positions = (vrp_pred > vrp_mean).astype(int)
    
    # 전략 수익
    returns = positions * y_vrp_test
    
    # 연율화 수익률
    n_years = len(returns) / 252
    total_return = returns.sum()
    annual_return = total_return / n_years
    
    # Sharpe Ratio (연율화)
    daily_returns = returns[positions == 1]
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    
    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    sortino = daily_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    # Calmar Ratio
    cum_returns = pd.Series(returns).cumsum()
    max_dd = (cum_returns.cummax() - cum_returns).max()
    calmar = annual_return / max_dd if max_dd > 0 else 0
    
    # Information Ratio (vs Buy&Hold)
    bh_returns = y_vrp_test
    excess_returns = returns - bh_returns
    ir = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    print(f"\n  📊 경제적 성과 지표:")
    print(f"     총 수익: {total_return:.2f}%")
    print(f"     연율화 수익: {annual_return:.2f}%")
    print(f"     Sharpe Ratio: {sharpe:.2f}")
    print(f"     Sortino Ratio: {sortino:.2f}")
    print(f"     Calmar Ratio: {calmar:.2f}")
    print(f"     Information Ratio: {ir:.2f}")
    
    # 통계적 유의성 (Bootstrap)
    n_bootstrap = 1000
    bootstrap_sharpe = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(daily_returns), len(daily_returns), replace=True)
        boot_returns = np.array(daily_returns)[idx]
        boot_sharpe = boot_returns.mean() / boot_returns.std() * np.sqrt(252)
        bootstrap_sharpe.append(boot_sharpe)
    
    sharpe_ci = np.percentile(bootstrap_sharpe, [2.5, 97.5])
    
    print(f"\n  📊 Sharpe Ratio 95% CI: [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]")
    print(f"     유의성: {'유의함 (0 미포함)' if sharpe_ci[0] > 0 else '유의하지 않음'}")
    
    return {
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'calmar': float(calmar),
        'information_ratio': float(ir),
        'sharpe_ci': [float(sharpe_ci[0]), float(sharpe_ci[1])]
    }


def main():
    print("\n" + "🎯" * 30)
    print("SCI 출판용 보완 실험")
    print("🎯" * 30)
    
    results = {}
    
    results['deep_learning'] = experiment_1_deep_learning()
    results['long_term'] = experiment_2_long_term()
    results['multi_market'] = experiment_3_multi_market()
    results['ensemble'] = experiment_4_ensemble()
    results['microstructure'] = experiment_5_microstructure()
    results['economic_significance'] = experiment_6_economic_significance()
    
    # 저장
    results['timestamp'] = datetime.now().isoformat()
    
    with open('paper/sci_enhancements.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 요약
    print("\n" + "=" * 70)
    print("📊 SCI 보완 실험 요약")
    print("=" * 70)
    
    print("""
    ✅ 딥러닝: MLP 모델 추가 (ElasticNet과 비교 가능)
    ✅ 장기 데이터: 2010-2025 롤링 윈도우 분석
    ✅ 다중 시장: SPY, EFA, EEM, GLD 분석
    ✅ 앙상블: ML + 전통 모델 조합
    ✅ 마이크로스트럭처: 거래량, 변동폭 특성 추가
    ✅ 경제적 유의성: Sharpe, Sortino, Calmar, IR
    """)
    
    print(f"\n💾 결과 저장: paper/sci_enhancements.json")


if __name__ == '__main__':
    main()
