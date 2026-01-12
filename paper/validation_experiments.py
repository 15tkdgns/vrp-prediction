#!/usr/bin/env python3
"""
논문용 추가 검증 실험
=====================

1. 표본 분할 안정성
2. VXX/UVXY 실제 수익
3. GARCH 모델 비교
4. 정권 전환 분석
5. 특성 상호작용
6. VRP Persistence 분석
7. VRP → 주식 수익 예측
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
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
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VRP
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
    # 미래 수익
    spy['return_future_22d'] = spy['returns'].rolling(22).sum().shift(-22)
    
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


def experiment_1_split_stability(spy):
    """실험 1: 표본 분할 안정성"""
    print("\n" + "=" * 60)
    print("[1/7] 표본 분할 안정성")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    splits = [0.6, 0.7, 0.8, 0.85, 0.9]
    seeds = [42, 123, 456, 789, 2024]
    
    results = []
    
    print(f"\n  {'분할비':>8} | {'시드':>6} | {'Test':>6} | {'R²':>8} | {'방향':>8}")
    print("  " + "-" * 50)
    
    for split in splits:
        for seed in seeds[:2]:  # 각 분할당 2개 시드
            np.random.seed(seed)
            
            # 시계열이므로 순차적 분할
            split_idx = int(len(spy_clean) * split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train = y[:split_idx]
            vix_test = vix[split_idx:]
            y_vrp_test = y_vrp[split_idx:]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed, max_iter=10000)
            en.fit(X_train_s, y_train)
            vrp_pred = vix_test - en.predict(X_test_s)
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            results.append({
                'split': split,
                'seed': seed,
                'test_size': len(X_test),
                'r2': float(r2),
                'direction_accuracy': float(dir_acc)
            })
            
            print(f"  {split:>8.0%} | {seed:>6} | {len(X_test):>6} | {r2:>8.4f} | {dir_acc*100:>7.1f}%")
    
    # 요약
    r2_values = [r['r2'] for r in results]
    print(f"\n  📊 요약:")
    print(f"     R² 범위: [{min(r2_values):.4f}, {max(r2_values):.4f}]")
    print(f"     R² 평균: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
    
    return results


def experiment_2_vxx_returns(spy):
    """실험 2: VXX/UVXY 실제 수익"""
    print("\n" + "=" * 60)
    print("[2/7] VXX/UVXY 실제 수익 시뮬레이션")
    print("=" * 60)
    
    # VXX 데이터 다운로드
    try:
        vxx = yf.download('VIXY', start='2020-01-01', end='2025-01-01', progress=False)
        if isinstance(vxx.columns, pd.MultiIndex):
            vxx.columns = vxx.columns.get_level_values(0)
        
        if len(vxx) < 100:
            print("  ⚠️ VXX 데이터 부족")
            return {'status': 'insufficient_data'}
        
        vxx['returns'] = vxx['Close'].pct_change()
        
        # SPY와 합치기
        spy_vxx = spy.copy()
        spy_vxx['VXX_returns'] = vxx['returns'].reindex(spy_vxx.index)
        spy_vxx = spy_vxx.dropna(subset=['VXX_returns', 'VRP_true'])
        
        print(f"  ✓ 데이터: {len(spy_vxx)} 행")
        
        # 전략: VRP > 평균이면 VXX 숏 (= 변동성 숏)
        vrp_mean = spy_vxx['VRP_true'].mean()
        
        strategies = {
            'VXX 숏 (VRP>평균)': -1 * (spy_vxx['VRP_true'] > vrp_mean).astype(int) * spy_vxx['VXX_returns'],
            'VXX 롱 (VRP<평균)': (spy_vxx['VRP_true'] < vrp_mean).astype(int) * spy_vxx['VXX_returns'],
            'VXX Buy&Hold 숏': -1 * spy_vxx['VXX_returns']
        }
        
        results = {}
        
        print(f"\n  {'전략':25s} | {'총수익':>10} | {'Sharpe':>8} | {'승률':>8}")
        print("  " + "-" * 60)
        
        for name, returns in strategies.items():
            total = returns.sum() * 100
            avg = returns.mean() * 100
            std = returns.std() * 100
            sharpe = avg / std * np.sqrt(252) if std > 0 else 0
            win_rate = (returns > 0).mean()
            
            results[name] = {
                'total_return': float(total),
                'sharpe': float(sharpe),
                'win_rate': float(win_rate)
            }
            
            print(f"  {name:25s} | {total:>9.2f}% | {sharpe:>8.2f} | {win_rate*100:>7.1f}%")
        
        return results
        
    except Exception as e:
        print(f"  ⚠️ VXX 오류: {e}")
        return {'status': 'error', 'message': str(e)}


def experiment_3_garch_comparison(spy):
    """실험 3: GARCH 모델 비교"""
    print("\n" + "=" * 60)
    print("[3/7] GARCH 모델 비교")
    print("=" * 60)
    
    try:
        from arch import arch_model
        HAS_ARCH = True
    except:
        HAS_ARCH = False
        print("  ⚠️ arch 패키지 없음 - 간단한 변동성 모델로 대체")
    
    spy_clean = spy.dropna(subset=['returns', 'RV_future', 'VRP_true'])
    returns = spy_clean['returns'].values * 100
    
    split_idx = int(len(spy_clean) * 0.8)
    vix_test = spy_clean['VIX'].values[split_idx:]
    y_vrp_test = spy_clean['VRP_true'].values[split_idx:]
    
    results = {}
    
    if HAS_ARCH:
        # GARCH(1,1)
        print("\n  🔹 GARCH(1,1)")
        try:
            garch = arch_model(returns[:split_idx], vol='Garch', p=1, q=1)
            garch_fit = garch.fit(disp='off')
            
            # 예측
            forecast = garch_fit.forecast(horizon=22)
            rv_pred_garch = np.sqrt(forecast.variance.values[-1, -1]) * np.sqrt(252)
            
            # 간단히 마지막 예측값으로
            vrp_pred = vix_test[:1] - rv_pred_garch
            print(f"     GARCH 조건부 변동성 예측: {rv_pred_garch:.2f}%")
            results['GARCH(1,1)'] = {'conditional_vol': float(rv_pred_garch)}
        except Exception as e:
            print(f"     오류: {e}")
    
    # 간단한 변동성 모델 (EWMA)
    print("\n  🔹 EWMA (Exponentially Weighted MA)")
    lambda_param = 0.94
    
    ewma_var = np.zeros(len(returns))
    ewma_var[0] = returns[0]**2
    
    for i in range(1, len(returns)):
        ewma_var[i] = lambda_param * ewma_var[i-1] + (1 - lambda_param) * returns[i-1]**2
    
    ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)
    
    rv_test = spy_clean['RV_22d'].values[split_idx:]
    ewma_test = ewma_vol[split_idx:]
    
    r2_ewma = r2_score(rv_test, ewma_test)
    mae_ewma = mean_absolute_error(rv_test, ewma_test)
    
    vrp_pred_ewma = vix_test - ewma_test
    vrp_r2 = r2_score(y_vrp_test, vrp_pred_ewma)
    
    results['EWMA'] = {
        'rv_r2': float(r2_ewma),
        'rv_mae': float(mae_ewma),
        'vrp_r2': float(vrp_r2)
    }
    
    print(f"     RV 예측 R²: {r2_ewma:.4f}")
    print(f"     VRP 예측 R²: {vrp_r2:.4f}")
    
    # ElasticNet 비교
    print("\n  🔹 ElasticNet (현재 모델)")
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_f = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    X = spy_f[feature_cols].values
    y = spy_f['RV_future'].values
    
    split_idx_f = int(len(spy_f) * 0.8)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx_f])
    X_test_s = scaler.transform(X[split_idx_f:])
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx_f])
    
    rv_pred_en = en.predict(X_test_s)
    vrp_pred_en = spy_f['VIX'].values[split_idx_f:] - rv_pred_en
    y_vrp_test_en = spy_f['VRP_true'].values[split_idx_f:]
    
    en_r2 = r2_score(y_vrp_test_en, vrp_pred_en)
    results['ElasticNet'] = {'vrp_r2': float(en_r2)}
    print(f"     VRP 예측 R²: {en_r2:.4f}")
    
    return results


def experiment_4_regime_analysis(spy):
    """실험 4: 정권 전환 분석"""
    print("\n" + "=" * 60)
    print("[4/7] 정권 전환 분석 (Bear/Bull)")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    # 시장 정권 정의
    spy_clean['cumret'] = (1 + spy_clean['returns']).cumprod()
    spy_clean['drawdown'] = spy_clean['cumret'] / spy_clean['cumret'].cummax() - 1
    
    regimes = {
        'Bull (DD > -5%)': spy_clean['drawdown'] > -0.05,
        'Correction (-5% ~ -10%)': (spy_clean['drawdown'] <= -0.05) & (spy_clean['drawdown'] > -0.10),
        'Bear (-10% ~ -20%)': (spy_clean['drawdown'] <= -0.10) & (spy_clean['drawdown'] > -0.20),
        'Crash (DD < -20%)': spy_clean['drawdown'] <= -0.20
    }
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
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
    
    results = {}
    
    print(f"\n  {'정권':25s} | {'샘플':>6} | {'R²':>8} | {'방향':>8}")
    print("  " + "-" * 55)
    
    for regime, mask in regimes.items():
        test_mask = mask.values[split_idx:]
        
        if test_mask.sum() >= 10:
            r2 = r2_score(y_vrp_test[test_mask], vrp_pred[test_mask])
            dir_acc = ((y_vrp_test[test_mask] > y_vrp_test.mean()) == 
                      (vrp_pred[test_mask] > y_vrp_test.mean())).mean()
            
            results[regime] = {
                'n_samples': int(test_mask.sum()),
                'r2': float(r2),
                'direction_accuracy': float(dir_acc)
            }
            
            print(f"  {regime:25s} | {test_mask.sum():>6} | {r2:>8.4f} | {dir_acc*100:>7.1f}%")
        else:
            print(f"  {regime:25s} | {test_mask.sum():>6} | 샘플 부족")
    
    return results


def experiment_5_feature_interaction(spy):
    """실험 5: 특성 상호작용"""
    print("\n" + "=" * 60)
    print("[5/7] 특성 상호작용")
    print("=" * 60)
    
    base_features = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                    'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                    'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=base_features + ['RV_future', 'VRP_true'])
    
    # 상호작용 특성 추가
    spy_clean['VIX_RV_ratio'] = spy_clean['VIX'] / (spy_clean['RV_22d'] + 1e-8)
    spy_clean['VIX_VRP_interact'] = spy_clean['VIX_lag1'] * spy_clean['VRP_lag1']
    spy_clean['RV_momentum'] = spy_clean['RV_5d'] - spy_clean['RV_22d']
    spy_clean['VIX_zscore'] = (spy_clean['VIX'] - spy_clean['VIX'].rolling(20).mean()) / (spy_clean['VIX'].rolling(20).std() + 1e-8)
    
    spy_clean = spy_clean.dropna()
    
    interaction_features = base_features + ['VIX_RV_ratio', 'VIX_VRP_interact', 'RV_momentum', 'VIX_zscore']
    
    X_base = spy_clean[base_features].values
    X_interact = spy_clean[interaction_features].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    split_idx = int(len(spy_clean) * 0.8)
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    results = {}
    
    # 기본 모델
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_base[:split_idx])
    X_test_s = scaler.transform(X_base[split_idx:])
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    vrp_pred = vix_test - en.predict(X_test_s)
    r2_base = r2_score(y_vrp_test, vrp_pred)
    
    print(f"\n  기본 모델 ({len(base_features)}개 특성): R² = {r2_base:.4f}")
    results['base'] = {'n_features': len(base_features), 'r2': float(r2_base)}
    
    # 상호작용 모델
    scaler2 = StandardScaler()
    X_train_i = scaler2.fit_transform(X_interact[:split_idx])
    X_test_i = scaler2.transform(X_interact[split_idx:])
    
    en2 = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en2.fit(X_train_i, y[:split_idx])
    vrp_pred2 = vix_test - en2.predict(X_test_i)
    r2_interact = r2_score(y_vrp_test, vrp_pred2)
    
    print(f"  상호작용 모델 ({len(interaction_features)}개 특성): R² = {r2_interact:.4f}")
    results['interaction'] = {'n_features': len(interaction_features), 'r2': float(r2_interact)}
    
    improvement = (r2_interact - r2_base) / abs(r2_base) * 100
    print(f"  개선: {improvement:+.1f}%")
    results['improvement'] = float(improvement)
    
    return results


def experiment_6_persistence(spy):
    """실험 6: VRP Persistence 분석"""
    print("\n" + "=" * 60)
    print("[6/7] VRP Persistence 분석")
    print("=" * 60)
    
    spy_clean = spy.dropna(subset=['VRP', 'VRP_true'])
    
    # 자기상관
    lags = [1, 2, 5, 10, 22, 44]
    
    print(f"\n  📊 VRP 자기상관:")
    print(f"  {'Lag':>6} | {'AC':>8} | {'설명':>15}")
    print("  " + "-" * 35)
    
    ac_results = {}
    for lag in lags:
        ac = spy_clean['VRP'].autocorr(lag=lag)
        ac_results[lag] = float(ac)
        
        desc = "매우 강함" if ac > 0.8 else "강함" if ac > 0.5 else "중간" if ac > 0.3 else "약함"
        bar = "█" * int(abs(ac) * 20)
        print(f"  {lag:>6} | {ac:>8.4f} | {desc:>15} {bar}")
    
    # AR(1) 계수 추정
    from sklearn.linear_model import LinearRegression
    
    vrp = spy_clean['VRP'].values
    X_ar = vrp[:-1].reshape(-1, 1)
    y_ar = vrp[1:]
    
    lr = LinearRegression()
    lr.fit(X_ar, y_ar)
    ar1_coef = lr.coef_[0]
    
    print(f"\n  📊 AR(1) 계수: {ar1_coef:.4f}")
    
    # 반감기
    if ar1_coef > 0 and ar1_coef < 1:
        half_life = -np.log(2) / np.log(ar1_coef)
        print(f"  📊 반감기: {half_life:.1f}일")
    else:
        half_life = None
    
    return {
        'autocorrelation': ac_results,
        'ar1_coefficient': float(ar1_coef),
        'half_life': float(half_life) if half_life else None
    }


def experiment_7_vrp_return_prediction(spy):
    """실험 7: VRP → 주식 수익 예측"""
    print("\n" + "=" * 60)
    print("[7/7] VRP → 주식 수익 예측")
    print("=" * 60)
    
    spy_clean = spy.dropna(subset=['VRP', 'VRP_true', 'return_future_22d'])
    
    # VRP가 미래 수익을 예측하는가?
    print("\n  📊 VRP와 미래 수익의 관계:")
    
    correlation = spy_clean['VRP'].corr(spy_clean['return_future_22d'])
    print(f"     VRP-미래수익(22일) 상관관계: {correlation:.4f}")
    
    # 회귀 분석
    X_vrp = spy_clean['VRP'].values.reshape(-1, 1)
    y_ret = spy_clean['return_future_22d'].values * 100
    
    split_idx = int(len(spy_clean) * 0.8)
    
    lr = LinearRegression()
    lr.fit(X_vrp[:split_idx], y_ret[:split_idx])
    
    y_pred = lr.predict(X_vrp[split_idx:])
    y_test = y_ret[split_idx:]
    
    r2 = r2_score(y_test, y_pred)
    
    print(f"     VRP → 수익 예측 R²: {r2:.4f}")
    print(f"     계수: {lr.coef_[0]:.4f} (VRP 1%↑ → 수익 {lr.coef_[0]:.4f}%↑)")
    
    # VRP 분위별 수익
    spy_clean['VRP_quantile'] = pd.qcut(spy_clean['VRP'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    quantile_returns = spy_clean.groupby('VRP_quantile')['return_future_22d'].mean() * 100
    
    print(f"\n  📊 VRP 분위별 미래 22일 수익:")
    print(f"  {'분위':>6} | {'평균 수익':>10}")
    print("  " + "-" * 20)
    
    quantile_results = {}
    for q, ret in quantile_returns.items():
        quantile_results[str(q)] = float(ret)
        print(f"  {q:>6} | {ret:>9.2f}%")
    
    return {
        'correlation': float(correlation),
        'r2': float(r2),
        'coefficient': float(lr.coef_[0]),
        'quantile_returns': quantile_results
    }


def main():
    print("\n" + "🔬" * 30)
    print("논문용 추가 검증 실험")
    print("🔬" * 30)
    
    # 데이터 로드
    print("\n데이터 로드...")
    spy = load_data()
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    results = {}
    
    # 실험 실행
    results['split_stability'] = experiment_1_split_stability(spy)
    results['vxx_returns'] = experiment_2_vxx_returns(spy)
    results['garch_comparison'] = experiment_3_garch_comparison(spy)
    results['regime_analysis'] = experiment_4_regime_analysis(spy)
    results['feature_interaction'] = experiment_5_feature_interaction(spy)
    results['persistence'] = experiment_6_persistence(spy)
    results['vrp_return_prediction'] = experiment_7_vrp_return_prediction(spy)
    
    # 저장
    results['timestamp'] = datetime.now().isoformat()
    
    with open('paper/validation_experiments.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 모든 검증 실험 완료")
    print("=" * 60)
    print(f"  ✓ paper/validation_experiments.json")


if __name__ == '__main__':
    main()
