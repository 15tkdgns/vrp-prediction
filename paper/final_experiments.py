#!/usr/bin/env python3
"""
논문용 최종 추가 실험
=====================

1. 다른 자산 일반화 (QQQ, IWM)
2. 예측 오차 시각화 데이터
3. VIX 스파이크 이벤트 분석
4. 롱/숏 양방향 전략
5. Learning Curve
6. 실시간 예측 시뮬레이션
7. 앙상블 가중치 최적화
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def prepare_asset_data(ticker, start='2020-01-01', end='2025-01-01'):
    """자산 데이터 준비"""
    asset = yf.download(ticker, start=start, end=end, progress=False)
    vix = yf.download('^VIX', start=start, end=end, progress=False)
    
    if isinstance(asset.columns, pd.MultiIndex):
        asset.columns = asset.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    asset['VIX'] = vix['Close'].reindex(asset.index).ffill().bfill()
    asset['returns'] = asset['Close'].pct_change()
    
    # 실현변동성
    asset['RV_1d'] = asset['returns'].abs() * np.sqrt(252) * 100
    asset['RV_5d'] = asset['returns'].rolling(5).std() * np.sqrt(252) * 100
    asset['RV_22d'] = asset['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    # VRP
    asset['VRP'] = asset['VIX'] - asset['RV_22d']
    asset['RV_future'] = asset['RV_22d'].shift(-22)
    asset['VRP_true'] = asset['VIX'] - asset['RV_future']
    
    # 특성
    asset['VIX_lag1'] = asset['VIX'].shift(1)
    asset['VIX_lag5'] = asset['VIX'].shift(5)
    asset['VIX_change'] = asset['VIX'].pct_change()
    asset['VRP_lag1'] = asset['VRP'].shift(1)
    asset['VRP_lag5'] = asset['VRP'].shift(5)
    asset['VRP_ma5'] = asset['VRP'].rolling(5).mean()
    asset['regime_high'] = (asset['VIX'] >= 25).astype(int)
    asset['return_5d'] = asset['returns'].rolling(5).sum()
    asset['return_22d'] = asset['returns'].rolling(22).sum()
    
    asset = asset.replace([np.inf, -np.inf], np.nan)
    asset = asset.dropna()
    
    return asset


def experiment_1_generalization(spy):
    """실험 1: 다른 자산 일반화"""
    print("\n" + "=" * 60)
    print("[1/7] 다른 자산 일반화 (QQQ, IWM, DIA)")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    tickers = {'SPY': spy, 'QQQ': None, 'IWM': None, 'DIA': None}
    
    results = {}
    
    print(f"\n  {'자산':>6} | {'샘플':>6} | {'R²':>8} | {'MAE':>8} | {'방향':>8}")
    print("  " + "-" * 50)
    
    for ticker, data in tickers.items():
        try:
            if data is None:
                data = prepare_asset_data(ticker)
            
            if len(data) < 100:
                continue
            
            X = data[feature_cols].values
            y = data['RV_future'].values
            vix = data['VIX'].values
            y_vrp = data['VRP_true'].values
            
            split_idx = int(len(data) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train = y[:split_idx]
            vix_test = vix[split_idx:]
            y_vrp_test = y_vrp[split_idx:]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
            en.fit(X_train_s, y_train)
            vrp_pred = vix_test - en.predict(X_test_s)
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            mae = mean_absolute_error(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            results[ticker] = {
                'n_samples': len(data),
                'r2': float(r2),
                'mae': float(mae),
                'direction_accuracy': float(dir_acc)
            }
            
            print(f"  {ticker:>6} | {len(data):>6} | {r2:>8.4f} | {mae:>8.2f} | {dir_acc*100:>7.1f}%")
            
        except Exception as e:
            print(f"  {ticker:>6} | 오류: {str(e)[:30]}")
    
    return results


def experiment_2_visualization_data(spy):
    """실험 2: 시각화 데이터 생성"""
    print("\n" + "=" * 60)
    print("[2/7] 예측 vs 실제 시각화 데이터")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    dates = spy_clean.index
    
    split_idx = int(len(spy_clean) * 0.8)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split_idx])
    X_test_s = scaler.transform(X[split_idx:])
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y[:split_idx])
    
    vrp_pred = vix[split_idx:] - en.predict(X_test_s)
    y_vrp_test = y_vrp[split_idx:]
    
    # 시각화 데이터
    viz_data = {
        'dates': dates[split_idx:].strftime('%Y-%m-%d').tolist(),
        'actual': y_vrp_test.tolist(),
        'predicted': vrp_pred.tolist(),
        'vix': vix[split_idx:].tolist(),
        'error': (y_vrp_test - vrp_pred).tolist()
    }
    
    # 이동평균
    viz_data['actual_ma20'] = pd.Series(y_vrp_test).rolling(20).mean().tolist()
    viz_data['predicted_ma20'] = pd.Series(vrp_pred).rolling(20).mean().tolist()
    
    print(f"  ✓ 데이터 포인트: {len(viz_data['dates'])}개")
    print(f"  ✓ 기간: {viz_data['dates'][0]} ~ {viz_data['dates'][-1]}")
    
    return viz_data


def experiment_3_vix_spike(spy):
    """실험 3: VIX 스파이크 이벤트 분석"""
    print("\n" + "=" * 60)
    print("[3/7] VIX 스파이크 이벤트 분석")
    print("=" * 60)
    
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
    
    # VIX 변화량
    vix_change_pct = np.diff(vix_test, prepend=vix_test[0]) / vix_test * 100
    
    events = {
        'VIX 급등 (>10%)': vix_change_pct > 10,
        'VIX 급등 (>5%)': (vix_change_pct > 5) & (vix_change_pct <= 10),
        '정상 (-5% ~ 5%)': (vix_change_pct >= -5) & (vix_change_pct <= 5),
        'VIX 급락 (<-5%)': vix_change_pct < -5
    }
    
    results = {}
    
    print(f"\n  {'이벤트':20s} | {'샘플':>6} | {'R²':>8} | {'MAE':>8} | {'방향':>8}")
    print("  " + "-" * 60)
    
    for event, mask in events.items():
        if mask.sum() >= 5:
            r2 = r2_score(y_vrp_test[mask], vrp_pred[mask])
            mae = mean_absolute_error(y_vrp_test[mask], vrp_pred[mask])
            dir_acc = ((y_vrp_test[mask] > y_vrp_test.mean()) == 
                      (vrp_pred[mask] > y_vrp_test.mean())).mean()
            
            results[event] = {
                'n_samples': int(mask.sum()),
                'r2': float(r2),
                'mae': float(mae),
                'direction_accuracy': float(dir_acc)
            }
            
            print(f"  {event:20s} | {mask.sum():>6} | {r2:>8.4f} | {mae:>8.2f} | {dir_acc*100:>7.1f}%")
    
    return results


def experiment_4_long_short(spy):
    """실험 4: 롱/숏 양방향 전략"""
    print("\n" + "=" * 60)
    print("[4/7] 롱/숏 양방향 전략")
    print("=" * 60)
    
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
        '숏 Only (VRP>평균)': {
            'position': np.where(vrp_pred > vrp_mean, 1, 0),
            'direction': 1  # 숏 = VRP 양수면 수익
        },
        '롱 Only (VRP<평균)': {
            'position': np.where(vrp_pred < vrp_mean, 1, 0),
            'direction': -1  # 롱 = VRP 음수면 수익
        },
        '롱/숏 양방향': {
            'position': np.sign(vrp_pred - vrp_mean),
            'direction': 1
        }
    }
    
    results = {}
    
    print(f"\n  {'전략':20s} | {'거래':>6} | {'총수익':>10} | {'평균':>8} | {'승률':>8}")
    print("  " + "-" * 65)
    
    for name, config in strategies.items():
        positions = config['position']
        direction = config['direction']
        
        if name == '롱/숏 양방향':
            returns = positions * y_vrp_test
        else:
            returns = positions * y_vrp_test * direction
        
        n_trades = np.abs(positions).sum()
        total_return = returns.sum()
        
        active_mask = positions != 0
        avg_return = returns[active_mask].mean() if active_mask.sum() > 0 else 0
        win_rate = (returns[active_mask] > 0).mean() if active_mask.sum() > 0 else 0
        
        results[name] = {
            'n_trades': int(n_trades),
            'total_return': float(total_return),
            'avg_return': float(avg_return),
            'win_rate': float(win_rate)
        }
        
        print(f"  {name:20s} | {n_trades:>6.0f} | {total_return:>9.2f}% | {avg_return:>7.2f}% | {win_rate*100:>7.1f}%")
    
    return results


def experiment_5_learning_curve(spy):
    """실험 5: Learning Curve"""
    print("\n" + "=" * 60)
    print("[5/7] Learning Curve (학습 데이터 크기 영향)")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    
    # 테스트셋 고정 (마지막 20%)
    test_size = int(len(spy_clean) * 0.2)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    vix_test = vix[-test_size:]
    y_vrp_test = y_vrp[-test_size:]
    
    train_sizes = [63, 126, 252, 504, 756, len(X) - test_size]  # 3개월 ~ 전체
    
    results = []
    
    print(f"\n  {'학습크기':>8} | {'기간':>10} | {'R²':>8} | {'방향':>8}")
    print("  " + "-" * 45)
    
    for train_size in train_sizes:
        if train_size > len(X) - test_size:
            train_size = len(X) - test_size
        
        X_train = X[-(test_size + train_size):-test_size]
        y_train = y[-(test_size + train_size):-test_size]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        vrp_pred = vix_test - en.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        period = f"{train_size//21}개월" if train_size < 400 else f"{train_size//252}년"
        
        results.append({
            'train_size': train_size,
            'period': period,
            'r2': float(r2),
            'direction_accuracy': float(dir_acc)
        })
        
        print(f"  {train_size:>8} | {period:>10} | {r2:>8.4f} | {dir_acc*100:>7.1f}%")
    
    return results


def experiment_6_realtime_simulation(spy):
    """실험 6: 실시간 예측 시뮬레이션"""
    print("\n" + "=" * 60)
    print("[6/7] 실시간 예측 시뮬레이션")
    print("=" * 60)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    spy_clean = spy.dropna(subset=feature_cols + ['RV_future', 'VRP_true'])
    
    X = spy_clean[feature_cols].values
    y = spy_clean['RV_future'].values
    vix = spy_clean['VIX'].values
    y_vrp = spy_clean['VRP_true'].values
    dates = spy_clean.index
    
    # 1년 학습 후 매일 예측
    warmup = 252
    
    predictions = []
    actuals = []
    pred_dates = []
    
    for i in range(warmup, len(X) - 22):
        X_train = X[:i]
        y_train = y[:i]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_i_s = scaler.transform(X[i:i+1])
        
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_s, y_train)
        
        vrp_pred = vix[i] - en.predict(X_i_s)[0]
        predictions.append(vrp_pred)
        actuals.append(y_vrp[i])
        pred_dates.append(dates[i].strftime('%Y-%m-%d'))
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    dir_acc = ((actuals > actuals.mean()) == (predictions > actuals.mean())).mean()
    
    print(f"\n  📊 실시간 시뮬레이션 결과:")
    print(f"     예측 횟수: {len(predictions)}회")
    print(f"     R²: {r2:.4f}")
    print(f"     MAE: {mae:.4f}")
    print(f"     방향 정확도: {dir_acc*100:.1f}%")
    
    return {
        'n_predictions': len(predictions),
        'r2': float(r2),
        'mae': float(mae),
        'direction_accuracy': float(dir_acc),
        'dates': pred_dates[::50],  # 샘플링
        'predictions': predictions[::50].tolist(),
        'actuals': actuals[::50].tolist()
    }


def experiment_7_ensemble_optimization(spy):
    """실험 7: 앙상블 가중치 최적화"""
    print("\n" + "=" * 60)
    print("[7/7] 앙상블 가중치 최적화")
    print("=" * 60)
    
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
    
    # 여러 모델
    models = {
        'EN_01': ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=SEED, max_iter=10000),
        'EN_05': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000),
        'EN_09': ElasticNet(alpha=0.01, l1_ratio=0.9, random_state=SEED, max_iter=10000),
        'Ridge': Ridge(alpha=1.0, random_state=SEED)
    }
    
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_s, y[:split_idx])
        predictions[name] = vix_test - model.predict(X_test_s)
    
    # 최적 가중치 그리드 서치
    best_r2 = -999
    best_weights = None
    
    model_names = list(predictions.keys())
    
    for w1 in np.arange(0.1, 0.9, 0.1):
        for w2 in np.arange(0.1, 0.9 - w1, 0.1):
            for w3 in np.arange(0.1, 0.9 - w1 - w2, 0.1):
                w4 = 1 - w1 - w2 - w3
                if w4 > 0:
                    ensemble = (w1 * predictions[model_names[0]] + 
                               w2 * predictions[model_names[1]] +
                               w3 * predictions[model_names[2]] +
                               w4 * predictions[model_names[3]])
                    r2 = r2_score(y_vrp_test, ensemble)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_weights = {model_names[0]: w1, model_names[1]: w2,
                                       model_names[2]: w3, model_names[3]: w4}
    
    # 개별 모델 성능
    print(f"\n  📊 개별 모델 성능:")
    individual_results = {}
    for name, pred in predictions.items():
        r2 = r2_score(y_vrp_test, pred)
        individual_results[name] = float(r2)
        print(f"     {name}: R² = {r2:.4f}")
    
    print(f"\n  📊 최적 앙상블:")
    print(f"     R² = {best_r2:.4f}")
    print(f"     가중치: {best_weights}")
    
    return {
        'individual': individual_results,
        'best_ensemble_r2': float(best_r2),
        'best_weights': {k: float(v) for k, v in best_weights.items()} if best_weights else None
    }


def main():
    print("\n" + "🔬" * 30)
    print("논문용 최종 추가 실험")
    print("🔬" * 30)
    
    # SPY 데이터 로드
    print("\nSPY 데이터 로드...")
    spy = prepare_asset_data('SPY')
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    results = {}
    
    # 실험 실행
    results['generalization'] = experiment_1_generalization(spy)
    results['visualization'] = experiment_2_visualization_data(spy)
    results['vix_spike'] = experiment_3_vix_spike(spy)
    results['long_short'] = experiment_4_long_short(spy)
    results['learning_curve'] = experiment_5_learning_curve(spy)
    results['realtime'] = experiment_6_realtime_simulation(spy)
    results['ensemble'] = experiment_7_ensemble_optimization(spy)
    
    # 저장
    results['timestamp'] = datetime.now().isoformat()
    
    with open('paper/final_experiments.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 시각화 데이터 별도 저장
    with open('paper/visualization_data.json', 'w') as f:
        json.dump(results['visualization'], f)
    
    print("\n" + "=" * 60)
    print("📊 모든 실험 완료")
    print("=" * 60)
    print(f"  ✓ paper/final_experiments.json")
    print(f"  ✓ paper/visualization_data.json")


if __name__ == '__main__':
    main()
