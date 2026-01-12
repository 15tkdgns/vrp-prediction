#!/usr/bin/env python3
"""
모델 벤치마킹 비교 검증
======================

1. 전통 변동성 모델 (GARCH, EWMA, HAR-RV)
2. 선형 모델 (OLS, Ridge, Lasso, ElasticNet)
3. 트리 기반 모델 (RF, GB, XGBoost, LightGBM, CatBoost)
4. 딥러닝 모델 (MLP, LSTM, GRU)
5. 앙상블 모델 (Stacking, Voting)
6. 통계 검정 및 순위
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
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
    import catboost as cb
    HAS_CB = True
except:
    HAS_CB = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(SEED)
    HAS_TF = True
except:
    HAS_TF = False

try:
    from arch import arch_model
    HAS_ARCH = True
except:
    HAS_ARCH = False


def load_data():
    """데이터 로드"""
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


def benchmark_1_traditional(spy, X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test):
    """전통 변동성 모델"""
    print("\n  📊 1. 전통 변동성 모델")
    print("  " + "-" * 50)
    
    results = {}
    
    # EWMA
    lambda_param = 0.94
    returns = spy['returns'].values * 100
    
    ewma_var = np.zeros(len(returns))
    ewma_var[0] = returns[0]**2
    
    for i in range(1, len(returns)):
        ewma_var[i] = lambda_param * ewma_var[i-1] + (1 - lambda_param) * returns[i-1]**2
    
    ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)
    
    split_idx = len(spy) - len(y_vrp_test)
    ewma_test = ewma_vol[split_idx:]
    vrp_pred_ewma = vix_test - ewma_test
    
    r2_ewma = r2_score(y_vrp_test, vrp_pred_ewma)
    mae_ewma = mean_absolute_error(y_vrp_test, vrp_pred_ewma)
    dir_ewma = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred_ewma > y_vrp_test.mean())).mean()
    
    results['EWMA'] = {'r2': r2_ewma, 'mae': mae_ewma, 'direction': dir_ewma}
    print(f"     EWMA:        R² = {r2_ewma:>8.4f} | MAE = {mae_ewma:>6.2f} | 방향 = {dir_ewma*100:>5.1f}%")
    
    # HAR-RV
    har_features = ['RV_1d', 'RV_5d', 'RV_22d']
    har_idx = [['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                'regime_high', 'return_5d', 'return_22d'].index(f) for f in har_features]
    
    X_train_har = X_train_s[:, har_idx]
    X_test_har = X_test_s[:, har_idx]
    
    har = LinearRegression()
    har.fit(X_train_har, y_train)
    vrp_pred_har = vix_test - har.predict(X_test_har)
    
    r2_har = r2_score(y_vrp_test, vrp_pred_har)
    mae_har = mean_absolute_error(y_vrp_test, vrp_pred_har)
    dir_har = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred_har > y_vrp_test.mean())).mean()
    
    results['HAR-RV'] = {'r2': r2_har, 'mae': mae_har, 'direction': dir_har}
    print(f"     HAR-RV:      R² = {r2_har:>8.4f} | MAE = {mae_har:>6.2f} | 방향 = {dir_har*100:>5.1f}%")
    
    # GARCH
    if HAS_ARCH:
        try:
            returns_train = spy['returns'].values[:len(y_train)] * 100
            garch = arch_model(returns_train, vol='Garch', p=1, q=1)
            garch_fit = garch.fit(disp='off')
            
            all_returns = spy['returns'].values * 100
            garch_full = arch_model(all_returns, vol='Garch', p=1, q=1)
            garch_full_fit = garch_full.fit(disp='off')
            
            garch_vol = garch_full_fit.conditional_volatility[-len(y_vrp_test):] * np.sqrt(252)
            vrp_pred_garch = vix_test - garch_vol
            
            r2_garch = r2_score(y_vrp_test, vrp_pred_garch)
            mae_garch = mean_absolute_error(y_vrp_test, vrp_pred_garch)
            dir_garch = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred_garch > y_vrp_test.mean())).mean()
            
            results['GARCH(1,1)'] = {'r2': r2_garch, 'mae': mae_garch, 'direction': dir_garch}
            print(f"     GARCH(1,1):  R² = {r2_garch:>8.4f} | MAE = {mae_garch:>6.2f} | 방향 = {dir_garch*100:>5.1f}%")
        except:
            print(f"     GARCH(1,1):  오류")
    
    return results


def benchmark_2_linear(X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test):
    """선형 모델"""
    print("\n  📊 2. 선형 모델")
    print("  " + "-" * 50)
    
    models = {
        'OLS': LinearRegression(),
        'Ridge (α=0.1)': Ridge(alpha=0.1, random_state=SEED),
        'Ridge (α=1.0)': Ridge(alpha=1.0, random_state=SEED),
        'Ridge (α=10)': Ridge(alpha=10.0, random_state=SEED),
        'Lasso': Lasso(alpha=0.01, random_state=SEED, max_iter=10000),
        'ElasticNet (0.1)': ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000),
        'ElasticNet (0.5)': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=SEED, max_iter=10000),
        'ElasticNet (0.9)': ElasticNet(alpha=1.0, l1_ratio=0.9, random_state=SEED, max_iter=10000),
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        vrp_pred = vix_test - model.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        mae = mean_absolute_error(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results[name] = {'r2': r2, 'mae': mae, 'direction': dir_acc}
        print(f"     {name:<18} R² = {r2:>8.4f} | MAE = {mae:>6.2f} | 방향 = {dir_acc*100:>5.1f}%")
    
    return results


def benchmark_3_tree(X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test):
    """트리 기반 모델"""
    print("\n  📊 3. 트리 기반 모델")
    print("  " + "-" * 50)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED),
    }
    
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, 
                                              random_state=SEED, verbosity=0)
    
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, 
                                                random_state=SEED, verbosity=-1)
    
    if HAS_CB:
        models['CatBoost'] = cb.CatBoostRegressor(iterations=100, depth=4, learning_rate=0.05, 
                                                   random_state=SEED, verbose=False)
    
    results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train)
            vrp_pred = vix_test - model.predict(X_test_s)
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            mae = mean_absolute_error(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            results[name] = {'r2': r2, 'mae': mae, 'direction': dir_acc}
            print(f"     {name:<18} R² = {r2:>8.4f} | MAE = {mae:>6.2f} | 방향 = {dir_acc*100:>5.1f}%")
        except Exception as e:
            print(f"     {name:<18} 오류: {str(e)[:30]}")
    
    return results


def benchmark_4_deep_learning(spy, X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test):
    """딥러닝 모델"""
    print("\n  📊 4. 딥러닝 모델")
    print("  " + "-" * 50)
    
    results = {}
    
    # MLP
    mlp_configs = [
        ('MLP (64)', (64,)),
        ('MLP (128,64)', (128, 64)),
        ('MLP (256,128,64)', (256, 128, 64)),
    ]
    
    for name, hidden_layers in mlp_configs:
        try:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=500, 
                              random_state=SEED, early_stopping=True)
            mlp.fit(X_train_s, y_train)
            vrp_pred = vix_test - mlp.predict(X_test_s)
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            mae = mean_absolute_error(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            results[name] = {'r2': r2, 'mae': mae, 'direction': dir_acc}
            print(f"     {name:<18} R² = {r2:>8.4f} | MAE = {mae:>6.2f} | 방향 = {dir_acc*100:>5.1f}%")
        except Exception as e:
            print(f"     {name:<18} 오류: {str(e)[:30]}")
    
    # LSTM/GRU
    if HAS_TF:
        feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                       'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                       'regime_high', 'return_5d', 'return_22d']
        
        X_full = spy[feature_cols].values
        y_full = spy['RV_future'].values
        vol_full = spy['VIX'].values
        y_vrp_full = spy['VRP_true'].values
        
        split_idx = len(X_full) - len(y_vrp_test)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X_full)
        y_scaled = scaler_y.fit_transform(y_full.reshape(-1, 1)).flatten()
        
        seq_length = 22
        
        def create_sequences(X, y, seq_len):
            Xs, ys = [], []
            for i in range(len(X) - seq_len):
                Xs.append(X[i:i+seq_len])
                ys.append(y[i+seq_len])
            return np.array(Xs), np.array(ys)
        
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
        
        train_idx = split_idx - seq_length
        X_train_seq = X_seq[:train_idx]
        y_train_seq = y_seq[:train_idx]
        X_test_seq = X_seq[train_idx:]
        y_test_seq = y_seq[train_idx:]
        
        vol_test_seq = vol_full[seq_length + train_idx:]
        y_vrp_test_seq = y_vrp_full[seq_length + train_idx:]
        
        rnn_configs = [
            ('LSTM (64)', LSTM, [64]),
            ('LSTM (64,32)', LSTM, [64, 32]),
            ('GRU (64)', GRU, [64]),
        ]
        
        for name, layer_type, units in rnn_configs:
            try:
                model = Sequential()
                model.add(layer_type(units[0], input_shape=(seq_length, len(feature_cols)),
                                    return_sequences=(len(units) > 1)))
                if len(units) > 1:
                    for u in units[1:]:
                        model.add(layer_type(u))
                model.add(Dropout(0.2))
                model.add(Dense(1))
                
                model.compile(optimizer='adam', loss='mse')
                early_stop = EarlyStopping(patience=10, restore_best_weights=True)
                
                model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32,
                         validation_split=0.2, callbacks=[early_stop], verbose=0)
                
                y_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                vrp_pred = vol_test_seq[:len(y_pred)] - y_pred
                y_vrp_actual = y_vrp_test_seq[:len(y_pred)]
                
                r2 = r2_score(y_vrp_actual, vrp_pred)
                mae = mean_absolute_error(y_vrp_actual, vrp_pred)
                dir_acc = ((y_vrp_actual > y_vrp_actual.mean()) == (vrp_pred > y_vrp_actual.mean())).mean()
                
                results[name] = {'r2': r2, 'mae': mae, 'direction': dir_acc}
                print(f"     {name:<18} R² = {r2:>8.4f} | MAE = {mae:>6.2f} | 방향 = {dir_acc*100:>5.1f}%")
                
            except Exception as e:
                print(f"     {name:<18} 오류: {str(e)[:30]}")
    
    return results


def benchmark_5_ensemble(X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test):
    """앙상블 모델"""
    print("\n  📊 5. 앙상블 모델")
    print("  " + "-" * 50)
    
    results = {}
    
    # 기본 모델
    en = ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)
    ridge = Ridge(alpha=1.0, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=SEED, n_jobs=-1)
    
    # Voting Ensemble
    try:
        voting = VotingRegressor([('en', en), ('ridge', ridge), ('rf', rf)])
        voting.fit(X_train_s, y_train)
        vrp_pred = vix_test - voting.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        mae = mean_absolute_error(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results['Voting (EN+Ridge+RF)'] = {'r2': r2, 'mae': mae, 'direction': dir_acc}
        print(f"     {'Voting (EN+Ridge+RF)':<22} R² = {r2:>8.4f} | MAE = {mae:>6.2f} | 방향 = {dir_acc*100:>5.1f}%")
    except Exception as e:
        print(f"     Voting: 오류")
    
    # Stacking Ensemble
    try:
        stacking = StackingRegressor(
            estimators=[('en', ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=SEED, max_iter=10000)),
                       ('ridge', Ridge(alpha=1.0, random_state=SEED))],
            final_estimator=Ridge(alpha=0.1, random_state=SEED)
        )
        stacking.fit(X_train_s, y_train)
        vrp_pred = vix_test - stacking.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        mae = mean_absolute_error(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results['Stacking (EN+Ridge)'] = {'r2': r2, 'mae': mae, 'direction': dir_acc}
        print(f"     {'Stacking (EN+Ridge)':<22} R² = {r2:>8.4f} | MAE = {mae:>6.2f} | 방향 = {dir_acc*100:>5.1f}%")
    except Exception as e:
        print(f"     Stacking: 오류")
    
    return results


def statistical_tests(all_results, y_vrp_test):
    """통계 검정"""
    print("\n  📊 6. 통계 검정 및 순위")
    print("  " + "-" * 50)
    
    # R² 순위
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    print("\n     🏆 R² 순위 (Top 10):")
    for i, (name, metrics) in enumerate(sorted_results[:10], 1):
        print(f"     {i:>2}. {name:<22} R² = {metrics['r2']:>8.4f}")
    
    # 방향 정확도 순위
    sorted_by_dir = sorted(all_results.items(), key=lambda x: x[1]['direction'], reverse=True)
    
    print("\n     🏆 방향 정확도 순위 (Top 10):")
    for i, (name, metrics) in enumerate(sorted_by_dir[:10], 1):
        print(f"     {i:>2}. {name:<22} 방향 = {metrics['direction']*100:>5.1f}%")
    
    return sorted_results


def main():
    print("\n" + "🔬" * 30)
    print("모델 벤치마킹 비교 검증")
    print("🔬" * 30)
    
    # 데이터 로드
    print("\n데이터 로드...")
    spy = load_data()
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
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
    
    print(f"  ✓ 학습: {len(X_train)}, 테스트: {len(X_test)}")
    
    # 벤치마킹
    print("\n" + "=" * 70)
    print("📊 벤치마킹 결과")
    print("=" * 70)
    
    all_results = {}
    
    all_results.update(benchmark_1_traditional(spy, X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test))
    all_results.update(benchmark_2_linear(X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test))
    all_results.update(benchmark_3_tree(X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test))
    all_results.update(benchmark_4_deep_learning(spy, X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test))
    all_results.update(benchmark_5_ensemble(X_train_s, X_test_s, y_train, y_test, vix_test, y_vrp_test))
    
    sorted_results = statistical_tests(all_results, y_vrp_test)
    
    # 저장
    results_for_save = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}
    results_for_save['timestamp'] = datetime.now().isoformat()
    
    with open('paper/model_benchmark.json', 'w') as f:
        json.dump(results_for_save, f, indent=2)
    
    # 최종 요약
    print("\n" + "=" * 70)
    print("📊 최종 요약")
    print("=" * 70)
    
    best_r2 = sorted_results[0]
    best_dir = max(all_results.items(), key=lambda x: x[1]['direction'])
    
    print(f"""
    🏆 최고 R² 모델: {best_r2[0]}
       R² = {best_r2[1]['r2']:.4f}
       방향 = {best_r2[1]['direction']*100:.1f}%
    
    🏆 최고 방향 정확도: {best_dir[0]}
       방향 = {best_dir[1]['direction']*100:.1f}%
       R² = {best_dir[1]['r2']:.4f}
    
    📊 총 {len(all_results)}개 모델 비교 완료
    """)
    
    print(f"\n💾 결과 저장: paper/model_benchmark.json")


if __name__ == '__main__':
    main()
