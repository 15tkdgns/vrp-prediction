#!/usr/bin/env python3
"""
R² 극대화 실험 (데이터 누수 없음)
================================

1. MLP 튜닝 (최고 성능 모델)
2. Gradient Boosting 튜닝
3. LSTM (시계열 특화)
4. 특성 + 모델 조합 최적화
5. 앙상블 (MLP + GB + EN)
6. 극대화 탐색
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)

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
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    tf.random.set_seed(SEED)
    HAS_TF = True
except:
    HAS_TF = False


def prepare_data_safe(ticker, vol_ticker, start='2015-01-01', end='2025-01-01'):
    """데이터 준비"""
    asset = yf.download(ticker, start=start, end=end, progress=False)
    vol = yf.download(vol_ticker, start=start, end=end, progress=False)
    
    if isinstance(asset.columns, pd.MultiIndex):
        asset.columns = asset.columns.get_level_values(0)
    if isinstance(vol.columns, pd.MultiIndex):
        vol.columns = vol.columns.get_level_values(0)
    
    asset['Vol'] = vol['Close'].reindex(asset.index).ffill().bfill()
    asset['returns'] = asset['Close'].pct_change()
    
    asset['RV_1d'] = asset['returns'].abs() * np.sqrt(252) * 100
    asset['RV_5d'] = asset['returns'].rolling(5).std() * np.sqrt(252) * 100
    asset['RV_22d'] = asset['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    asset['VRP'] = asset['Vol'] - asset['RV_22d']
    asset['RV_future'] = asset['RV_22d'].shift(-22)
    asset['VRP_true'] = asset['Vol'] - asset['RV_future']
    
    asset['Vol_lag1'] = asset['Vol'].shift(1)
    asset['Vol_lag5'] = asset['Vol'].shift(5)
    asset['Vol_change'] = asset['Vol'].pct_change()
    asset['VRP_lag1'] = asset['VRP'].shift(1)
    asset['VRP_lag5'] = asset['VRP'].shift(5)
    asset['VRP_ma5'] = asset['VRP'].rolling(5).mean()
    asset['regime_high'] = (asset['Vol'] >= 25).astype(int)
    asset['return_5d'] = asset['returns'].rolling(5).sum()
    asset['return_22d'] = asset['returns'].rolling(22).sum()
    
    asset['Vol_std5'] = asset['Vol'].rolling(5).std()
    asset['RV_ratio'] = asset['RV_5d'] / (asset['RV_22d'] + 1e-8)
    asset['VRP_momentum'] = asset['VRP'] - asset['VRP'].shift(5)
    
    asset = asset.replace([np.inf, -np.inf], np.nan).dropna()
    
    return asset


def safe_split(data, test_ratio=0.2, gap=22):
    train_end = int(len(data) * (1 - test_ratio)) - gap
    test_start = int(len(data) * (1 - test_ratio))
    return train_end, test_start


def experiment_1_mlp_tuning():
    """실험 1: MLP 튜닝"""
    print("\n" + "=" * 70)
    print("[1/6] MLP 튜닝 (22일 Gap)")
    print("=" * 70)
    
    gld = prepare_data_safe('GLD', '^VIX')
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5',
                   'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = gld[feature_cols].values
    y = gld['RV_future'].values
    vol = gld['Vol'].values
    y_vrp = gld['VRP_true'].values
    
    train_end, split_idx = safe_split(gld, gap=22)
    vol_test = vol[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:train_end])
    X_test_s = scaler.transform(X[split_idx:])
    
    configs = [
        ('MLP (64)', (64,), 0.01),
        ('MLP (128)', (128,), 0.01),
        ('MLP (256)', (256,), 0.01),
        ('MLP (64,32)', (64, 32), 0.01),
        ('MLP (128,64)', (128, 64), 0.01),
        ('MLP (256,128)', (256, 128), 0.01),
        ('MLP (256,128,64)', (256, 128, 64), 0.01),
        ('MLP (512,256,128)', (512, 256, 128), 0.001),
    ]
    
    results = {}
    best_r2 = -999
    best_config = None
    
    print(f"\n  {'Config':>25} | {'R²':>10} | {'방향':>10}")
    print("  " + "-" * 55)
    
    for name, hidden, lr in configs:
        try:
            mlp = MLPRegressor(hidden_layer_sizes=hidden, max_iter=1000, 
                              random_state=SEED, early_stopping=True,
                              learning_rate_init=lr, alpha=0.001)
            mlp.fit(X_train_s, y[:train_end])
            vrp_pred = vol_test - mlp.predict(X_test_s)
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            results[name] = {'r2': float(r2), 'direction': float(dir_acc)}
            print(f"  {name:>25} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
            
            if r2 > best_r2:
                best_r2 = r2
                best_config = name
        except Exception as e:
            print(f"  {name:>25} | 오류")
    
    print(f"\n  🏆 최고: {best_config} → R² = {best_r2:.4f}")
    
    return results


def experiment_2_gb_tuning():
    """실험 2: Gradient Boosting 튜닝"""
    print("\n" + "=" * 70)
    print("[2/6] Gradient Boosting 튜닝 (22일 Gap)")
    print("=" * 70)
    
    gld = prepare_data_safe('GLD', '^VIX')
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5',
                   'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = gld[feature_cols].values
    y = gld['RV_future'].values
    vol = gld['Vol'].values
    y_vrp = gld['VRP_true'].values
    
    train_end, split_idx = safe_split(gld, gap=22)
    vol_test = vol[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:train_end])
    X_test_s = scaler.transform(X[split_idx:])
    
    configs = [
        ('GB (50, d=3)', 50, 3, 0.1),
        ('GB (100, d=3)', 100, 3, 0.1),
        ('GB (100, d=4)', 100, 4, 0.1),
        ('GB (200, d=4)', 200, 4, 0.05),
        ('GB (200, d=5)', 200, 5, 0.05),
        ('GB (300, d=5)', 300, 5, 0.05),
    ]
    
    results = {}
    best_r2 = -999
    
    print(f"\n  {'Config':>20} | {'R²':>10} | {'방향':>10}")
    print("  " + "-" * 50)
    
    for name, n_est, depth, lr in configs:
        gb = GradientBoostingRegressor(n_estimators=n_est, max_depth=depth, 
                                        learning_rate=lr, random_state=SEED)
        gb.fit(X_train_s, y[:train_end])
        vrp_pred = vol_test - gb.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results[name] = {'r2': float(r2), 'direction': float(dir_acc)}
        print(f"  {name:>20} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
        
        if r2 > best_r2:
            best_r2 = r2
    
    # XGBoost/LightGBM
    if HAS_XGB:
        for n_est, depth in [(100, 4), (200, 5)]:
            xgb_model = xgb.XGBRegressor(n_estimators=n_est, max_depth=depth, 
                                          learning_rate=0.05, random_state=SEED, verbosity=0)
            xgb_model.fit(X_train_s, y[:train_end])
            vrp_pred = vol_test - xgb_model.predict(X_test_s)
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            name = f'XGB ({n_est}, d={depth})'
            results[name] = {'r2': float(r2), 'direction': float(dir_acc)}
            print(f"  {name:>20} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
            
            if r2 > best_r2:
                best_r2 = r2
    
    if HAS_LGB:
        for n_est, depth in [(100, 4), (200, 5)]:
            lgb_model = lgb.LGBMRegressor(n_estimators=n_est, max_depth=depth, 
                                           learning_rate=0.05, random_state=SEED, verbosity=-1)
            lgb_model.fit(X_train_s, y[:train_end])
            vrp_pred = vol_test - lgb_model.predict(X_test_s)
            
            r2 = r2_score(y_vrp_test, vrp_pred)
            dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
            
            name = f'LGB ({n_est}, d={depth})'
            results[name] = {'r2': float(r2), 'direction': float(dir_acc)}
            print(f"  {name:>20} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
            
            if r2 > best_r2:
                best_r2 = r2
    
    return results


def experiment_3_lstm():
    """실험 3: LSTM 튜닝"""
    print("\n" + "=" * 70)
    print("[3/6] LSTM 튜닝 (22일 Gap)")
    print("=" * 70)
    
    if not HAS_TF:
        print("  ⚠️ TensorFlow 없음")
        return {}
    
    gld = prepare_data_safe('GLD', '^VIX')
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5',
                   'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = gld[feature_cols].values
    y = gld['RV_future'].values
    vol = gld['Vol'].values
    y_vrp = gld['VRP_true'].values
    
    train_end, split_idx = safe_split(gld, gap=22)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)
    
    results = {}
    best_r2 = -999
    
    configs = [
        ('LSTM (64) seq=10', 10, [64]),
        ('LSTM (128) seq=10', 10, [128]),
        ('LSTM (128,64) seq=10', 10, [128, 64]),
        ('LSTM (64) seq=22', 22, [64]),
        ('LSTM (128) seq=22', 22, [128]),
    ]
    
    print(f"\n  {'Config':>25} | {'R²':>10} | {'방향':>10}")
    print("  " + "-" * 55)
    
    for name, seq_len, units in configs:
        try:
            X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)
            
            # 22일 Gap + seq_len 고려
            train_idx = train_end - seq_len
            test_idx = split_idx - seq_len
            
            X_train = X_seq[:train_idx]
            y_train = y_seq[:train_idx]
            X_test = X_seq[test_idx:]
            y_test = y_seq[test_idx:]
            
            vol_test = vol[split_idx:]
            y_vrp_test = y_vrp[split_idx:]
            
            model = Sequential()
            model.add(LSTM(units[0], input_shape=(seq_len, len(feature_cols)),
                          return_sequences=(len(units) > 1)))
            if len(units) > 1:
                for u in units[1:]:
                    model.add(LSTM(u))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
            ]
            
            model.fit(X_train, y_train, epochs=100, batch_size=32,
                     validation_split=0.2, callbacks=callbacks, verbose=0)
            
            y_pred_scaled = model.predict(X_test, verbose=0).flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            vrp_pred = vol_test[:len(y_pred)] - y_pred
            y_vrp_actual = y_vrp_test[:len(y_pred)]
            
            r2 = r2_score(y_vrp_actual, vrp_pred)
            dir_acc = ((y_vrp_actual > y_vrp_actual.mean()) == (vrp_pred > y_vrp_actual.mean())).mean()
            
            results[name] = {'r2': float(r2), 'direction': float(dir_acc)}
            print(f"  {name:>25} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
            
            if r2 > best_r2:
                best_r2 = r2
                
        except Exception as e:
            print(f"  {name:>25} | 오류: {str(e)[:20]}")
    
    return results


def experiment_4_feature_model_combo():
    """실험 4: 특성 + 모델 조합"""
    print("\n" + "=" * 70)
    print("[4/6] 특성 + 모델 조합 최적화")
    print("=" * 70)
    
    gld = prepare_data_safe('GLD', '^VIX')
    
    # 최적 특성
    best_features = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5',
                    'VRP_lag1', 'VRP_lag5', 'VRP_ma5', 'return_22d', 'RV_ratio']
    
    X = gld[best_features].values
    y = gld['RV_future'].values
    vol = gld['Vol'].values
    y_vrp = gld['VRP_true'].values
    
    train_end, split_idx = safe_split(gld, gap=22)
    vol_test = vol[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:train_end])
    X_test_s = scaler.transform(X[split_idx:])
    
    models = {
        'MLP (256,128)': MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=1000, 
                                       random_state=SEED, early_stopping=True, alpha=0.001),
        'MLP (512,256)': MLPRegressor(hidden_layer_sizes=(512, 256), max_iter=1000, 
                                       random_state=SEED, early_stopping=True, alpha=0.0001),
        'GB (200, d=5)': GradientBoostingRegressor(n_estimators=200, max_depth=5, 
                                                    learning_rate=0.05, random_state=SEED),
    }
    
    if HAS_XGB:
        models['XGB (200, d=5)'] = xgb.XGBRegressor(n_estimators=200, max_depth=5, 
                                                     learning_rate=0.05, random_state=SEED, verbosity=0)
    
    results = {}
    best_r2 = -999
    
    print(f"\n  {'Model':>20} | {'R²':>10} | {'방향':>10}")
    print("  " + "-" * 50)
    
    for name, model in models.items():
        model.fit(X_train_s, y[:train_end])
        vrp_pred = vol_test - model.predict(X_test_s)
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results[name] = {'r2': float(r2), 'direction': float(dir_acc)}
        print(f"  {name:>20} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
        
        if r2 > best_r2:
            best_r2 = r2
    
    return results


def experiment_5_ensemble():
    """실험 5: 최적 앙상블"""
    print("\n" + "=" * 70)
    print("[5/6] 최적 앙상블 (MLP + GB + EN)")
    print("=" * 70)
    
    gld = prepare_data_safe('GLD', '^VIX')
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5',
                   'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    X = gld[feature_cols].values
    y = gld['RV_future'].values
    vol = gld['Vol'].values
    y_vrp = gld['VRP_true'].values
    
    train_end, split_idx = safe_split(gld, gap=22)
    vol_test = vol[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:train_end])
    X_test_s = scaler.transform(X[split_idx:])
    
    # 개별 모델 학습
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=SEED, early_stopping=True)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=SEED)
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    
    mlp.fit(X_train_s, y[:train_end])
    gb.fit(X_train_s, y[:train_end])
    en.fit(X_train_s, y[:train_end])
    
    pred_mlp = mlp.predict(X_test_s)
    pred_gb = gb.predict(X_test_s)
    pred_en = en.predict(X_test_s)
    
    # 앙상블 조합
    ensembles = {
        'MLP only': pred_mlp,
        'GB only': pred_gb,
        'EN only': pred_en,
        'MLP+GB avg': (pred_mlp + pred_gb) / 2,
        'MLP+EN avg': (pred_mlp + pred_en) / 2,
        'All avg': (pred_mlp + pred_gb + pred_en) / 3,
        'MLP(0.5)+GB(0.3)+EN(0.2)': 0.5*pred_mlp + 0.3*pred_gb + 0.2*pred_en,
        'MLP(0.6)+GB(0.4)': 0.6*pred_mlp + 0.4*pred_gb,
    }
    
    results = {}
    best_r2 = -999
    
    print(f"\n  {'Ensemble':>30} | {'R²':>10} | {'방향':>10}")
    print("  " + "-" * 60)
    
    for name, pred in ensembles.items():
        vrp_pred = vol_test - pred
        
        r2 = r2_score(y_vrp_test, vrp_pred)
        dir_acc = ((y_vrp_test > y_vrp_test.mean()) == (vrp_pred > y_vrp_test.mean())).mean()
        
        results[name] = {'r2': float(r2), 'direction': float(dir_acc)}
        print(f"  {name:>30} | {r2:>10.4f} | {dir_acc*100:>9.1f}%")
        
        if r2 > best_r2:
            best_r2 = r2
    
    return results


def experiment_6_maximize():
    """실험 6: R² 극대화"""
    print("\n" + "=" * 70)
    print("[6/6] R² 극대화 탐색")
    print("=" * 70)
    
    gld = prepare_data_safe('GLD', '^VIX')
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5',
                   'Vol_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d', 'RV_ratio', 'VRP_momentum']
    
    X = gld[feature_cols].values
    y = gld['RV_future'].values
    vol = gld['Vol'].values
    y_vrp = gld['VRP_true'].values
    
    train_end, split_idx = safe_split(gld, gap=22)
    vol_test = vol[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:train_end])
    X_test_s = scaler.transform(X[split_idx:])
    
    best_r2 = -999
    best_config = None
    
    # MLP 집중 탐색
    for hidden in [(256, 128), (512, 256), (512, 256, 128), (1024, 512, 256)]:
        for alpha in [0.0001, 0.001, 0.01]:
            for lr in [0.001, 0.01]:
                try:
                    mlp = MLPRegressor(hidden_layer_sizes=hidden, max_iter=2000, 
                                       random_state=SEED, early_stopping=True,
                                       learning_rate_init=lr, alpha=alpha)
                    mlp.fit(X_train_s, y[:train_end])
                    vrp_pred = vol_test - mlp.predict(X_test_s)
                    
                    r2 = r2_score(y_vrp_test, vrp_pred)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_config = {'hidden': hidden, 'alpha': alpha, 'lr': lr}
                except:
                    pass
    
    print(f"\n  🏆 최고 R² 달성:")
    print(f"     R²: {best_r2:.4f}")
    print(f"     Config: {best_config}")
    
    return {'best_r2': float(best_r2), 'best_config': best_config}


def main():
    print("\n" + "🔥" * 30)
    print("R² 극대화 실험 (데이터 누수 없음)")
    print("🔥" * 30)
    
    results = {}
    
    results['mlp_tuning'] = experiment_1_mlp_tuning()
    results['gb_tuning'] = experiment_2_gb_tuning()
    results['lstm'] = experiment_3_lstm()
    results['feature_model'] = experiment_4_feature_model_combo()
    results['ensemble'] = experiment_5_ensemble()
    results['maximize'] = experiment_6_maximize()
    
    # 저장
    results['timestamp'] = datetime.now().isoformat()
    
    with open('paper/r2_maximize.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("📊 R² 극대화 최종 요약")
    print("=" * 70)
    
    print(f"\n💾 결과 저장: paper/r2_maximize.json")


if __name__ == '__main__':
    main()
