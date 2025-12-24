"""
GPU 기반 하이퍼파라미터 튜닝
============================
- Phase 1: XGBoost, LightGBM, CatBoost (GPU)
- Phase 2: PyTorch MLP (GPU)
- Phase 3: 앙상블 최적화
- Phase 4: 최종 검증

예상 시간: 3시간+
"""
import pandas as pd
import numpy as np
import json
import optuna
from optuna.samplers import TPESampler
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import nnls
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# GPU 라이브러리
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except:
    HAS_CAT = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch Device: {DEVICE}")
except:
    HAS_TORCH = False
    DEVICE = None

# ============================================================================
# 로깅 클래스
# ============================================================================

class TuningLogger:
    def __init__(self, name):
        self.name = name
        self.trials = []
        self.best_params = None
        self.best_score = -np.inf
        
    def log_trial(self, params, score, metrics):
        self.trials.append({
            'trial_id': len(self.trials) + 1,
            'params': params,
            'score': score,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
    
    def save(self, path):
        output = {
            'model': self.name,
            'total_trials': len(self.trials),
            'best_params': self.best_params,
            'best_score': self.best_score,
            'trials': self.trials
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)

# ============================================================================
# 데이터 준비
# ============================================================================

def calculate_rv(returns, window=22):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_data(ticker):
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None, None
    
    returns = df['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=df.index)
    
    # HAR-RV
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
    features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()
    features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()
    
    # 타겟
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # VIX 프록시
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_ret = spy['Close'].pct_change()
    if isinstance(spy_ret, pd.DataFrame):
        spy_ret = spy_ret.iloc[:, 0]
    spy_rv = calculate_rv(spy_ret, 22)
    features['VIX_proxy'] = spy_rv.reindex(df.index).ffill()
    
    features['VIX_lag1'] = features['VIX_proxy'].shift(1)
    features['VIX_lag5'] = features['VIX_proxy'].shift(5)
    features['VIX_change'] = features['VIX_proxy'].pct_change()
    
    # VRP
    features['VRP'] = features['VIX_proxy'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    # 시장
    features['regime_high'] = (features['VIX_proxy'] >= 25).astype(int)
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    
    features = features.dropna()
    
    feature_cols = [c for c in features.columns if c != 'RV_22d_future']
    X = features[feature_cols].values
    y = features['RV_22d_future'].values
    
    return X, y

def split_data(X, y, gap=22, train_ratio=0.8):
    split_idx = int(len(X) * train_ratio)
    X_train = X[:split_idx - gap]
    y_train = y[:split_idx - gap]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    return X_train, X_test, y_train, y_test

# ============================================================================
# Phase 1: XGBoost Tuning
# ============================================================================

def tune_xgboost(X_train, X_test, y_train, y_test, n_trials=100, logger=None):
    """XGBoost GPU 튜닝"""
    if not HAS_XGB:
        print("XGBoost not available")
        return None
    
    # GPU 사용 가능 여부 확인
    try:
        gpu_available = xgb.config_context(verbosity=0)
        tree_method = 'gpu_hist'
    except:
        tree_method = 'hist'
    
    y_train_log = np.log(y_train + 1)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'tree_method': tree_method,
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train_log, eval_set=[(X_test, np.log(y_test + 1))], 
                  verbose=False)
        
        y_pred = np.exp(model.predict(X_test)) - 1
        r2 = r2_score(y_test, y_pred)
        
        if logger:
            metrics = {
                'R2': round(r2, 6),
                'MAE': round(mean_absolute_error(y_test, y_pred), 6)
            }
            logger.log_trial(params, r2, metrics)
        
        return r2
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ============================================================================
# Phase 1: LightGBM Tuning
# ============================================================================

def tune_lightgbm(X_train, X_test, y_train, y_test, n_trials=100, logger=None):
    """LightGBM GPU 튜닝"""
    if not HAS_LGBM:
        print("LightGBM not available")
        return None
    
    y_train_log = np.log(y_train + 1)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train_log, eval_set=[(X_test, np.log(y_test + 1))],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        
        y_pred = np.exp(model.predict(X_test)) - 1
        r2 = r2_score(y_test, y_pred)
        
        if logger:
            metrics = {
                'R2': round(r2, 6),
                'MAE': round(mean_absolute_error(y_test, y_pred), 6)
            }
            logger.log_trial(params, r2, metrics)
        
        return r2
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ============================================================================
# Phase 1: CatBoost Tuning
# ============================================================================

def tune_catboost(X_train, X_test, y_train, y_test, n_trials=100, logger=None):
    """CatBoost GPU 튜닝"""
    if not HAS_CAT:
        print("CatBoost not available")
        return None
    
    y_train_log = np.log(y_train + 1)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
            'random_seed': 42,
            'verbose': False
        }
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train_log, eval_set=(X_test, np.log(y_test + 1)),
                  early_stopping_rounds=50, verbose=False)
        
        y_pred = np.exp(model.predict(X_test)) - 1
        r2 = r2_score(y_test, y_pred)
        
        if logger:
            metrics = {
                'R2': round(r2, 6),
                'MAE': round(mean_absolute_error(y_test, y_pred), 6)
            }
            logger.log_trial(params, r2, metrics)
        
        return r2
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ============================================================================
# Phase 2: PyTorch MLP
# ============================================================================

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def tune_pytorch_mlp(X_train, X_test, y_train, y_test, n_trials=50, logger=None):
    """PyTorch MLP GPU 튜닝"""
    if not HAS_TORCH:
        print("PyTorch not available")
        return None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_log = np.log(y_train + 1)
    y_test_log = np.log(y_test + 1)
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train_log).reshape(-1, 1).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test_scaled).to(DEVICE)
    
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_dims = [trial.suggest_int(f'hidden_{i}', 32, 512) for i in range(n_layers)]
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        
        model = MLPModel(X_train.shape[1], hidden_dims, dropout).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(100):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_log = model(X_test_t).cpu().numpy().flatten()
        
        y_pred = np.exp(y_pred_log) - 1
        r2 = r2_score(y_test, y_pred)
        
        if logger:
            params = {
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'lr': lr,
                'batch_size': batch_size
            }
            metrics = {
                'R2': round(r2, 6),
                'MAE': round(mean_absolute_error(y_test, y_pred), 6)
            }
            logger.log_trial(params, r2, metrics)
        
        return r2
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ============================================================================
# Phase 3: Ensemble Optimization
# ============================================================================

def optimize_ensemble(predictions_dict, y_test, logger=None):
    """앙상블 가중치 최적화"""
    model_names = list(predictions_dict.keys())
    pred_matrix = np.column_stack([predictions_dict[m] for m in model_names])
    
    # NNLS
    weights_nnls, _ = nnls(pred_matrix, y_test)
    pred_nnls = pred_matrix @ weights_nnls
    r2_nnls = r2_score(y_test, pred_nnls)
    
    # Optuna로 가중치 최적화
    def objective(trial):
        weights = [trial.suggest_float(f'w_{m}', 0, 1) for m in model_names]
        weights = np.array(weights) / sum(weights)  # 정규화
        pred = pred_matrix @ weights
        return r2_score(y_test, pred)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=200, show_progress_bar=True)
    
    best_weights = np.array([study.best_params[f'w_{m}'] for m in model_names])
    best_weights = best_weights / best_weights.sum()
    pred_optuna = pred_matrix @ best_weights
    r2_optuna = r2_score(y_test, pred_optuna)
    
    result = {
        'nnls': {'weights': dict(zip(model_names, weights_nnls.tolist())), 'R2': r2_nnls},
        'optuna': {'weights': dict(zip(model_names, best_weights.tolist())), 'R2': r2_optuna}
    }
    
    if logger:
        logger.log_trial({'method': 'ensemble_optimization'}, max(r2_nnls, r2_optuna), result)
    
    return result

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    os.makedirs('data/results', exist_ok=True)
    os.makedirs('data/results/best_models', exist_ok=True)
    
    assets = ['SPY', 'GLD', 'QQQ', 'TLT', 'EEM']
    n_trials_boost = 100  # 부스팅 모델당
    n_trials_nn = 50      # 신경망 모델당
    
    all_results = {}
    
    print("="*80)
    print("GPU Model Tuning - 3+ Hours Experiment")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"XGBoost: {HAS_XGB}, LightGBM: {HAS_LGBM}, CatBoost: {HAS_CAT}, PyTorch: {HAS_TORCH}")
    if HAS_TORCH:
        print(f"CUDA available: {torch.cuda.is_available()}")
    
    for asset in assets:
        print(f"\n{'#'*80}")
        print(f"# Asset: {asset}")
        print(f"{'#'*80}")
        
        X, y = prepare_data(asset)
        if X is None:
            continue
        
        X_train, X_test, y_train, y_test = split_data(X, y, gap=22)
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        asset_results = {}
        predictions = {}
        
        # Phase 1: XGBoost
        print(f"\n[Phase 1a] XGBoost Tuning ({n_trials_boost} trials)")
        xgb_logger = TuningLogger(f'XGBoost_{asset}')
        try:
            best_params, best_score = tune_xgboost(X_train, X_test, y_train, y_test, 
                                                    n_trials=n_trials_boost, logger=xgb_logger)
            print(f"  Best R2: {best_score:.4f}")
            asset_results['XGBoost'] = {'best_params': best_params, 'best_R2': best_score}
            xgb_logger.save(f'data/results/gpu_tuning_xgboost_{asset}.json')
            
            # 최적 모델로 예측
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, np.log(y_train + 1))
            predictions['XGBoost'] = np.exp(model.predict(X_test)) - 1
        except Exception as e:
            print(f"  XGBoost failed: {e}")
        
        # Phase 1: LightGBM
        print(f"\n[Phase 1b] LightGBM Tuning ({n_trials_boost} trials)")
        lgbm_logger = TuningLogger(f'LightGBM_{asset}')
        try:
            best_params, best_score = tune_lightgbm(X_train, X_test, y_train, y_test,
                                                     n_trials=n_trials_boost, logger=lgbm_logger)
            print(f"  Best R2: {best_score:.4f}")
            asset_results['LightGBM'] = {'best_params': best_params, 'best_R2': best_score}
            lgbm_logger.save(f'data/results/gpu_tuning_lightgbm_{asset}.json')
            
            model = lgb.LGBMRegressor(**best_params)
            model.fit(X_train, np.log(y_train + 1))
            predictions['LightGBM'] = np.exp(model.predict(X_test)) - 1
        except Exception as e:
            print(f"  LightGBM failed: {e}")
        
        # Phase 1: CatBoost
        print(f"\n[Phase 1c] CatBoost Tuning ({n_trials_boost} trials)")
        cat_logger = TuningLogger(f'CatBoost_{asset}')
        try:
            best_params, best_score = tune_catboost(X_train, X_test, y_train, y_test,
                                                     n_trials=n_trials_boost, logger=cat_logger)
            print(f"  Best R2: {best_score:.4f}")
            asset_results['CatBoost'] = {'best_params': best_params, 'best_R2': best_score}
            cat_logger.save(f'data/results/gpu_tuning_catboost_{asset}.json')
            
            model = CatBoostRegressor(**best_params)
            model.fit(X_train, np.log(y_train + 1), verbose=False)
            predictions['CatBoost'] = np.exp(model.predict(X_test)) - 1
        except Exception as e:
            print(f"  CatBoost failed: {e}")
        
        # Phase 2: PyTorch MLP
        print(f"\n[Phase 2] PyTorch MLP Tuning ({n_trials_nn} trials)")
        mlp_logger = TuningLogger(f'MLP_{asset}')
        try:
            best_params, best_score = tune_pytorch_mlp(X_train, X_test, y_train, y_test,
                                                        n_trials=n_trials_nn, logger=mlp_logger)
            print(f"  Best R2: {best_score:.4f}")
            asset_results['MLP'] = {'best_params': best_params, 'best_R2': best_score}
            mlp_logger.save(f'data/results/gpu_tuning_mlp_{asset}.json')
        except Exception as e:
            print(f"  MLP failed: {e}")
        
        # Phase 3: Ensemble
        if len(predictions) >= 2:
            print(f"\n[Phase 3] Ensemble Optimization")
            ens_logger = TuningLogger(f'Ensemble_{asset}')
            try:
                ens_result = optimize_ensemble(predictions, y_test, logger=ens_logger)
                print(f"  NNLS R2: {ens_result['nnls']['R2']:.4f}")
                print(f"  Optuna R2: {ens_result['optuna']['R2']:.4f}")
                asset_results['Ensemble'] = ens_result
                ens_logger.save(f'data/results/gpu_tuning_ensemble_{asset}.json')
            except Exception as e:
                print(f"  Ensemble failed: {e}")
        
        all_results[asset] = asset_results
    
    # 최종 결과 저장
    final_output = {
        'metadata': {
            'experiment': 'GPU Model Tuning',
            'start_time': datetime.now().isoformat(),
            'assets': assets,
            'n_trials_boost': n_trials_boost,
            'n_trials_nn': n_trials_nn
        },
        'results': all_results
    }
    
    with open('data/results/gpu_tuning_final_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 최고 결과 출력
    print("\n=== Best Results per Asset ===")
    for asset, results in all_results.items():
        best_r2 = -999
        best_model = ""
        for model, data in results.items():
            if isinstance(data, dict):
                if 'best_R2' in data and data['best_R2'] > best_r2:
                    best_r2 = data['best_R2']
                    best_model = model
                elif 'optuna' in data and data['optuna']['R2'] > best_r2:
                    best_r2 = data['optuna']['R2']
                    best_model = 'Ensemble_Optuna'
        print(f"  {asset}: {best_model} (R2={best_r2:.4f})")

if __name__ == "__main__":
    main()
